# ============================================================
# AI INVENTORY DASHBOARD — Hosted Gradio Web App
# Modern Cards | Blue & White | In-page color + light/dark controls
# ============================================================
# Hosted on Render: upload this project to a GitHub repository, then create a Render Web Service.
# Upload CSV/Excel from the app page; uploaded data is handled during the current app session.
#
# คอลัมน์ที่แนะนำ:
# Sales: date, sku, demand
# Inventory: sku, on_hand, unit_cost, backorders
# รองรับชื่อคอลัมน์ภาษาไทย/อังกฤษหลายรูปแบบ โดยพยายามตรวจจับให้อัตโนมัติ
# หมายเหตุ: หากไม่มี backorders จะประเมินอัตราสินค้าขาดไม่ได้อย่างน่าเชื่อถือ
# และหากไม่มี unit_cost จะคำนวณต้นทุนสต็อกเป็น 0 จนกว่าจะเติมข้อมูล
# ============================================================

import os

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
from pathlib import Path
import tempfile
import html
import re
import warnings
warnings.filterwarnings("ignore")

APP_TITLE = "AI Inventory Dashboard"

# ---------- Column detection / data preparation ----------
ALIASES = {
    "date": ["date", "วันที่", "วัน", "order_date", "sales_date", "transaction_date", "วันที่ขาย"],
    "sku": ["sku", "item", "item_id", "product", "product_id", "รหัสสินค้า", "รหัส", "สินค้า", "รหัสผลิตภัณฑ์", "material"],
    "demand": ["demand", "sales", "qty_sold", "quantity_sold", "units_sold", "sales_qty", "จำนวนขาย", "ยอดขาย", "ความต้องการ", "ปริมาณความต้องการ", "จำนวนที่ขาย"],
    "on_hand": ["on_hand", "stock", "inventory", "stock_qty", "quantity_on_hand", "current_stock", "คงเหลือ", "สินค้าคงคลัง", "จำนวนคงเหลือ", "สต็อก"],
    "unit_cost": ["unit_cost", "cost", "price", "cost_per_unit", "unit_price", "ต้นทุนต่อหน่วย", "ต้นทุน", "ราคาต่อหน่วย"],
    "backorders": ["backorders", "stockout", "shortage", "lost_sales", "ขาดสต็อก", "สินค้าขาด", "จำนวนขาด", "ยอดค้างส่ง"],
    "lead_time": ["lead_time", "leadtime", "ระยะเวลานำ", "ระยะเวลาจัดส่ง", "วันนำ"],
}

def _key(x):
    return re.sub(r"[^a-z0-9ก-๙]+", "", str(x).strip().lower())

_ALIAS_LOOKUP = {_key(alias): canonical for canonical, names in ALIASES.items() for alias in [canonical] + names}

def standardize_columns(df):
    df = df.copy()
    rename = {}
    for col in df.columns:
        k = _key(col)
        if k in _ALIAS_LOOKUP:
            rename[col] = _ALIAS_LOOKUP[k]
    df = df.rename(columns=rename)
    # Keep the first recognized column if duplicate aliases mapped to the same canonical name.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def read_one_file(file_path):
    path = str(file_path)
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp874")
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path)
    if ext == ".xls":
        raise ValueError("ไฟล์ .xls รุ่นเก่าไม่รองรับโดยตรง กรุณาบันทึกเป็น .xlsx หรือ .csv")
    raise ValueError(f"ไม่รองรับไฟล์นามสกุล {ext}")

def _numeric(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def normalize_data(raw):
    """Standardize uploaded or edited data to the dashboard's canonical schema."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", "sku", "demand", "on_hand", "unit_cost", "backorders", "lead_time"])
    df = standardize_columns(pd.DataFrame(raw))
    if "sku" not in df.columns:
        # If the file is otherwise useful, don't silently invent product IDs.
        raise ValueError("ไม่พบคอลัมน์ SKU/รหัสสินค้า กรุณาตรวจสอบหัวตาราง")
    df["sku"] = df["sku"].astype(str).str.strip()
    df.loc[df["sku"].isin(["", "nan", "None"]), "sku"] = "UNKNOWN"
    if "date" not in df.columns:
        df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for c in ["demand", "on_hand", "unit_cost", "backorders", "lead_time"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = _numeric(df[c], 0)
    # Demand cannot be negative in the forecasting model; preserve other fields.
    df["demand"] = df["demand"].clip(lower=0)
    df["on_hand"] = df["on_hand"].clip(lower=0)
    df["unit_cost"] = df["unit_cost"].clip(lower=0)
    df["backorders"] = df["backorders"].clip(lower=0)
    df["lead_time"] = df["lead_time"].clip(lower=0)
    return df[["date", "sku", "demand", "on_hand", "unit_cost", "backorders", "lead_time"]].sort_values(["sku", "date"], na_position="last").reset_index(drop=True)

def combine_uploaded_files(files):
    if not files:
        raise ValueError("กรุณาเลือกไฟล์ Excel หรือ CSV อย่างน้อย 1 ไฟล์")
    if not isinstance(files, list):
        files = [files]
    frames = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        part = standardize_columns(read_one_file(path))
        frames.append(part)
    # If a separate inventory file and sales-history file were uploaded, merge inventory
    # attributes by SKU onto the dated sales rows. If there is only one file, normalize it.
    if len(frames) > 1:
        sales_frames = [x for x in frames if "date" in x.columns and "demand" in x.columns and "sku" in x.columns]
        inventory_frames = [x for x in frames if "sku" in x.columns and any(c in x.columns for c in ["on_hand", "unit_cost", "backorders", "lead_time"])]
        if sales_frames:
            sales = pd.concat(sales_frames, ignore_index=True, sort=False)
            sales = standardize_columns(sales)
            if inventory_frames:
                inv = pd.concat(inventory_frames, ignore_index=True, sort=False)
                inv = standardize_columns(inv)
                cols = [c for c in ["on_hand", "unit_cost", "backorders", "lead_time"] if c in inv.columns]
                if cols:
                    inv["sku"] = inv["sku"].astype(str).str.strip()
                    inv[cols] = inv[cols].apply(pd.to_numeric, errors="coerce")
                    inv = inv.groupby("sku", as_index=False)[cols].last()
                    sales["sku"] = sales["sku"].astype(str).str.strip()
                    sales = sales.merge(inv, on="sku", how="left", suffixes=("", "_inventory"))
                    for c in cols:
                        alt = c + "_inventory"
                        if alt in sales.columns:
                            if c in sales.columns:
                                sales[c] = sales[c].where(sales[c].notna(), sales[alt])
                                sales = sales.drop(columns=[alt])
                            else:
                                sales = sales.rename(columns={alt: c})
            return normalize_data(sales)
        return normalize_data(pd.concat(frames, ignore_index=True, sort=False))
    return normalize_data(frames[0])

# ---------- Forecasting / KPIs ----------
def _lag_features(series):
    s = pd.Series(series, dtype=float).reset_index(drop=True)
    X, y = [], []
    for i in range(7, len(s)):
        X.append([s.iloc[i-1], s.iloc[i-2], s.iloc[i-3], s.iloc[i-7], float(s.iloc[max(0, i-7):i].mean()), float(i % 7)])
        y.append(float(s.iloc[i]))
    return np.asarray(X), np.asarray(y)

def forecast_one_sku(group, horizon):
    g = group.dropna(subset=["date"]).copy()
    if g.empty:
        return pd.DataFrame(columns=["date", "sku", "forecast_demand"]), "ค่าเฉลี่ย"
    daily = g.groupby("date", as_index=True)["demand"].sum().sort_index()
    idx = pd.date_range(daily.index.min().normalize(), daily.index.max().normalize(), freq="D")
    daily = daily.reindex(idx, fill_value=0.0).astype(float)
    sku = str(g["sku"].iloc[0])
    vals = daily.to_numpy()
    method = "ค่าเฉลี่ย 7 วัน"
    preds = []
    # Use Random Forest only with enough observations; otherwise use a transparent baseline.
    if len(vals) >= 35:
        try:
            X, y = _lag_features(vals)
            model = RandomForestRegressor(n_estimators=180, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(X, y)
            history = list(vals)
            for step in range(horizon):
                i = len(history)
                row = [[history[-1], history[-2], history[-3], history[-7], float(np.mean(history[-7:])), float(i % 7)]]
                pred = float(model.predict(row)[0])
                pred = max(0.0, pred)
                preds.append(pred)
                history.append(pred)
            method = "Random Forest (lag features)"
        except Exception:
            preds = []
    if not preds:
        base = float(np.mean(vals[-min(7, len(vals)):])) if len(vals) else 0.0
        # Slightly favor recent demand while avoiding unsupported trend extrapolation.
        recent = float(np.mean(vals[-min(3, len(vals)):])) if len(vals) else base
        base = max(0.0, 0.65 * base + 0.35 * recent)
        preds = [base] * horizon
    future = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.DataFrame({"date": future, "sku": sku, "forecast_demand": np.round(preds, 2)})
    return out, method

def backtest_metrics(data):
    errors_actual, errors_pred = [], []
    for sku, grp in data.dropna(subset=["date"]).groupby("sku"):
        daily = grp.groupby("date")["demand"].sum().sort_index()
        if len(daily) < 8:
            continue
        # Rolling one-step baseline on the final 20% (at least one observation).
        n_test = max(1, min(14, int(round(len(daily) * 0.2))))
        vals = daily.to_numpy(dtype=float)
        start = len(vals) - n_test
        for i in range(start, len(vals)):
            hist = vals[:i]
            pred = float(np.mean(hist[-min(7, len(hist)):])) if len(hist) else 0.0
            errors_actual.append(float(vals[i]))
            errors_pred.append(max(0.0, pred))
    if not errors_actual:
        return np.nan, np.nan, "ข้อมูลประวัติไม่พอ (ต้องมีอย่างน้อย 8 วันต่อ SKU)"
    actual = np.asarray(errors_actual, dtype=float)
    pred = np.asarray(errors_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    nz = actual != 0
    mape = float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100) if nz.any() else np.nan
    return mape, rmse, "ประเมินย้อนหลังด้วยค่าเฉลี่ย 7 วัน (ไม่ใช่คะแนนของ Random Forest โดยตรง)"

def make_kpi_values(df, forecast_df):
    mape, rmse, metric_note = backtest_metrics(df)
    # Latest inventory snapshot per SKU; if rows are sales history with repeated stock values,
    # take the last non-null value per SKU rather than summing repeated snapshots.
    inv = df.sort_values("date", na_position="first").groupby("sku", as_index=False).agg(
        on_hand=("on_hand", "last"), unit_cost=("unit_cost", "last"), backorders=("backorders", "last"), demand=("demand", "last")
    )
    stock_cost = float((inv["on_hand"].fillna(0) * inv["unit_cost"].fillna(0)).sum())
    total_backorders = float(inv["backorders"].fillna(0).sum())
    demand_base = float(inv["demand"].fillna(0).sum())
    stockout = (100.0 * total_backorders / (demand_base + total_backorders)) if (demand_base + total_backorders) > 0 else np.nan
    return mape, rmse, stock_cost, stockout, metric_note

def money(v):
    return "—" if pd.isna(v) else "฿{:,.2f}".format(float(v))

def number(v, digits=2, suffix=""):
    return "—" if pd.isna(v) else f"{float(v):,.{digits}f}{suffix}"

# ---------- Plotting ----------
def build_forecast_plot(df, forecast_df, sku):
    fig = go.Figure()
    history = df[(df["sku"].astype(str) == str(sku)) & df["date"].notna()].groupby("date", as_index=False)["demand"].sum().sort_values("date")
    fut = forecast_df[forecast_df["sku"].astype(str) == str(sku)].sort_values("date") if not forecast_df.empty else pd.DataFrame()
    fig.add_trace(go.Scatter(x=history["date"], y=history["demand"], mode="lines+markers", name="Demand จริง", line=dict(color="#2878E5", width=3), marker=dict(size=5)))
    if not fut.empty:
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast_demand"], mode="lines+markers", name="Forecast", line=dict(color="#F59E0B", width=3, dash="dash"), marker=dict(size=6)))
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=f"Demand Forecast — SKU {sku}", height=360, margin=dict(l=20,r=20,t=55,b=25), legend=dict(orientation="h", y=1.12), xaxis_title="วันที่", yaxis_title="จำนวน", hovermode="x unified")
    return fig

def build_stock_plot(df):
    inv = df.sort_values("date", na_position="first").groupby("sku", as_index=False).agg(on_hand=("on_hand", "last"), unit_cost=("unit_cost", "last"))
    inv["stock_value"] = inv["on_hand"].fillna(0) * inv["unit_cost"].fillna(0)
    inv = inv.sort_values("stock_value", ascending=False).head(15)
    fig = go.Figure(go.Bar(x=inv["sku"].astype(str), y=inv["stock_value"], marker_color="#2878E5", hovertemplate="SKU %{x}<br>มูลค่า ฿%{y:,.2f}<extra></extra>"))
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title="มูลค่าสต็อกตาม SKU (สูงสุด 15 รายการ)", height=330, margin=dict(l=20,r=20,t=55,b=45), xaxis_title="SKU", yaxis_title="มูลค่าสต็อก (บาท)")
    return fig

# ---------- Event callbacks ----------
def load_files(files):
    try:
        df = combine_uploaded_files(files)
        if df.empty:
            raise ValueError("ไฟล์ไม่มีข้อมูล")
        skus = sorted(df["sku"].astype(str).unique().tolist())
        status = f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว | {len(skus):,} SKU | คอลัมน์มาตรฐานพร้อมแก้ไข"
        return df, df, gr.update(choices=skus, value=skus[0] if skus else None), status
    except Exception as e:
        empty = pd.DataFrame(columns=["date", "sku", "demand", "on_hand", "unit_cost", "backorders", "lead_time"])
        return empty, empty, gr.update(choices=[], value=None), f"❌ โหลดไฟล์ไม่สำเร็จ: {e}"

def save_edits(editor_df):
    try:
        df = normalize_data(editor_df)
        skus = sorted(df["sku"].astype(str).unique().tolist())
        return df, gr.update(choices=skus, value=skus[0] if skus else None), f"✅ บันทึกการแก้ไขแล้ว: {len(df):,} แถว"
    except Exception as e:
        return None, gr.update(), f"❌ ตรวจสอบข้อมูลไม่สำเร็จ: {e}"

def run_forecast(df, sku, horizon):
    if df is None or len(df) == 0:
        raise gr.Error("กรุณาอัปโหลดไฟล์ก่อน")
    df = normalize_data(df)
    if sku is None or str(sku) not in set(df["sku"].astype(str)):
        sku = str(df["sku"].iloc[0])
    horizon = int(horizon)
    all_forecasts, methods = [], []
    selected = df[df["sku"].astype(str) == str(sku)]
    forecast, method = forecast_one_sku(selected, horizon)
    all_forecasts.append(forecast)
    methods.append(method)
    # KPI metrics cover the complete uploaded dataset, while forecast chart/table show selected SKU.
    mape, rmse, stock_cost, stockout, metric_note = make_kpi_values(df, forecast)
    kpi_html = f"""<div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">MAPE</div><div class="kpi-value">{number(mape, 2, "%")}</div><div class="kpi-note">ความคลาดเคลื่อนเฉลี่ยแบบเปอร์เซ็นต์</div></div>
      <div class="kpi-card"><div class="kpi-label">RMSE</div><div class="kpi-value">{number(rmse, 2)}</div><div class="kpi-note">หน่วยเดียวกับ Demand</div></div>
      <div class="kpi-card"><div class="kpi-label">ต้นทุนสินค้าคงคลัง</div><div class="kpi-value">{money(stock_cost)}</div><div class="kpi-note">สต็อกล่าสุด × ต้นทุนต่อหน่วย</div></div>
      <div class="kpi-card"><div class="kpi-label">อัตราสินค้าขาด (ประมาณ)</div><div class="kpi-value">{number(stockout, 2, "%")}</div><div class="kpi-note">คำนวณจาก Backorders และ Demand ล่าสุด</div></div>
    </div><div class="metric-note">{html.escape(metric_note)}<br>วิธีพยากรณ์ SKU ที่เลือก: {html.escape(method)}</div>"""
    forecast_df = forecast.copy()
    fig_forecast = build_forecast_plot(df, forecast_df, sku)
    fig_stock = build_stock_plot(df)
    summary = f"สร้าง Forecast สำเร็จสำหรับ SKU {sku} จำนวน {horizon} วัน | วิธี: {method}"
    return kpi_html, forecast_df, fig_forecast, fig_stock, summary, forecast_df

def export_excel(df, forecast_df):
    if df is None or len(df) == 0:
        raise gr.Error("ไม่มีข้อมูลสำหรับส่งออก")
    df = normalize_data(df)
    if forecast_df is None:
        forecast_df = pd.DataFrame(columns=["date", "sku", "forecast_demand"])
    mape, rmse, stock_cost, stockout, metric_note = make_kpi_values(df, forecast_df)
    kpis = pd.DataFrame([
        {"Metric": "MAPE (%)", "Value": mape},
        {"Metric": "RMSE", "Value": rmse},
        {"Metric": "Inventory Cost (THB)", "Value": stock_cost},
        {"Metric": "Estimated Stockout Rate (%)", "Value": stockout},
        {"Metric": "Metric note", "Value": metric_note},
    ])
    fd, path = tempfile.mkstemp(prefix="inventory_dashboard_", suffix=".xlsx")
    os.close(fd)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
        forecast_df.to_excel(writer, index=False, sheet_name="Forecast")
        kpis.to_excel(writer, index=False, sheet_name="KPIs")
    return path

# ---------- Styling: eight theme buttons + light/dark toggle ----------
CSS = r"""
:root { --accent:#2878E5; --accent-soft:#EAF2FF; --page:#F3F7FC; --panel:#FFFFFF; --ink:#17365C; --muted:#718198; --line:#DDE6F2; --grid:#E6ECF4; }
.gradio-container { background:var(--page) !important; color:var(--ink) !important; min-height:100vh; }
.gradio-container main, .gradio-container .main, .gradio-container .contain, .gradio-container .wrap, .gradio-container .block, .gradio-container .form { background-color:transparent; color:var(--ink) !important; }
.gradio-container .block:not(.gradio-accordion), .gradio-container .accordion, .gradio-container [data-testid="block-info"] { border-color:var(--line) !important; }
#app-header { border:1px solid var(--line); border-radius:18px; padding:16px 18px; background:var(--panel); margin-bottom:10px; }
#header-controls { gap:6px; align-items:center; justify-content:flex-end; }
#theme-button button { min-width:42px !important; width:42px !important; height:38px !important; padding:0 !important; border-radius:12px !important; font-size:17px !important; }
#theme-buttons-top, #theme-buttons-bottom { gap:8px; margin:8px 0 12px; }
#theme-buttons button { min-height:38px !important; border-radius:11px !important; font-size:12px !important; font-weight:650 !important; border:1px solid var(--line) !important; transition:transform .12s ease, filter .12s ease; }
#theme-buttons button:hover { filter:brightness(1.06); transform:translateY(-1px); }
#theme-modern-light button { background:#2563EB !important; color:#fff !important; }
#theme-midnight-dark button { background:#111827 !important; color:#F8FAFC !important; border-color:#475569 !important; }
#theme-orange-executive button { background:#F97316 !important; color:#171717 !important; }
#theme-emerald-business button { background:#047857 !important; color:#fff !important; }
#theme-purple-ai button { background:#7C3AED !important; color:#fff !important; }
#theme-navy-corporate button { background:#0F172A !important; color:#DBEAFE !important; }
#theme-soft-minimal button { background:#A3B18A !important; color:#24352B !important; }
#theme-cyber-ai button { background:#0F766E !important; color:#E0FFFF !important; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:4px 0 12px; }
.kpi-card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:17px; box-shadow:0 3px 12px rgba(25,55,95,.04); min-width:0; }
.kpi-label { color:var(--muted); font-size:13px; margin-bottom:8px; }
.kpi-value { color:var(--accent); font-size:clamp(20px,2vw,28px); font-weight:750; line-height:1.2; overflow-wrap:anywhere; }
.kpi-note,.metric-note { color:var(--muted); font-size:12px; margin-top:8px; line-height:1.5; }
.panel-card { border:1px solid var(--line); border-radius:16px; padding:12px; background:var(--panel); }
button.primary { background:var(--accent) !important; border-color:var(--accent) !important; }
/* Theme variables style Gradio blocks, editable tables and form controls consistently. */
.gradio-container .block, .gradio-container .form, .gradio-container .panel, .gradio-container .wrap { color:var(--ink); }
.gradio-container input, .gradio-container textarea, .gradio-container select { background:var(--panel) !important; color:var(--ink) !important; border-color:var(--line) !important; }
.gradio-container .table-wrap, .gradio-container .table-container, .gradio-container table,
.gradio-container thead, .gradio-container tbody, .gradio-container tr,
.gradio-container th, .gradio-container td,
.gradio-container [role="grid"], .gradio-container [role="gridcell"],
.gradio-container [role="columnheader"] { background-color:var(--panel) !important; color:var(--ink) !important; border-color:var(--line) !important; }
html.dark-mode { color-scheme:dark; }
html.dark-mode .gradio-container { color-scheme:dark; }
html.dark-mode .gradio-container .prose, html.dark-mode .gradio-container label, html.dark-mode .gradio-container h1, html.dark-mode .gradio-container h2, html.dark-mode .gradio-container h3 { color:var(--ink) !important; }
html.dark-mode .prose, html.dark-mode label, html.dark-mode p, html.dark-mode h1, html.dark-mode h2, html.dark-mode h3 { color:var(--ink); }
html.dark-mode button:not(.primary) { background:var(--panel) !important; color:var(--ink) !important; border-color:var(--line) !important; }
html.dark-mode #theme-buttons button { filter:saturate(.9); }
html.dark-mode .plot-container, html.dark-mode .js-plotly-plot, html.dark-mode .svg-container { background:transparent !important; }
@media (max-width:850px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } #app-header { padding:12px; } }
@media (max-width:500px) { .kpi-grid { grid-template-columns:1fr 1fr; gap:8px; } .kpi-card { padding:12px; } #theme-buttons-top button, #theme-buttons-bottom button { font-size:10px !important; padding:5px !important; } }
"""

HEAD = r"""
<script>
(function () {
  const palettes = {
    'Modern Light': {accent:'#2563EB',soft:'#EAF2FF',light:['#F8FAFC','#FFFFFF','#0F172A','#64748B','#DCE5F0','#E8EEF6'],dark:['#111827','#1F2937','#F8FAFC','#A8B6C9','#374151','#374151']},
    'Midnight Dark': {accent:'#38BDF8',soft:'#1E293B',light:['#F8FAFC','#FFFFFF','#0F172A','#64748B','#DCE5F0','#E8EEF6'],dark:['#0B1120','#172033','#F1F5F9','#A8B6C9','#334155','#263449']},
    'Orange Executive': {accent:'#F97316',soft:'#FFF0E0',light:['#FFF7ED','#FFFFFF','#292524','#78716C','#E7D5C6','#F3E8DD'],dark:['#171717','#262626','#FAFAF9','#A8A29E','#44403C','#33302E']},
    'Emerald Business': {accent:'#047857',soft:'#D1FAE5',light:['#F0FDF4','#FFFFFF','#14532D','#64748B','#D1E7D7','#E2F0E6'],dark:['#0B1F1A','#15352B','#ECFDF5','#A7C9BA','#2D5145','#234337']},
    'Purple AI': {accent:'#7C3AED',soft:'#EDE9FE',light:['#FAF5FF','#FFFFFF','#2E1065','#6B7280','#E5D9F7','#EEE7F8'],dark:['#171126','#251B3A','#F5F3FF','#C4B5FD','#493767','#38294F']},
    'Navy Corporate': {accent:'#1D4ED8',soft:'#DBEAFE',light:['#EFF6FF','#FFFFFF','#0F172A','#64748B','#D7E3F4','#E2EAF5'],dark:['#0B1220','#172338','#EFF6FF','#A9BCD8','#35445D','#293A54']},
    'Soft Minimal': {accent:'#58745B',soft:'#E9E5D9',light:['#FAF7F0','#FFFFFF','#344E41','#77796F','#E4DFD2','#EAE6DC'],dark:['#20251F','#2D342B','#F4F1E8','#B8C0B2','#4A5447','#3C4639']},
    'Cyber AI': {accent:'#2DD4BF',soft:'#123B3A',light:['#ECFEFF','#FFFFFF','#164E63','#64748B','#CBE7EB','#D9EEF0'],dark:['#071A22','#0D2A33','#E0FFFF','#9CCED0','#28505A','#1B3D47']}
  };
  const ids = {
    'theme-modern-light':'Modern Light','theme-midnight-dark':'Midnight Dark',
    'theme-orange-executive':'Orange Executive','theme-emerald-business':'Emerald Business',
    'theme-purple-ai':'Purple AI','theme-navy-corporate':'Navy Corporate',
    'theme-soft-minimal':'Soft Minimal','theme-cyber-ai':'Cyber AI'
  };
  let selected = 'Modern Light';
  let dark = false;
  const root = () => document.documentElement;
  function restylePlots() {
    const r=root(), cs=getComputedStyle(r);
    const c={accent:cs.getPropertyValue('--accent').trim(),panel:cs.getPropertyValue('--panel').trim(),ink:cs.getPropertyValue('--ink').trim(),muted:cs.getPropertyValue('--muted').trim(),line:cs.getPropertyValue('--line').trim(),grid:cs.getPropertyValue('--grid').trim()};
    document.querySelectorAll('.js-plotly-plot').forEach(plot=>{
      if(!window.Plotly || !plot.data || !plot.layout) return;
      try {
        window.Plotly.relayout(plot,{'paper_bgcolor':c.panel,'plot_bgcolor':c.panel,'font.color':c.ink,'title.font.color':c.ink,'xaxis.color':c.muted,'yaxis.color':c.muted,'xaxis.gridcolor':c.grid,'yaxis.gridcolor':c.grid,'xaxis.linecolor':c.line,'yaxis.linecolor':c.line,'legend.font.color':c.ink});
        plot.data.forEach((trace,i)=>{
          if(trace.type==='bar') window.Plotly.restyle(plot,{'marker.color':c.accent},[i]);
          else if(trace.type==='scatter' && i===0) window.Plotly.restyle(plot,{'line.color':c.accent},[i]);
          else if(trace.type==='scatter' && i===1) window.Plotly.restyle(plot,{'line.color':'#F59E0B'},[i]);
        });
      } catch(e) {}
    });
  }
  function applyColors() {
    const p=palettes[selected] || palettes['Modern Light'];
    const v=dark?p.dark:p.light, r=root();
    const vars={'--accent':p.accent,'--accent-soft':p.soft,'--page':v[0],'--panel':v[1],'--ink':v[2],'--muted':v[3],'--line':v[4],'--grid':v[5]};
    Object.keys(vars).forEach(k=>r.style.setProperty(k,vars[k]));
    r.classList.toggle('dark-mode',dark);
    // Also set the container directly so Gradio's own CSS cannot mask inherited colors.
    document.querySelectorAll('.gradio-container').forEach(el=>{
      Object.keys(vars).forEach(k=>el.style.setProperty(k,vars[k]));
      el.style.setProperty('background-color',v[0],'important');
      el.style.setProperty('color',v[2],'important');
    });
    document.querySelectorAll('.kpi-card,.panel-card,#app-header').forEach(el=>{
      el.style.setProperty('background-color',v[1],'important');
      el.style.setProperty('color',v[2],'important');
      el.style.setProperty('border-color',v[4],'important');
    });
    restylePlots();
  }
  function selectPalette(name) { if(!palettes[name]) return; selected=name; dark=(name==='Midnight Dark'||name==='Cyber AI'); applyColors(); }
  function handleClick(event) {
    const target=event.target.closest('[id]');
    if(!target) return;
    if(ids[target.id]) { selectPalette(ids[target.id]); return; }
    if(target.id==='theme-button' || target.closest('#theme-button')) { dark=!dark; applyColors(); }
  }
  // Capture clicks directly. This works even if Gradio's optional JS callback API differs by version.
  document.addEventListener('click',handleClick,true);
  window.setInventoryPalette=selectPalette;
  window.toggleInventoryMode=function(){dark=!dark;applyColors();return dark;};
  function start(){applyColors(); let queued=false; const observer=new MutationObserver(()=>{if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;restylePlots();});}); observer.observe(document.body,{childList:true,subtree:true});}
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start,{once:true}); else start();
})();
</script>
"""

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=CSS, head=HEAD) as demo:
    data_state = gr.State(pd.DataFrame(columns=["date", "sku", "demand", "on_hand", "unit_cost", "backorders", "lead_time"]))
    forecast_state = gr.State(pd.DataFrame(columns=["date", "sku", "forecast_demand"]))
    dark_state = gr.State(False)

    with gr.Row(elem_id="app-header"):
        with gr.Column(scale=8, min_width=220):
            gr.Markdown("# 📦 AI Inventory Dashboard\nForecasting · Inventory Cost · Stockout Analysis")
        with gr.Column(scale=2, min_width=110, elem_id="header-controls"):
            theme_btn = gr.Button("🌙", size="sm", elem_id="theme-button", scale=0, min_width=42)

    palette_state = gr.State("Modern Light")
    gr.Markdown("**เลือกธีมสี Dashboard** — เปลี่ยนสีพื้นหลัง การ์ด ตาราง และกราฟ โดยไม่กระทบข้อมูลหรือผลพยากรณ์")
    with gr.Row(elem_id="theme-buttons-top"):
        theme_modern = gr.Button("🔵 Modern Light", elem_id="theme-modern-light", size="sm")
        theme_midnight = gr.Button("🌑 Midnight Dark", elem_id="theme-midnight-dark", size="sm")
        theme_orange = gr.Button("🟠 Orange Executive", elem_id="theme-orange-executive", size="sm")
        theme_emerald = gr.Button("🟢 Emerald Business", elem_id="theme-emerald-business", size="sm")
    with gr.Row(elem_id="theme-buttons-bottom"):
        theme_purple = gr.Button("🟣 Purple AI", elem_id="theme-purple-ai", size="sm")
        theme_navy = gr.Button("🔷 Navy Corporate", elem_id="theme-navy-corporate", size="sm")
        theme_soft = gr.Button("🌿 Soft Minimal", elem_id="theme-soft-minimal", size="sm")
        theme_cyber = gr.Button("🧪 Cyber AI", elem_id="theme-cyber-ai", size="sm")

    gr.Markdown("อัปโหลดไฟล์ เปลี่ยนธีมได้ทันที หรือใช้ปุ่ม 🌙/☀️ เพื่อสลับสว่าง–มืด โดยไม่เปิดหน้าต่างใหม่")

    with gr.Accordion("① อัปโหลดไฟล์ Excel / CSV", open=True):
        file_input = gr.File(label="เลือก inventory.csv, sales_history.csv หรือไฟล์ Excel", file_count="multiple", file_types=[".csv", ".xlsx", ".xlsm"], type="filepath")
        load_btn = gr.Button("📥 โหลดไฟล์และรวมข้อมูล", variant="primary")
        load_status = gr.Markdown("ยังไม่ได้โหลดข้อมูล")
        gr.Markdown("**แนะนำ:** ประวัติขายใช้ `date, sku, demand` และสต็อกใช้ `sku, on_hand, unit_cost, backorders` หากอัปโหลดสองไฟล์ ระบบจะพยายามรวมข้อมูลด้วย SKU")

    with gr.Accordion("② ตารางข้อมูลสินค้า — แก้ไขได้", open=True):
        editor = gr.Dataframe(
            headers=["date", "sku", "demand", "on_hand", "unit_cost", "backorders", "lead_time"],
            datatype=["date", "str", "number", "number", "number", "number", "number"],
            value=pd.DataFrame(columns=["date", "sku", "demand", "on_hand", "unit_cost", "backorders", "lead_time"]),
            type="pandas", interactive=True, row_count=(5, "dynamic"), col_count=(7, "fixed"),
            label="แก้ไขค่าตามต้องการ: date, sku, demand, on_hand, unit_cost, backorders, lead_time"
        )
        save_btn = gr.Button("💾 บันทึกการแก้ไข", variant="secondary")
        edit_status = gr.Markdown("หลังแก้ตาราง กดบันทึกการแก้ไขก่อนพยากรณ์")

    with gr.Accordion("③ ตั้งค่าการพยากรณ์", open=True):
        with gr.Row():
            sku_select = gr.Dropdown(choices=[], label="เลือก SKU", interactive=True, scale=3, allow_custom_value=False)
            horizon = gr.Slider(minimum=1, maximum=90, value=14, step=1, label="ระยะเวลาพยากรณ์ (วัน)", scale=4)
        forecast_btn = gr.Button("🤖 วิเคราะห์และพยากรณ์ Demand", variant="primary")
        forecast_status = gr.Markdown("รอข้อมูลสำหรับพยากรณ์")

    gr.Markdown("## KPI Overview")
    kpi_html = gr.HTML("<div class='metric-note'>อัปโหลดข้อมูลและกดพยากรณ์เพื่อแสดง KPI</div>")

    with gr.Row():
        with gr.Column(scale=3):
            forecast_plot = gr.Plot(label="Demand Forecast")
        with gr.Column(scale=2):
            stock_plot = gr.Plot(label="Inventory Value")

    with gr.Accordion("④ ผลพยากรณ์", open=True):
        forecast_table = gr.Dataframe(
            headers=["date", "sku", "forecast_demand"],
            datatype=["date", "str", "number"],
            value=pd.DataFrame(columns=["date", "sku", "forecast_demand"]),
            type="pandas", interactive=False, label="Forecast Demand"
        )

    with gr.Row():
        export_btn = gr.Button("📤 ส่งออก Excel (.xlsx)", variant="primary")
        export_file = gr.File(label="ดาวน์โหลดรายงาน Excel", interactive=False)
    gr.Markdown("**หมายเหตุการคำนวณ:** MAPE/RMSE ประเมินย้อนหลังด้วยค่าเฉลี่ย 7 วัน เพื่อเป็น baseline เปรียบเทียบ; MAPE ไม่นับวันที่ Demand จริงเป็นศูนย์. อัตราสินค้าขาดเป็นค่าประมาณจาก Backorders และ Demand ล่าสุด ไม่ใช่อัตราขาดย้อนหลังจริง. เติม `unit_cost` และ `backorders` เพื่อให้ KPI มีความหมาย")

    load_btn.click(load_files, inputs=file_input, outputs=[data_state, editor, sku_select, load_status])
    file_input.upload(load_files, inputs=file_input, outputs=[data_state, editor, sku_select, load_status])
    save_btn.click(save_edits, inputs=editor, outputs=[data_state, sku_select, edit_status])
    forecast_btn.click(run_forecast, inputs=[data_state, sku_select, horizon], outputs=[kpi_html, forecast_table, forecast_plot, stock_plot, forecast_status, forecast_state])
    export_btn.click(export_excel, inputs=[data_state, forecast_state], outputs=export_file)

    # Theme buttons only change client-side styles; data_state and forecast_state remain untouched.
    _theme_buttons = [
        (theme_modern, "Modern Light"), (theme_midnight, "Midnight Dark"),
        (theme_orange, "Orange Executive"), (theme_emerald, "Emerald Business"),
        (theme_purple, "Purple AI"), (theme_navy, "Navy Corporate"),
        (theme_soft, "Soft Minimal"), (theme_cyber, "Cyber AI"),
    ]
    for _button, _name in _theme_buttons:
        _button.click(fn=lambda name=_name: name, inputs=None, outputs=palette_state)
    theme_btn.click(
        fn=lambda state: (not state, "☀️" if not state else "🌙"),
        inputs=dark_state, outputs=[dark_state, theme_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "10000")), share=False, debug=False)
