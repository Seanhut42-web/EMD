# app.py
# v0.2 — EMD FoF Performance (New Dashboard mapping built-in)
# NOTE: Delivered as app.py.txt — rename to app.py after download.

import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from calc import (
    load_from_new_dashboard,
    compute_weights,
    cumulative,
    compute_alpha_decomp,
    monthly_returns_table,
    mask_terminated_managers,
)

st.set_page_config(page_title="EMD FoF — Performance Explorer", layout="wide")
st.title("EMD Fund-of-Funds — Performance Explorer (v0.2.1)")

with st.sidebar:
    st.header("Data source: New Dashboard workbook")
    xlsx_path = st.text_input("Path to New Dashboard EMD.xlsx", value=str(Path('./data/New Dashboard EMD.xlsx').resolve()))
    returns_sheet = st.text_input("Returns sheet name", value="EMD_Returns")
    navs_sheet = st.text_input("NAVs sheet name", value="Market_Values")

    st.caption("The app parses Tab 1 (returns) & Tab 2 (NAVs) from your New Dashboard workbook.")

    st.header("Labels & units")
    returns_in_percent = st.checkbox("Returns are in % (e.g., 0.25 = 0.25%)", value=True)

    # Canonical labels per Sean
    bm_name_default = "EMD Benchmark"
    portfolio_candidates = ["Portfolio", "Fixed Income"]
    # You can override portfolio series if auto-detect picks the wrong one
    portfolio_override = st.text_input("Portfolio series name (optional)", value="")

    st.caption("Managers (as canonical labels): Overlay, EMSO, Global Evolution, Neuberger Berman, Schroders")

    st.divider()
    st.header("Schroders termination")
    schroders_name = "Schroders"
    schroders_termination = st.text_input("Termination date (YYYY-MM-DD) — leave blank to infer from NAVs", value="")
    exclude_post_term = st.checkbox("Exclude Schroders post-termination (set NAV=0, ignore returns)", value=True)

# Load data directly from the New Dashboard workbook
try:
    returns, navs, detected_labels = load_from_new_dashboard(
        xlsx_path, returns_sheet, navs_sheet, returns_in_percent=returns_in_percent
    )
except Exception as e:
    st.error(f"Failed to parse workbook: {e}")
    st.stop()

# Determine portfolio & benchmark
bm_name = "EMD Benchmark"
# Auto-detect portfolio from common candidates, allow override
series_all = sorted(returns['Series'].unique())
portfolio_name = portfolio_override.strip() or next((s for s in ["Portfolio","Fixed Income"] if s in series_all), None)
if portfolio_name is None:
    # fallback: if there's exactly one non-benchmark non-managers total-like label
    portfolio_name = detected_labels.get('portfolio', None) or 'Portfolio'

# Apply Schroders termination logic from NAV inference or explicit input
if exclude_post_term:
    returns, navs, inferred_date = mask_terminated_managers(returns, navs, schroders_name, schroders_termination or None)
    if inferred_date and not schroders_termination:
        st.info(f"Inferred Schroders termination date from NAVs: {inferred_date}")

# Compute weights from NAVs (Overlay is treated as a manager by design)
weights = compute_weights(navs)

# Identify manager list (includes Overlay; excludes portfolio & benchmark)
manager_series = [s for s in series_all if s not in {bm_name, portfolio_name}]

# Date range filter
min_date, max_date = returns['Date'].min(), returns['Date'].max()
dstart, dend = st.date_input("Date range", value=(min_date, max_date))
returns = returns[(returns['Date'] >= pd.to_datetime(dstart)) & (returns['Date'] <= pd.to_datetime(dend))].copy()
navs = navs[(navs['Date'] >= pd.to_datetime(dstart)) & (navs['Date'] <= pd.to_datetime(dend))].copy()
weights = weights[(weights['Date'] >= pd.to_datetime(dstart)) & (weights['Date'] <= pd.to_datetime(dend))].copy()

# Cumulative performance — Portfolio vs Benchmark
df_pb = returns[returns['Series'].isin([portfolio_name, bm_name])].pivot(index='Date', columns='Series', values='Return').sort_index()
for c in df_pb.columns:
    df_pb[c] = df_pb[c].astype(float)

cum_pb = cumulative(df_pb)
fig_pb = px.line(cum_pb * 100, labels={'value': 'Cumulative Return (%)', 'Date': ''})
fig_pb.update_layout(legend_title_text='')

# Cumulative alpha (Portfolio vs Benchmark)
if {portfolio_name, bm_name}.issubset(df_pb.columns):
    alpha_curve = (1 + df_pb[portfolio_name]).cumprod() / (1 + df_pb[bm_name]).cumprod() - 1
    fig_alpha = px.line((alpha_curve * 100).rename('Cumulative Alpha (%)'))
else:
    fig_alpha = go.Figure()

# Per-manager cumulative vs benchmark
manager_pick = st.selectbox("Manager for detailed view", ["(All managers)"] + manager_series, index=0)
if manager_pick and manager_pick != "(All managers)":
    df_m = returns[returns['Series'].isin([manager_pick, bm_name])].pivot(index='Date', columns='Series', values='Return').sort_index()
    cum_m = cumulative(df_m)
    fig_mgr = px.line(cum_m * 100, labels={'value': 'Cumulative Return (%)', 'Date': ''})
    fig_mgr.update_layout(legend_title_text='')
else:
    # Show a small multiples style (optional future), for now just blank
    fig_mgr = go.Figure()

# Alpha decomposition — Selection vs Construction (Overlay included as a manager)
alpha_daily, mtd_bars = compute_alpha_decomp(returns, weights, portfolio_name, bm_name, manager_series)

# Monthly table (compounded)
monthlies = monthly_returns_table(returns, series=[portfolio_name, bm_name] + manager_series)

# Layout
a, b = st.columns(2)
with a:
    st.subheader("Cumulative: Portfolio vs EMD Benchmark")
    st.plotly_chart(fig_pb, use_container_width=True)
with b:
    st.subheader("Cumulative Alpha (Portfolio vs Benchmark)")
    st.plotly_chart(fig_alpha, use_container_width=True)

st.subheader("Manager vs Benchmark")
st.plotly_chart(fig_mgr, use_container_width=True)

st.subheader("Alpha Decomposition (MTD): Selection vs Construction")
st.plotly_chart(mtd_bars, use_container_width=True)

st.subheader("Monthly Returns (%) — compounded")
st.dataframe((monthlies*100).round(2))

st.divider()
st.subheader("Audit this number")
with st.expander("Formulas"):
    st.markdown("""
**Daily definitions**  
- Weighted manager return:  \( r^{\text{Mgr}}_t = \sum_i w_{i,t} r^i_t \)  
- Selection:  \( S_t = \sum_i w_{i,t}(r^i_t - r^B_t) \)  
- Construction:  \( C_t = r^P_t - r^{\text{Mgr}}_t \)  
- Alpha check:  \( r^P_t - r^B_t = S_t + C_t \)

**MTD**: arithmetic sum of daily values. **Monthly table** is compounded from daily (\(\prod (1+r) - 1\)).
""")

with st.expander("Audit a specific day"):
    ad = st.date_input("Pick a date", value=returns['Date'].max())
    ad = pd.to_datetime(ad)
    df_day = returns[returns['Date']==ad].set_index('Series')['Return']
    bm = df_day.get(bm_name, np.nan)
    port = df_day.get(portfolio_name, np.nan)

    wt_day = compute_weights(navs[navs['Date']==ad])
    wt_ser = wt_day.set_index('Manager')['Weight'] if not wt_day.empty else pd.Series(dtype=float)

    rows = []
    for m in manager_series:
        w = float(wt_ser.get(m, 0.0)) if not wt_ser.empty else np.nan
        r = float(df_day.get(m, np.nan))
        sel = w * (r - bm) if pd.notna(w) and pd.notna(r) and pd.notna(bm) else np.nan
        rows.append((m, w, r, sel))
    mgr_df = pd.DataFrame(rows, columns=["Manager","Weight","Return","Selection_Contribution"]) 

    r_mgr = np.nansum(mgr_df['Weight'] * mgr_df['Return']) if not mgr_df.empty else np.nan
    construction = port - r_mgr if pd.notna(port) and pd.notna(r_mgr) else np.nan

    st.write("**Inputs**")
    st.dataframe(mgr_df)
    st.write("**Computed**", {
        'Benchmark (rB)': bm,
        'Portfolio (rP)': port,
        'Weighted Manager Return (rMgr)': r_mgr,
        'Sum Selection (∑ w*(ri-rB))': float(np.nansum(mgr_df['Selection_Contribution'])) if not mgr_df.empty else np.nan,
        'Construction (rP - rMgr)': construction,
        'Alpha check (rP - rB)': (port - bm) if pd.notna(port) and pd.notna(bm) else np.nan,
    })

st.caption("v0.2 — reads New Dashboard EMD.xlsx (Tabs 1 & 2), units in percent, Schroders termination inferred from NAVs, Overlay treated as a manager.")
