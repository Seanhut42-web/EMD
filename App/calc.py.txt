# calc.py
# v0.2 parsing for "New Dashboard EMD.xlsx"

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

DATE_RANGE_RE = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})\s*to\s*(\d{1,2}-[A-Za-z]{3}-\d{4})")
DATE_SINGLE_RE = re.compile(r"\d{1,2}-[A-Za-z]{3}-\d{4}")

# Canonical names requested by Sean
CANONICAL = {
    'EMD': 'EMD Benchmark',
    'EMD BENCH': 'EMD Benchmark',
    'EMD BENCHMARK': 'EMD Benchmark',
    'EMSO MASTER FUND': 'EMSO',
    'GLOBAL EVOLUTION': 'Global Evolution',
    'NEUBERGER BERMAN SHORT DURATION': 'Neuberger Berman',
    'SCHRODERS EMD': 'Schroders',
    'EMD OVERLAY': 'Overlay',
    'OVERLAY': 'Overlay',
    'FIXED INCOME': 'Portfolio',
    'PORTFOLIO': 'Portfolio',
}

MANAGER_SET = {
    'Overlay', 'EMSO', 'Global Evolution', 'Neuberger Berman', 'Schroders'
}


def _parse_header_to_date(col) -> pd.Timestamp | None:
    s = str(col).strip()
    m = DATE_RANGE_RE.search(s)
    if m:
        # use end date of the range
        end = pd.to_datetime(m.group(2), format='%d-%b-%Y', errors='coerce')
        return end
    # try single date
    if DATE_SINGLE_RE.fullmatch(s):
        return pd.to_datetime(s, format='%d-%b-%Y', errors='coerce')
    return None


def _canon(series_name: str) -> str:
    if series_name is None:
        return ''
    s = str(series_name).strip()
    s_up = s.upper()
    # exact map
    if s_up in CANONICAL:
        return CANONICAL[s_up]
    # fuzzy contains for known labels
    for k, v in CANONICAL.items():
        if k in s_up:
            return v
    return s


def load_from_new_dashboard(xlsx_path: str | Path, returns_sheet: str = 'EMD_Returns', navs_sheet: str = 'Market_Values', returns_in_percent: bool = True):
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {xlsx_path}")

    # --- RETURNS (Tab 1) ---
    try:
        r_raw = pd.read_excel(xlsx_path, sheet_name=returns_sheet, header=0)
    except Exception:
        # fallback to first sheet
        r_raw = pd.read_excel(xlsx_path, sheet_name=0, header=0)

    # Identify the series column (Ticker/Series). Fallback to first col.
    first_col = r_raw.columns[0]
    series_col = None
    for c in r_raw.columns[:3]:
        if str(c).strip().lower() in {'ticker','series','name'}:
            series_col = c
            break
    if series_col is None:
        series_col = first_col

    # Collect date columns by parsing headers
    date_cols = []
    dates = []
    for c in r_raw.columns:
        dt = _parse_header_to_date(c)
        if dt is not None:
            date_cols.append(c)
            dates.append(dt)

    # Filter out footer rows like 'Total' and blank series
    r_use = r_raw[[series_col] + date_cols].copy()
    r_use = r_use.rename(columns={series_col: 'Series'})
    r_use['Series'] = r_use['Series'].astype(str)
    r_use = r_use[~r_use['Series'].str.strip().str.upper().isin({'', 'TOTAL'})]

    # Map to canonical names and keep only series of interest (benchmark, portfolio, managers)
    r_use['Series'] = r_use['Series'].apply(_canon)

    # Drop duplicate series rows (some sheets have multiple sections). Keep first occurrence.
    r_use = r_use[~r_use['Series'].duplicated(keep='first')]

    # Melt to tidy
    r_tidy = r_use.melt(id_vars=['Series'], value_vars=date_cols, var_name='Col', value_name='Return')
    # attach parsed dates
    col_to_date = {c: _parse_header_to_date(c) for c in date_cols}
    r_tidy['Date'] = r_tidy['Col'].map(col_to_date)
    r_tidy = r_tidy.drop(columns=['Col'])

    # Clean numeric and units
    r_tidy['Return'] = pd.to_numeric(r_tidy['Return'], errors='coerce')
    if returns_in_percent:
        r_tidy['Return'] = r_tidy['Return'] / 100.0

    r_tidy = r_tidy.dropna(subset=['Date']).sort_values('Date')

    # Keep only the canonical benchmark + managers + portfolio (if present)
    keep_labels = {'EMD Benchmark', 'Portfolio'} | MANAGER_SET
    r_tidy = r_tidy[r_tidy['Series'].isin(keep_labels)].copy()

    # --- NAVs (Tab 2: Market_Values) ---
    try:
        n_raw = pd.read_excel(xlsx_path, sheet_name=navs_sheet, header=0)
    except Exception:
        # fallback to second sheet (index=1)
        n_raw = pd.read_excel(xlsx_path, sheet_name=1, header=0)

    # Heuristic: first column is series/manager label; date-like headers become dates
    n_first_col = n_raw.columns[0]
    n_raw = n_raw.rename(columns={n_first_col: 'Manager'})

    nav_date_cols = []
    for c in n_raw.columns:
        dt = _parse_header_to_date(c)
        if dt is not None:
            nav_date_cols.append(c)

    if not nav_date_cols:
        # also accept pure dates in header
        for c in n_raw.columns[1:]:
            try:
                dt = pd.to_datetime(str(c), errors='coerce')
            except Exception:
                dt = None
            if pd.notna(dt):
                nav_date_cols.append(c)

    n_use = n_raw[['Manager'] + nav_date_cols].copy()
    n_use['Manager'] = n_use['Manager'].astype(str).apply(_canon)
    # Keep only managers
    n_use = n_use[n_use['Manager'].isin(MANAGER_SET)].copy()

    n_tidy = n_use.melt(id_vars=['Manager'], value_vars=nav_date_cols, var_name='Col', value_name='NAV')
    # Map header to dates
    col_to_date_n = {}
    for c in nav_date_cols:
        dt = _parse_header_to_date(c)
        if dt is None:
            dt = pd.to_datetime(str(c), errors='coerce')
        col_to_date_n[c] = dt
    n_tidy['Date'] = n_tidy['Col'].map(col_to_date_n)
    n_tidy = n_tidy.drop(columns=['Col'])

    n_tidy['NAV'] = pd.to_numeric(n_tidy['NAV'], errors='coerce')
    n_tidy = n_tidy.dropna(subset=['Date']).sort_values('Date')

    detected = {'portfolio': 'Portfolio' if 'Portfolio' in r_tidy['Series'].unique() else 'Fixed Income' if 'Fixed Income' in r_raw.iloc[:,0].astype(str).str.upper().tolist() else None}

    return r_tidy, n_tidy, detected


def compute_weights(navs: pd.DataFrame) -> pd.DataFrame:
    if navs.empty:
        return pd.DataFrame(columns=['Date','Manager','Weight'])
    w = navs.copy()
    w['NAV'] = pd.to_numeric(w['NAV'], errors='coerce')
    w = w.dropna(subset=['NAV'])
    w['TotalNAV'] = w.groupby('Date')['NAV'].transform('sum')
    w['Weight'] = np.where(w['TotalNAV']!=0, w['NAV']/w['TotalNAV'], np.nan)
    return w[['Date','Manager','Weight']]


def cumulative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    return (1 + df.fillna(0)).cumprod() - 1


def compute_alpha_decomp(returns: pd.DataFrame, weights: pd.DataFrame, portfolio_name: str, bm_name: str, manager_series: list[str]):
    import plotly.graph_objects as go
    r_wide = returns.pivot(index='Date', columns='Series', values='Return').sort_index()
    bm = r_wide.get(bm_name)
    port = r_wide.get(portfolio_name)
    mgr = r_wide.reindex(columns=manager_series)

    if not weights.empty:
        w_wide = weights.pivot(index='Date', columns='Manager', values='Weight').reindex_like(mgr)
    else:
        w_wide = pd.DataFrame(index=mgr.index, columns=mgr.columns, data=1.0/len(manager_series) if manager_series else np.nan)

    sel_daily = (w_wide * (mgr.sub(bm, axis=0))).sum(axis=1)
    r_mgr_daily = (w_wide * mgr).sum(axis=1)
    cons_daily = port.sub(r_mgr_daily, fill_value=np.nan)

    alpha_daily = pd.DataFrame({'Selection': sel_daily, 'Construction': cons_daily, 'Alpha (P - B)': port.sub(bm, fill_value=np.nan)})

    if not alpha_daily.empty:
        last_date = alpha_daily.index.max()
        month_mask = alpha_daily.index.to_period('M') == last_date.to_period('M')
        mtd = alpha_daily.loc[month_mask].sum()
        bars = go.Figure(go.Bar(x=['Selection','Construction','Alpha (P - B)'], y=[mtd['Selection']*1e4, mtd['Construction']*1e4, mtd['Alpha (P - B)']*1e4], text=[f"{mtd['Selection']*1e4:.1f}", f"{mtd['Construction']*1e4:.1f}", f"{mtd['Alpha (P - B)']*1e4:.1f}"], textposition='auto'))
        bars.update_layout(title=f"MTD Decomposition — {str(last_date.to_period('M'))}", yaxis_title='bps')
    else:
        bars = go.Figure()

    return alpha_daily, bars


def monthly_returns_table(returns: pd.DataFrame, series: list[str] | None = None) -> pd.DataFrame:
    df = returns.copy()
    if series:
        df = df[df['Series'].isin(series)]
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    comp = df.groupby(['YearMonth','Series'])['Return'].apply(lambda x: (1+x.fillna(0)).prod()-1).reset_index()
    piv = comp.pivot(index='YearMonth', columns='Series', values='Return').sort_index()
    return piv


def mask_terminated_managers(returns: pd.DataFrame, navs: pd.DataFrame, manager_name: str, termination_date: str | None):
    inferred = None
    if termination_date:
        term = pd.to_datetime(termination_date)
    else:
        nav_m = navs[navs['Manager'].str.fullmatch(manager_name, case=False, na=False)]
        if nav_m.empty:
            nav_m = navs[navs['Manager'].str.contains(manager_name, case=False, na=False)]
        if not nav_m.empty:
            nav_m2 = nav_m.dropna(subset=['NAV'])
            nav_m2 = nav_m2[nav_m2['NAV']>0]
            if not nav_m2.empty:
                term = nav_m2['Date'].max()
                inferred = str(term.date())
            else:
                term = None
        else:
            term = None
    if term is not None:
        r = returns.copy(); n = navs.copy()
        r.loc[(r['Series'].str.fullmatch(manager_name, case=False) | r['Series'].str.contains(manager_name, case=False)) & (r['Date']>term), 'Return'] = np.nan
        n.loc[(n['Manager'].str.fullmatch(manager_name, case=False) | n['Manager'].str.contains(manager_name, case=False)) & (n['Date']>term), 'NAV'] = 0.0
        return r, n, inferred
    return returns, navs, inferred
