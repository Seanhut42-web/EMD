# EMD FoF — Performance Explorer (v0.2)

**What’s new in v0.2**
- Directly parses **Tab 1 & 2** of *New Dashboard EMD.xlsx*: 
  - Tab 1 (default `EMD_Returns`) → returns time series (Portfolio, EMD Benchmark, and managers including **Overlay**)
  - Tab 2 (default `Market_Values`) → **NAVs** by manager (used for weights)
- **Units in percent** handled (divide by 100).
- **Schroders** post‑termination excluded (auto‑inferred from last positive NAV unless you specify a date).
- **Overlay** treated as a **manager**.
- Benchmark labelled **EMD Benchmark**. Manager labels: **Overlay, EMSO, Global Evolution, Neuberger Berman, Schroders**.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Place your workbook at `data/New Dashboard EMD.xlsx` (or point the sidebar to its path).

## Data recognition
The parser reads header columns that look like date **ranges** (e.g., `31-MAR-2025 to 01-APR-2025`) and uses the **end date** as the time stamp. It also handles single‑date headers. Rows like `Fixed Income`, `EMD`, `EMSO MASTER FUND`, `GLOBAL EVOLUTION`, `NEUBERGER BERMAN SHORT DURATION`, `SCHRODERS EMD`, `EMD OVERLAY` are mapped to canonical labels.

## Attribution math
- Weighted manager return:  \( r^{\text{Mgr}} = \sum w_i r_i \)
- Selection (daily): \( \sum w_i (r_i - r_B) \)
- Construction (daily): \( r_P - r^{\text{Mgr}} \)
- MTD = sum of daily values; monthly table is **compounded** from daily.


This is the flattened v0.2.1 package for GitHub/Streamlit Cloud. Main file path = `app.py`.
