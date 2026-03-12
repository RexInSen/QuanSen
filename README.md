QuanSen_1.0
MPT Based Portfolio Optimizer 21:07 12-03-2026


************* THE GUI ********************


# QuanSen — Build Guide (Streamlit Edition)

## File layout

```
your_project_folder\
  ├── gui_portfolio.py          ← Streamlit GUI (your app)
  ├── portfolio_tool.py         ← Engine (imported by GUI)
  ├── quansen_splash.py         ← Splash launcher (this builds → QuanSen.exe)
  ├── quansen_splash.spec       ← PyInstaller spec for splash
  ├── quansen_app.spec          ← PyInstaller spec for Streamlit app
  ├── build_quansen.bat         ← One-click build script
  ├── quansen_installer.iss     ← Inno Setup script
  ├── quansen.ico               ← App icon
  └── quansen.png               ← Logo shown in splash
```

---

## How it works (and why the splash loop is impossible)

```
User double-clicks QuanSen.exe
    │
    ▼
Splash screen (tkinter, windowed, no console)
    │   shows logo + animated progress bar (~3 sec)
    │   spawns quansen_app.exe as DETACHED process
    │   fades out and exits completely
    ▼
quansen_app.exe  (Streamlit server + your GUI)
    │   starts on localhost:8501
    └── opens browser automatically → your Streamlit app
```

**Why no loop:** `QuanSen.exe` contains only tkinter + Pillow. It has zero
Streamlit/portfolio code and zero knowledge of itself. `quansen_app.exe` has
zero splash code. Neither can re-launch the other.

---

## Step 1 — Install dependencies

```
pip install pyinstaller pillow streamlit
pip install numpy pandas scipy cvxpy matplotlib seaborn plotly yfinance requests
```

Install **Inno Setup 6**: https://jrsoftware.org/isinfo.php

---

## Step 2 — Build the EXEs

Double-click `build_quansen.bat`

It will:
1. Auto-detect your Streamlit installation path
2. Patch `quansen_app.spec` with the real path
3. Build `quansen_app.exe` (Streamlit bundle)
4. Build `QuanSen.exe` (splash launcher)

Both land in the `dist\` folder.

---

## Step 3 — Test

```
cd dist
QuanSen.exe
```

Expected behaviour:
- Splash fades in showing your QS logo
- Progress bar animates ~3 seconds
- Splash fades out and disappears
- A console window opens briefly (Streamlit server starting)
- Your browser opens at `http://localhost:8501`
- **Splash does NOT reappear**

---

## Step 4 — Build the installer

1. Open `quansen_installer.iss` in Inno Setup Compiler
2. Press `Ctrl+F9`
3. Installer saved to `installer_output\QuanSen_Setup_v1.0.exe`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Splash opens but browser never launches | Make sure `quansen_app.exe` is in the same folder as `QuanSen.exe` |
| `ModuleNotFoundError` during build | `pip install <module>` then re-run build |
| Streamlit static assets missing (blank page) | The build script auto-patches the path — re-run `build_quansen.bat` |
| Port 8501 already in use | Close any other Streamlit instance, or edit the port in `quansen_splash.py` |
| Windows Defender blocks subprocess | Add an exclusion for your `dist\` folder |

---

## Do I need a separate launcher.py?

**No.** The splash IS the launcher. The flow is:

```
QuanSen.exe  →  spawns  →  quansen_app.exe
(splash)                   (streamlit run gui_portfolio.py)
```

A third "launcher.py" would just be a middleman that adds complexity and
another potential source of bugs. The two-exe architecture is the simplest,
most robust approach.









************* THE ENGINE ***************


# Quantitative Portfolio Optimizer v1.0

A Python tool for Modern Portfolio Theory (MPT) analysis.

## Pipeline
1. Ticker search & data download (Yahoo Finance)
2. Utility-maximized portfolio (CVXPY)
3. Correlation heatmap (Seaborn)
4. Efficient Frontier (CVXPY — 100 min-variance points)
5. Tangency Portfolio / Max Sharpe (Scipy SLSQP)
6. Static plot — Frontier + CML (Matplotlib)
7. Interactive plot + 3D Sharpe surface (Plotly)
8. Min-risk portfolio for a user-defined target return (CVXPY)

## Setup (one-time)

```bash
# 1. Clone or copy the folder to your machine
# 2. Create a virtual environment (recommended)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python portfolio_tool.py
```

## Configuration

Edit the top of `portfolio_tool.py` to change:
- `RF_ANNUAL = 0.075`  → risk-free rate (currently 7.5% for India)

## Notes
- Plotly charts open in your **browser** automatically
- If running inside Jupyter, change `pio.renderers.default = "notebook"`

