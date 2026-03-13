             QuanSen
MPT Based Portfolio Optimizer 21:07 12-03-2026

************* THE ENGINE ***************


# Quantitative Portfolio Optimizer # 

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

