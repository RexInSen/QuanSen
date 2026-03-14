# QuanSen_v1.1 — Quantitative Portfolio Optimizer

QuanSen is a Python-based portfolio optimization toolkit implementing
Modern Portfolio Theory (MPT) with an interactive GUI.

The system computes:

• Efficient Frontier
• Tangency Portfolio (Maximum Sharpe)
• Utility-optimized portfolio
• Correlation heatmap
* New added Backtesting software

The project is composed of two main modules:

| File                | Description                                |
| ------------------- | ------------------------------------------ |
| `portfolio_tool.py` | Core optimization engine implementing MPT  |
| `gui_portfolio.py`  | Streamlit GUI wrapper for user interaction |

---

# Features

✔ Efficient Frontier via convex optimization
✔ Tangency portfolio (maximum Sharpe ratio)
✔ Utility-maximization portfolio
✔ Correlation heatmap
✔ Benchmark shrinkage-adjusted expected returns
✔ NSE/BSE automatic ticker detection
✔ Interactive Streamlit interface

---

# Installation

Clone the repository

```
git clone https://github.com/<your-username>/QuanSen.git
cd QuanSen
```

Create virtual environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Running the GUI

```
streamlit run gui_portfolio.py
```

The application will open automatically in your browser.

---

# Example Workflow

1. Enter stock tickers
2. Select date range
3. Choose portfolio constraints
4. Generate:

• Efficient Frontier
• Tangency Portfolio
• Correlation matrix
• Monte Carlo simulations

---

# Mathematical Framework

The optimizer is based on **Modern Portfolio Theory (Markowitz)**.

Utility maximization

```
U = E[R] − ½ σ²
```

Tangency portfolio

```
max Sharpe = (E[R] − Rf) / σ
```

Efficient frontier

```
min σ²
subject to
E[R] ≥ target
```

Convex optimization is solved using **CVXPY (OSQP solver)**.

---

# Future Development

• Black-Litterman model (partially implemented via shrinkage)
• Factor models (Fama-French)
• Portfolio rebalancing tools
• Risk contribution analysis

---

# Author

Amatra Sen

---

# License

MIT License
