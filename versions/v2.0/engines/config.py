"""
============================================================
  QuanSen — Shared Configuration & Constants
============================================================
  Edit this file to change global settings.
  All other modules import from here.
============================================================
"""

# ── Risk-free rate ───────────────────────────────────────────
RF_ANNUAL = 0.075        # 7.5% annual (change to match current T-bill/G-sec rate)
RF_DAILY  = RF_ANNUAL / 252

# ── Shrinkage default ────────────────────────────────────────
# Weight on each stock's own historical mean.
# (1 - SHRINKAGE_ALPHA) is pulled toward the benchmark mean.
#   1.0 = pure history, no shrinkage
#   0.0 = everything collapses to benchmark
#   0.7 = default: 70% own history, 30% market anchor
SHRINKAGE_ALPHA = 0.70

# ── Benchmark index candidates ───────────────────────────────
# Tried in order; first one with enough data wins.
INDIAN_BENCHMARKS = ["^NSEI", "^BSESN"]    # Nifty 50, BSE Sensex
GLOBAL_BENCHMARKS = ["^GSPC", "^IXIC"]     # S&P 500, Nasdaq
