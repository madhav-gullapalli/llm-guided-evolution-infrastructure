"""Shared axis limits so slide charts compare fairly (chart normalization).

Override on the CLI with --ylim / --xlim / --auto-limits when a run needs autoscale.
Defaults target NAS-Bench-201 CIFAR-100 surrogate runs (τ ~0.5–0.6, acc ~68–73%).
"""

from __future__ import annotations

# Kendall τ (overall) — covers param baseline (~0.53) through observed best (~0.59)
TAU_YLIM = (0.0, 0.65)

# Top-10% τ is lower; keep a separate panel scale
TOP10_TAU_YLIM = (0.0, 0.45)

# τ on the x-axis of τ-vs-accuracy scatter (same window as TAU_YLIM)
TAU_XLIM = (0.0, 0.65)

# Guided / search accuracy on CIFAR-100 NB201 (opt ~73.26; seed-1 random ~68.7)
ACC_YLIM = (65.0, 74.0)

# Mean guided accuracy sits a bit lower than best
ACC_MEAN_YLIM = (60.0, 74.0)

# Fraction reaching within 1% of opt
REACH_FRAC_YLIM = (-0.05, 1.05)

# Anytime validation regret (opt − best-so-far) on CIFAR-100 NB201
REGRET_YLIM = (0.0, 8.0)
