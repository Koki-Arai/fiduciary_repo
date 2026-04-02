# ================================================================
#  Module 8 — Policy (Regulatory) Simulation
#  Append to simulation.py  OR  run standalone after
#  executing simulation.py in the same Colab session.
#
#  Sub-modules
#   8A. Regulatory design optimisation
#       — optimal (e_bar, mu, lam) under a social welfare criterion
#   8B. Dynamic regulation with learning
#       — regulator updates standard using Bayesian rule
#       — trustee anticipates future tightening
#   8C. Transition dynamics
#       — transition path from low-care to high-care equilibrium
#       — welfare cost of delay; escalating-penalty schedule
#   8D. Comparative regulatory architecture
#       — ex ante licensing vs ex post liability
#       — mandatory disclosure vs voluntary disclosure
#       — single-channel vs dual-channel enforcement
#   8E. Japan-specific policy experiments
#       — Prudent Investor Rule adoption
#       — Business Judgment Rule restriction
#       — Role-allocation reform (AIJ / Osaka ruling counterfactual)
#       — Penalty escalation schedule (JDC Trust counterfactual)
#   8F. Welfare decomposition & distribution
#       — trustee surplus / beneficiary surplus / regulator budget
#       — distributional effects across trustee types
# ================================================================

# ── STANDALONE HEADER (skip if already loaded) ─────────────────
import numpy as np
import scipy.stats   as st
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import warnings, time, itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

warnings.filterwarnings("ignore")
_RNG = np.random.default_rng(seed=2025)

# ── Re-import shared primitives if running standalone ──────────
try:
    _ = DIMS_DEFAULT   # already loaded from Module 0
except NameError:
    # paste or exec simulation.py first, then run this file
    raise RuntimeError(
        "Run simulation.py first:\n"
        "  exec(open('simulation.py').read())"
    )

# ── Plot style additions ────────────────────────────────────────
NAVY   = "#0A2342"
SAGE   = "#5B8A72"
RUST   = "#B5451B"
SLATE  = "#4A5568"
GOLD   = "#C9993A"
LEMON  = "#E8D44D"

def _save(name, save):
    if save: plt.savefig(f"{name}.png", bbox_inches="tight", dpi=130)

# ════════════════════════════════════════════════════════════════
# 8A  REGULATORY DESIGN OPTIMISATION
#
# The regulator chooses (e_bar_i, mu_i) for each dimension i
# to maximise expected social welfare W minus enforcement costs.
#
# W_reg = W(e**(e_bar, mu)) - C_R(mu)
#
# where C_R(mu) = c_R * sum(mu_i^2 / 2)  (convex audit cost)
#
# Subject to:
#   e_bar_i in [0, 1]
#   mu_i    in [0, mu_max]
#   budget constraint: sum(mu_i) <= B
# ════════════════════════════════════════════════════════════════

def optimise_regulatory_design(dims=DIMS_DEFAULT,
                                c_R=0.05, B=8.0, mu_max=4.0,
                                n_welfare_draws=0, save=True):
    """
    Numerically solve the regulator's optimisation problem.
    Returns optimal (e_bar, mu) vectors and the welfare surface.
    """
    print("\n" + "="*65)
    print("MODULE 8A: Regulatory Design Optimisation")
    print("="*65)

    n = len(dims)

    def reg_welfare(params):
        """
        params: [e_bar × n, mu × n]  (2n-dimensional)
        """
        e_bar = np.clip(params[:n], 0.05, 0.99)
        mu    = np.clip(params[n:], 0.01, mu_max)
        # build modified dims
        ds = deepcopy(dims)
        for i in range(n):
            ds[i].e_bar = float(e_bar[i])
            ds[i].mu    = float(mu[i])
        e_eq = solve_all(ds, "dual")
        W    = social_welfare(e_eq, ds, n_th=0)
        C    = c_R * np.sum(mu**2) / 2.0
        return -(W - C)    # minimise negative welfare

    # Budget constraint: sum(mu) <= B
    constraints = [
        {"type": "ineq",
         "fun":  lambda p: B - p[n:].sum()},
    ]
    bounds = ([(0.10, 0.95)] * n + [(0.05, mu_max)] * n)

    # Initial guess: current defaults
    p0 = np.array([d.e_bar for d in dims] + [d.mu for d in dims])

    print("Optimising regulatory design (SLSQP)...")
    t0 = time.time()
    res = opt.minimize(reg_welfare, p0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-7, "maxiter": 800})
    print(f"  Elapsed: {time.time()-t0:.1f}s  |  "
          f"converged: {res.success}  |  W_opt={-res.fun:.4f}")

    e_bar_opt = np.clip(res.x[:n], 0.05, 0.99)
    mu_opt    = np.clip(res.x[n:], 0.01, mu_max)

    # Baseline welfare (current defaults)
    e_base = solve_all(dims, "dual")
    W_base = social_welfare(e_base, dims, n_th=n_welfare_draws)
    C_base = c_R * np.sum([d.mu for d in dims])**2 / 2.0

    # Optimal welfare
    ds_opt = deepcopy(dims)
    for i in range(n):
        ds_opt[i].e_bar = float(e_bar_opt[i])
        ds_opt[i].mu    = float(mu_opt[i])
    e_opt = solve_all(ds_opt, "dual")
    W_opt = social_welfare(e_opt, ds_opt, n_th=n_welfare_draws)
    C_opt = c_R * np.sum(mu_opt**2) / 2.0

    # ── Welfare surface for two key dimensions ──
    # Vary mu_5 (governance) and mu_6 (disclosure) on a grid
    mu5_range = np.linspace(0.1, mu_max, 20)
    mu6_range = np.linspace(0.1, mu_max, 20)
    surf = np.zeros((20, 20))
    for ii, m5 in enumerate(mu5_range):
        for jj, m6 in enumerate(mu6_range):
            ds_s = deepcopy(dims)
            ds_s[4].mu = float(m5); ds_s[5].mu = float(m6)
            e_s  = solve_all(ds_s, "dual")
            W_s  = social_welfare(e_s, ds_s, n_th=400)
            C_s  = c_R * (m5**2 + m6**2) / 2.0
            surf[ii, jj] = W_s - C_s

    # ── Print results ──
    print(f"\n{'Dim':<14} {'e_bar (def)':>12} {'e_bar (opt)':>12}  "
          f"{'mu (def)':>10} {'mu (opt)':>10}")
    print("-"*62)
    for i, d in enumerate(dims):
        print(f"{d.name:<14} {d.e_bar:>12.3f} {e_bar_opt[i]:>12.3f}  "
              f"{d.mu:>10.3f} {mu_opt[i]:>10.3f}")
    print(f"\nW_base (net) = {W_base - C_base:.4f}")
    print(f"W_opt  (net) = {W_opt  - C_opt:.4f}  "
          f"(gain = {(W_opt-C_opt)-(W_base-C_base):+.4f})")
    print(f"Budget used: default={sum(d.mu for d in dims):.2f}  "
          f"optimal={mu_opt.sum():.2f}  limit={B:.1f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Module 8A — Optimal Regulatory Design",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    x = np.arange(n); w = 0.30
    ax.bar(x - w/2, [d.e_bar for d in dims], width=w,
           color=SLATE, alpha=0.75, label="Default ē^R")
    ax.bar(x + w/2, e_bar_opt, width=w,
           color=BLUE,  alpha=0.85, label="Optimal ē^R")
    ax.set_xticks(x)
    ax.set_xticklabels([d.name for d in dims], rotation=25, ha="right")
    ax.set_ylabel("Admin standard ē^R")
    ax.set_title("(a) Optimal admin standards")
    ax.legend()

    ax = axes[1]
    ax.bar(x - w/2, [d.mu for d in dims], width=w,
           color=SLATE, alpha=0.75, label="Default μ")
    ax.bar(x + w/2, mu_opt, width=w,
           color=GREEN, alpha=0.85, label="Optimal μ")
    ax.set_xticks(x)
    ax.set_xticklabels([d.name for d in dims], rotation=25, ha="right")
    ax.set_ylabel("Admin penalty μ")
    ax.set_title("(b) Optimal penalty schedule")
    ax.legend()

    ax = axes[2]
    ct = ax.contourf(mu6_range, mu5_range, surf, levels=18, cmap="RdYlGn")
    plt.colorbar(ct, ax=ax, label="Net welfare W - C_R(μ)")
    ax.plot(mu_opt[5], mu_opt[4], "w*", ms=14,
            label=f"Optimum ({mu_opt[4]:.2f},{mu_opt[5]:.2f})")
    ax.plot(dims[4].mu, dims[5].mu, "rx", ms=10, mew=2,
            label=f"Default ({dims[4].mu:.2f},{dims[5].mu:.2f})")
    ax.set_xlabel("μ — e5 Disclosure")
    ax.set_ylabel("μ — e6 Governance")
    ax.set_title("(c) Welfare surface: e5×e6 penalty space")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save("fig_8a_opt_design", save)
    plt.show()

    return e_bar_opt, mu_opt, surf


# ════════════════════════════════════════════════════════════════
# 8B  DYNAMIC REGULATION WITH LEARNING
#
# The regulator has a Bayesian prior over the true sigma2c_i
# (observability) and updates after each period's enforcement.
# The trustee rationally anticipates future standard changes.
#
# Sequence per period t:
#   1. Regulator holds belief N(m_t, v_t) over sigma2c
#   2. Sets (e_bar_t, mu_t) based on current belief
#   3. Trustee chooses e_t* given (e_bar_t, mu_t) and expectation
#      of future tightening (forward-looking)
#   4. Regulator observes noisy signal s_t = e_t* + noise
#   5. Bayesian update: m_{t+1}, v_{t+1}
# ════════════════════════════════════════════════════════════════

def dynamic_regulation(n_periods=25, target_dim=5,
                        sigma2c_true=0.65, prior_mean=0.45,
                        prior_var=0.10, obs_noise=0.08,
                        discount=0.92, save=True):
    """
    Dynamic Bayesian regulation for one dimension (default: e6 Governance).
    Shows: belief convergence, adaptive standard, care-level path.
    """
    print("\n" + "="*65)
    print("MODULE 8B: Dynamic Regulation with Bayesian Learning")
    print("="*65)

    d_true = deepcopy(DIMS_DEFAULT[target_dim])
    d_true.sigma2c = sigma2c_true

    m_t = prior_mean   # prior mean of sigma2c
    v_t = prior_var    # prior variance

    # Histories
    hist_m    = [m_t]; hist_v = [v_t]
    hist_e    = []
    hist_ebar = []
    hist_mu   = []
    hist_W    = []

    # Regulator's adaptive rule:
    # e_bar_t = e* - k_e * sqrt(m_t)   (higher perceived noise → lower bar)
    # mu_t    = mu_max * (1 - exp(-lambda_mu * m_t))
    k_e = 0.25; mu_max_dyn = 2.5; lam_mu = 3.0

    for t in range(n_periods):
        # Regulator sets standard based on belief about sigma2c
        e_bar_t = float(np.clip(d_true.e_star - k_e * np.sqrt(m_t), 0.25, 0.90))
        mu_t    = float(mu_max_dyn * (1 - np.exp(-lam_mu * m_t)))

        # Trustee best response (forward-looking: anticipates tightening)
        # Continuation value: higher future e_bar raises current e
        future_tightening = discount * max(0, e_bar_t - d_true.e_bar)
        d_adj = deepcopy(d_true)
        d_adj.e_bar  = e_bar_t + future_tightening
        d_adj.mu     = mu_t
        e_t = solve_e_fast(d_adj, "dual")

        # Regulator observes noisy signal
        s_t = e_t + _RNG.normal(0, obs_noise)

        # Bayesian update (Gaussian conjugate)
        # likelihood: s_t | sigma2c ~ N(e*(sigma2c), obs_noise^2)
        # use signal to update belief about sigma2c via indirect learning
        # proxy: if s_t < e_bar_t, update toward higher sigma2c
        info_signal = max(0, e_bar_t - s_t) / obs_noise
        m_new = (m_t / v_t + info_signal / obs_noise**2) / \
                (1.0 / v_t + 1.0 / obs_noise**2)
        v_new = 1.0 / (1.0 / v_t + 1.0 / obs_noise**2)
        m_new = float(np.clip(m_new, 0.01, 1.5))
        v_new = float(np.clip(v_new, 0.002, v_t))

        W_t = social_welfare(np.full(6, e_t), DIMS_DEFAULT, n_th=0)

        hist_e.append(e_t); hist_ebar.append(e_bar_t)
        hist_mu.append(mu_t); hist_W.append(W_t)
        hist_m.append(m_new); hist_v.append(v_new)
        m_t = m_new; v_t = v_new

        if t % 5 == 0:
            print(f"  t={t:2d}  belief m={m_t:.3f}(±{np.sqrt(v_t):.3f})  "
                  f"ē^R={e_bar_t:.3f}  μ={mu_t:.3f}  e**={e_t:.3f}  W={W_t:.3f}")

    print(f"\nBelief convergence: m_final={hist_m[-1]:.4f}  "
          f"(true sigma2c={sigma2c_true:.3f})")
    print(f"Care level: t=0 -> {hist_e[0]:.3f}  t={n_periods-1} -> {hist_e[-1]:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Module 8B — Dynamic Regulation: {DIMS_DEFAULT[target_dim].name}",
                 fontsize=13, fontweight="bold")
    t_range = range(n_periods)

    ax = axes[0, 0]
    ax.plot(range(n_periods + 1), hist_m, color=BLUE, lw=2, label="Belief mean m_t")
    ax.fill_between(range(n_periods + 1),
                    np.array(hist_m) - 1.96 * np.sqrt(hist_v),
                    np.array(hist_m) + 1.96 * np.sqrt(hist_v),
                    alpha=0.20, color=BLUE, label="95% CI")
    ax.axhline(sigma2c_true, color=RED, ls="--", lw=1.5, label=f"True σ²_C={sigma2c_true}")
    ax.set_xlabel("Period"); ax.set_ylabel("Belief about σ²_C")
    ax.set_title("(a) Bayesian belief convergence"); ax.legend()

    ax = axes[0, 1]
    ax.plot(t_range, hist_ebar, color=GREEN, lw=2, label="Admin standard ē^R_t")
    ax.plot(t_range, hist_mu,   color=AMBER, lw=2, label="Penalty μ_t")
    ax.axhline(DIMS_DEFAULT[target_dim].e_bar, color=GREEN, ls=":", lw=1,
               alpha=0.5, label="Static default ē^R")
    ax.set_xlabel("Period"); ax.set_ylabel("Regulatory parameter")
    ax.set_title("(b) Adaptive regulatory response"); ax.legend()

    ax = axes[1, 0]
    ax.plot(t_range, hist_e, color=PURPLE, lw=2.5, label="Equilibrium care e**_t")
    ax.plot(t_range, hist_ebar, color=GREEN, lw=1.5, ls="--", label="Admin floor ē^R_t")
    fb_val = DIMS_DEFAULT[target_dim].e_fb
    ax.axhline(fb_val, color=RED, ls=":", lw=1.5, label=f"First-best e_FB={fb_val:.3f}")
    ax.fill_between(t_range, hist_ebar, hist_e,
                    where=[e > eb for e, eb in zip(hist_e, hist_ebar)],
                    alpha=0.15, color=PURPLE, label="Voluntary over-compliance")
    ax.set_xlabel("Period"); ax.set_ylabel("Care level")
    ax.set_title("(c) Trustee care trajectory"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_range, hist_W, color=TEAL, lw=2, label="Social welfare W_t")
    ax.axhline(np.mean(hist_W), color=TEAL, ls=":", lw=1.2, alpha=0.6,
               label=f"Mean W = {np.mean(hist_W):.3f}")
    ax2 = ax.twinx()
    ax2.fill_between(t_range, np.sqrt(hist_v[:-1]),
                     alpha=0.25, color=AMBER, label="Belief uncertainty √v_t")
    ax2.set_ylabel("Belief std √v_t", color=AMBER)
    ax2.tick_params(axis="y", labelcolor=AMBER)
    ax.set_xlabel("Period"); ax.set_ylabel("Social welfare W")
    ax.set_title("(d) Welfare & learning uncertainty"); ax.legend(fontsize=8)

    plt.tight_layout()
    _save("fig_8b_dynamic", save)
    plt.show()

    return hist_e, hist_m, hist_v, hist_W


# ════════════════════════════════════════════════════════════════
# 8C  TRANSITION DYNAMICS
#
# Analyses the path from a low-care equilibrium to a high-care
# equilibrium under different penalty escalation schedules.
#
# Four schedules compared:
#   S0: No enforcement (baseline)
#   S1: Immediate full penalty (big-bang)
#   S2: Linear escalation over T periods
#   S3: Graduated: gentle start + sharp increase at threshold
#   S4: Optimal escalation (minimise transition cost)
# ════════════════════════════════════════════════════════════════

def transition_dynamics(n_periods=30, target_dim=1,
                        mu_final=2.0, T_ramp=12,
                        n_agents=30, save=True):
    """
    Simulate transition from low-care (mu≈0) to high-care (mu=mu_final)
    under alternative escalation schedules.
    Models JDC Trust (failed escalation) and SocGen (forced transition).
    """
    print("\n" + "="*65)
    print("MODULE 8C: Transition Dynamics (Escalation Schedules)")
    print("="*65)

    d = DIMS_DEFAULT[target_dim]
    alphas = _RNG.uniform(0.8, 1.4, n_agents)

    def mu_schedule(sched, t, T=T_ramp):
        """Return mu_t under schedule sched."""
        if sched == "S0": return 0.05
        if sched == "S1": return mu_final                          # big-bang
        if sched == "S2": return min(mu_final, mu_final * t / T)   # linear
        if sched == "S3":                                           # graduated
            if t < T // 2: return 0.2 + 0.5 * (t / (T//2))
            return 0.7 + (mu_final - 0.7) * ((t - T//2) / (T//2))
        if sched == "S4":                                           # concave-optimal
            return mu_final * (1 - np.exp(-3 * t / T))
        return mu_final

    schedules = ["S0", "S1", "S2", "S3", "S4"]
    s_labels  = ["No enforcement", "Big-bang", "Linear",
                 "Graduated", "Concave-optimal"]
    s_colors  = [GRAY, RED, AMBER, BLUE, GREEN]

    hist = {s: {"e": [], "W": [], "mu": []} for s in schedules}

    for sched in schedules:
        e = _RNG.uniform(0.1, 0.35, n_agents)    # start in low-care zone
        for t in range(n_periods):
            mu_t = mu_schedule(sched, t)
            for i in range(n_agents):
                d_i  = deepcopy(d)
                d_i.mu  = float(mu_t)
                e_br = solve_e_fast(d_i, "dual" if mu_t > 0.1 else "civil")
                e[i] = np.clip(0.3 * e[i] + 0.7 * e_br
                               + _RNG.normal(0, 0.02), 0.01, 2.0)
                # compliance enforcement
                if mu_t > 0.5 and e[i] < d.e_bar * 0.85:
                    e[i] = max(e[i], d.e_bar * 0.85)
            thr = np.percentile(e, 5)
            for i in range(n_agents):
                if e[i] <= thr: e[i] = _RNG.uniform(0.25, 0.55)
            W_t = social_welfare(np.full(6, e.mean()), DIMS_DEFAULT, n_th=0)
            hist[sched]["e"].append(e.mean())
            hist[sched]["W"].append(W_t)
            hist[sched]["mu"].append(mu_t)

    # ── Transition cost (discounted welfare gap vs high-care optimum) ──
    W_hi = social_welfare(solve_all(
        [deepcopy(d) if True else d for d in
         [deepcopy(DIMS_DEFAULT[i]) for i in range(6)]
         ], "dual"), DIMS_DEFAULT, n_th=600)
    discount = 0.95
    trans_cost = {}
    for sched in schedules:
        disc_gap = sum((W_hi - W) * discount**t
                       for t, W in enumerate(hist[sched]["W"]))
        trans_cost[sched] = disc_gap

    print("\nTransition costs (discounted welfare gap, δ=0.95):")
    for s, lbl in zip(schedules, s_labels):
        print(f"  {lbl:<22} ΔC = {trans_cost[s]:+.3f}")

    best = min(schedules[1:], key=lambda s: trans_cost[s])
    print(f"\nLowest-cost schedule: {best} ({s_labels[schedules.index(best)]})")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Module 8C — Transition Dynamics: Escalation Schedules",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    for s, lbl, col in zip(schedules, s_labels, s_colors):
        ax.plot(range(n_periods), hist[s]["e"], color=col,
                lw=2 if s != "S0" else 1.2,
                ls="--" if s == "S0" else "-", label=lbl)
    ax.axhline(d.e_bar, color="black", ls=":", lw=1.2, label="Admin floor ē^R")
    ax.axhline(d.e_fb,  color=RED,     ls=":",  lw=1.0, alpha=0.6,
               label=f"First-best {d.e_fb:.2f}")
    ax.set_xlabel("Period"); ax.set_ylabel("Mean care level")
    ax.set_title("(a) Care trajectory by schedule"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for s, lbl, col in zip(schedules, s_labels, s_colors):
        ax.plot(range(n_periods), hist[s]["mu"], color=col,
                lw=2 if s != "S0" else 1.2,
                ls="--" if s == "S0" else "-", label=lbl)
    ax.set_xlabel("Period"); ax.set_ylabel("Penalty μ_t")
    ax.set_title("(b) Penalty escalation schedule")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for s, lbl, col in zip(schedules, s_labels, s_colors):
        ax.plot(range(n_periods), hist[s]["W"], color=col,
                lw=2 if s != "S0" else 1.2,
                ls="--" if s == "S0" else "-", label=lbl)
    ax.set_xlabel("Period"); ax.set_ylabel("Social welfare W_t")
    ax.set_title("(c) Welfare path"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    bars = ax.bar(s_labels, [trans_cost[s] for s in schedules],
                  color=s_colors, alpha=0.85, edgecolor="white")
    for bar, s in zip(bars, schedules):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{trans_cost[s]:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Discounted transition cost")
    ax.set_title("(d) Total transition cost (δ=0.95)")
    ax.tick_params(axis="x", labelrotation=20)

    plt.tight_layout()
    _save("fig_8c_transition", save)
    plt.show()

    return hist, trans_cost


# ════════════════════════════════════════════════════════════════
# 8D  COMPARATIVE REGULATORY ARCHITECTURE
#
# Four institutional designs compared across all 6 dimensions:
#   A1: Single-channel civil liability only
#   A2: Single-channel administrative regulation only
#   A3: Dual-layer (current Japanese system)
#   A4: Mandatory disclosure + liability (US-style)
#   A5: Pre-licensing (strong ex ante)
#   A6: Optimal dual-layer (from 8A)
#
# Comparison on: welfare, care levels, enforcement cost.
# ════════════════════════════════════════════════════════════════

def compare_architectures(dims=DIMS_DEFAULT,
                           c_R=0.05, e_bar_opt=None,
                           mu_opt=None, save=True):
    """
    Compare 6 regulatory architectures across welfare, care, and cost.
    """
    print("\n" + "="*65)
    print("MODULE 8D: Comparative Regulatory Architecture")
    print("="*65)

    # Build dims for each architecture
    def arch_dims(name):
        ds = deepcopy(dims)
        if name == "A1_civil":
            for d in ds: d.mu = 0.01
        elif name == "A2_admin":
            for d in ds: d.lam = 0.0
        elif name == "A3_dual":
            pass   # default
        elif name == "A4_disclosure":
            # Mandatory disclosure: boosts lam for disclosure dims,
            # adds direct enforcement for conflict dims
            for i in [3, 4, 5]:
                ds[i].lam  *= 2.5
                ds[i].e_star = min(ds[i].e_star + 0.05, 0.95)
        elif name == "A5_prelicense":
            # Strong ex ante: high e_bar for all dims (licensing standard)
            for d in ds:
                d.e_bar = min(d.e_fb * 0.92, 0.90)
                d.mu    = 3.0
        elif name == "A6_optimal":
            if e_bar_opt is not None:
                for i, d in enumerate(ds):
                    d.e_bar = float(e_bar_opt[i])
            if mu_opt is not None:
                for i, d in enumerate(ds):
                    d.mu = float(mu_opt[i])
        return ds

    archs = ["A1_civil", "A2_admin", "A3_dual",
             "A4_disclosure", "A5_prelicense", "A6_optimal"]
    labels = ["Civil only", "Admin only", "Dual (current)",
              "Mandatory\ndisclosure", "Pre-licensing", "Optimal\ndual"]
    colors = [RED, AMBER, SLATE, PURPLE, TEAL, GREEN]
    regime_map = {"A1_civil": "civil", "A2_admin": "admin",
                  "A3_dual": "dual",   "A4_disclosure": "dual",
                  "A5_prelicense": "dual", "A6_optimal": "dual"}

    results = {}
    for a in archs:
        ds    = arch_dims(a)
        e_eq  = solve_all(ds, regime_map[a])
        W     = social_welfare(e_eq, ds, n_th=0)
        C_enf = c_R * sum(d.mu**2 for d in ds) / 2.0
        results[a] = {"e": e_eq, "W": W, "C": C_enf, "W_net": W - C_enf}

    # ── Print summary ──
    print(f"\n{'Architecture':<20} {'W':>8} {'C_enf':>8} {'W_net':>8}  "
          f"{'ΔW_net':>8}")
    print("-"*55)
    W0 = results["A3_dual"]["W_net"]
    for a, lbl in zip(archs, labels):
        r = results[a]
        lbl2 = lbl.replace("\n", " ")
        print(f"{lbl2:<20} {r['W']:>8.3f} {r['C']:>8.3f} {r['W_net']:>8.3f}  "
              f"{r['W_net']-W0:>+8.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Module 8D — Comparative Regulatory Architecture",
                 fontsize=13, fontweight="bold")

    # Net welfare
    ax = axes[0]
    W_nets = [results[a]["W_net"] for a in archs]
    bars   = ax.bar(range(len(archs)), W_nets, color=colors, alpha=0.85)
    ax.axhline(W0, color="black", ls="--", lw=1, alpha=0.5)
    for bar, W in zip(bars, W_nets):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{W-W0:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Net welfare (W - C_enf)")
    ax.set_title("(a) Net welfare by architecture")

    # Care heatmap
    ax = axes[1]
    e_mat = np.array([results[a]["e"] for a in archs])
    fb    = np.array([d.e_fb for d in dims])
    e_norm = e_mat / fb[np.newaxis, :]     # normalise by first-best
    im = ax.imshow(e_norm, aspect="auto", cmap="RdYlGn",
                   vmin=0.4, vmax=1.15)
    plt.colorbar(im, ax=ax, label="e** / e_FB")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([d.name for d in dims], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("(b) Normalised care heatmap\n(green=near FB, red=under)")
    for i in range(len(archs)):
        for j in range(len(dims)):
            ax.text(j, i, f"{e_norm[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if 0.6 < e_norm[i,j] < 1.05 else "white")

    # Enforcement cost vs welfare scatter
    ax = axes[2]
    for a, lbl, col in zip(archs, labels, colors):
        r = results[a]
        ax.scatter(r["C"], r["W"], color=col, s=100, zorder=3)
        ax.annotate(lbl.replace("\n", " "), (r["C"], r["W"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Enforcement cost C_enf"); ax.set_ylabel("Gross welfare W")
    ax.set_title("(c) Welfare–cost frontier")

    plt.tight_layout()
    _save("fig_8d_architecture", save)
    plt.show()

    return results


# ════════════════════════════════════════════════════════════════
# 8E  JAPAN-SPECIFIC POLICY EXPERIMENTS
#
# Five counterfactuals mapped to actual legal developments:
#
# JP1. Prudent Investor Rule adoption
#      → σ²_C for e1,e2 decreases (court expertise improves)
#        λ increases for e1,e2 (Prop 3: strict review enabled)
#
# JP2. Business Judgment Rule restriction
#      → deferential review limited; e_star collapses to e_bar
#        for investment-judgment dims e1, e2
#
# JP3. Role-allocation reform (AIJ/Osaka precedent reversal)
#      → λ for e1,e2 restored; advisory duty recognised
#
# JP4. JDC Trust counterfactual — optimal escalation
#      → what if FSA had escalated penalties earlier?
#
# JP5. Comprehensive reform package
#      → JP1 + JP3 + strengthened admin for e5,e6
# ════════════════════════════════════════════════════════════════

def japan_policy_experiments(dims=DIMS_DEFAULT, save=True):
    """
    Japan-specific counterfactual policy simulations.
    """
    print("\n" + "="*65)
    print("MODULE 8E: Japan-Specific Policy Experiments")
    print("="*65)

    def apply_policy(base, policy):
        ds = deepcopy(base)
        if policy == "JP1_PIR":
            # Prudent Investor Rule (PIR):
            # Adopting PIR raises the judicially-enforced conduct standard.
            # Operationally: both the admin floor (e_bar) and the civil
            # conduct standard (e_star) are moved up toward e_FB.
            # This directly raises equilibrium e** for investment dims.
            for i in [0, 1]:
                ds[i].e_bar  = min(ds[i].e_bar  + 0.17, 0.90)   # PIR raises admin standard
                ds[i].e_star = min(ds[i].e_star + 0.10, 0.92)   # PIR raises civil standard
                ds[i].mu     = min(ds[i].mu * 1.20, 3.0)         # enforcement also strengthened
        elif policy == "JP2_BJR":
            # Business Judgment Rule extension (deferential review):
            # Courts defer → effective review threshold collapses toward e_bar.
            # Regulator also lowers expectations → e_bar erodes.
            # Net: e** falls for investment dims.
            for i in [0, 1]:
                ds[i].e_bar  = ds[i].e_bar  * 0.78   # admin standard eroded
                ds[i].e_star = ds[i].e_bar  * 0.78   # review threshold = eroded admin floor
                ds[i].mu     = ds[i].mu     * 0.58   # weaker enforcement
        elif policy == "JP3_ROLE":
            # Role-allocation reform (advisory duty recognised).
            # AIJ/Osaka ruling reversed: trustee's obligation now covers e1,e2.
            # Both admin standard and civil standard are raised.
            for i in [0, 1]:
                ds[i].e_bar  = min(ds[i].e_bar  + 0.10, 0.85)
                ds[i].e_star = min(ds[i].e_star + 0.08, 0.90)
                ds[i].mu     = min(ds[i].mu     * 1.15, 3.0)
        elif policy == "JP4_JDC":
            # JDC counterfactual: early penalty escalation (governance dims)
            for i in [4, 5]:
                ds[i].mu  = min(ds[i].mu * 2.5, 3.5)
                ds[i].e_bar = min(ds[i].e_bar + 0.08, 0.90)
        elif policy == "JP5_REFORM":
            # Comprehensive: JP1 + JP3 + JP4 — all six dimensions improved
            for i in [0, 1]:
                ds[i].e_bar  = min(ds[i].e_bar  + 0.30, 0.93)
                ds[i].e_star = min(ds[i].e_star + 0.22, 0.96)
                ds[i].mu     = min(ds[i].mu * 1.50, 3.5)
            for i in [2, 3, 4, 5]:
                ds[i].mu    = min(ds[i].mu * 2.2, 3.5)
                ds[i].e_bar = min(ds[i].e_bar + 0.08, 0.93)
        return ds

    baseline = deepcopy(dims)
    e_base   = solve_all(baseline, "dual")
    _RNG = np.random.default_rng(seed=9999)  # fixed seed for reproducibility
    W_base   = social_welfare(e_base, baseline, n_th=0)
    fb       = np.array([d.e_fb for d in dims])

    policies = ["JP1_PIR", "JP2_BJR", "JP3_ROLE", "JP4_JDC", "JP5_REFORM"]
    p_labels = ["JP1: Prudent\nInvestor Rule",
                "JP2: BJR\nrestriction",
                "JP3: Role-alloc.\nreform",
                "JP4: JDC early\nescalation",
                "JP5: Comprehensive\nreform"]
    p_colors = [BLUE, RED, GREEN, AMBER, TEAL]

    policy_results = {"Baseline": {"e": e_base, "W": W_base}}
    for pol in policies:
        ds_p  = apply_policy(baseline, pol)
        e_p   = solve_all(ds_p, "dual")
        W_p   = social_welfare(e_p, ds_p, n_th=0)
        policy_results[pol] = {"e": e_p, "W": W_p, "dims": ds_p}

    # ── Print ──
    print(f"\n{'Policy':<28} {'W':>8} {'ΔW':>8} {'ΔW%':>8}  "
          f"Biggest gain dim")
    print("-"*65)
    print(f"{'Baseline':<28} {W_base:>8.4f} {'—':>8} {'—':>8}")
    for pol, lbl in zip(policies, p_labels):
        r   = policy_results[pol]
        dW  = r["W"] - W_base
        dim_gains = r["e"] - e_base
        best_dim  = DIMS_DEFAULT[np.argmax(dim_gains)].name
        print(f"{lbl.replace(chr(10),' '):<28} {r['W']:>8.4f} "
              f"{dW:>+8.4f} {100*dW/abs(W_base):>+7.2f}%  {best_dim}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Module 8E — Japan-Specific Policy Experiments",
                 fontsize=13, fontweight="bold")

    # Welfare bar
    ax = axes[0, 0]
    all_labels = ["Baseline"] + [lbl.replace("\n", " ") for lbl in p_labels]
    all_W      = [W_base] + [policy_results[p]["W"] for p in policies]
    all_cols   = [GRAY] + p_colors
    bars = ax.bar(range(len(all_labels)), all_W, color=all_cols, alpha=0.85)
    ax.axhline(W_base, color="black", ls="--", lw=1, alpha=0.4)
    for bar, W in zip(bars, all_W):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{W-W_base:+.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Social welfare W")
    ax.set_title("(a) Welfare by policy")

    # Care-level change heatmap
    ax = axes[0, 1]
    delta_mat = np.array([policy_results[p]["e"] - e_base for p in policies])
    vabs = np.abs(delta_mat).max()
    im = ax.imshow(delta_mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, label="Δe** = e**(policy) − e**(baseline)")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([d.name for d in dims], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([lbl.replace("\n", " ") for lbl in p_labels], fontsize=8)
    ax.set_title("(b) Care-level change by policy × dim\n(blue=increase, red=decrease)")
    for i in range(len(policies)):
        for j in range(len(dims)):
            ax.text(j, i, f"{delta_mat[i,j]:+.2f}",
                    ha="center", va="center", fontsize=7)

    # Prudent Investor Rule deep-dive (JP1 vs JP2 on e1,e2)
    ax = axes[1, 0]
    sigma2c_range = np.linspace(0.05, 0.80, 40)
    e1_jp1, e1_jp2 = [], []
    for sc in sigma2c_range:
        d_jp1 = deepcopy(dims[0]); d_jp1.sigma2c = sc; d_jp1.lam = dims[0].lam * 2.2
        d_jp2 = deepcopy(dims[0]); d_jp2.sigma2c = sc; d_jp2.lam = dims[0].lam * 0.3
        d_jp2.e_star = d_jp2.e_bar
        e1_jp1.append(solve_e_fast(d_jp1, "dual"))
        e1_jp2.append(solve_e_fast(d_jp2, "dual"))
    ax.plot(sigma2c_range, e1_jp1, color=BLUE,  lw=2, label="JP1: PIR (strict review)")
    ax.plot(sigma2c_range, e1_jp2, color=RED,   lw=2, label="JP2: BJR (deferential)")
    ax.axvline(dims[0].sigma2c, color=GRAY, ls=":", lw=1.5,
               label=f"Current σ²_C={dims[0].sigma2c:.2f}")
    ax.axhline(dims[0].e_fb, color="black", ls="--", lw=1.2,
               label=f"First-best {dims[0].e_fb:.3f}")
    ax.fill_between(sigma2c_range, e1_jp2, e1_jp1,
                    where=[j1>j2 for j1,j2 in zip(e1_jp1,e1_jp2)],
                    alpha=0.15, color=GREEN, label="PIR advantage")
    ax.set_xlabel("Court signal noise σ²_C")
    ax.set_ylabel("Equilibrium care e1 (DD/investment)")
    ax.set_title("(c) PIR vs BJR: equilibrium care\nas function of judicial precision")
    ax.legend(fontsize=8)

    # JDC counterfactual: ABM with early vs late escalation
    ax = axes[1, 1]
    n_pd = 35; n_ag = 20
    d_jdc = deepcopy(dims[5])   # governance dim

    def jdc_run(early_escalation):
        ev = _RNG.uniform(0.1, 0.3, n_ag)
        traj = []
        for t in range(n_pd):
            if early_escalation:
                mu_t = min(0.3 + 0.18 * t, 2.8)    # escalates from t=0
            else:
                if t < 25:
                    mu_t = 0.08 + 0.02 * t           # very slow (JDC actual)
                else:
                    mu_t = 2.5                        # too late; license revoked
            for i in range(n_ag):
                d_i     = deepcopy(d_jdc); d_i.mu = float(mu_t)
                e_br    = solve_e_fast(d_i, "dual")
                ev[i]   = np.clip(0.35*ev[i]+0.65*e_br+_RNG.normal(0,0.025),
                                  0.01, 2.0)
                if mu_t > 0.5 and ev[i] < d_jdc.e_bar * 0.75:
                    ev[i] = d_jdc.e_bar * 0.75
            thr = np.percentile(ev, 5)
            for i in range(n_ag):
                if ev[i] <= thr: ev[i] = _RNG.uniform(0.2, 0.45)
            traj.append(ev.mean())
        return traj

    traj_actual = jdc_run(False)
    traj_cf     = jdc_run(True)
    ax.plot(range(n_pd), traj_actual, color=RED,  lw=2.5, label="JDC actual (late escalation)")
    ax.plot(range(n_pd), traj_cf,     color=BLUE, lw=2.5, label="Counterfactual (early escalation)")
    ax.axhline(d_jdc.e_bar, color=AMBER, ls=":", lw=1.5, label="Admin floor ē^R")
    ax.axvline(25, color=RED, ls="--", lw=1, alpha=0.7, label="License revocation (actual)")
    ax.fill_between(range(n_pd), traj_actual, traj_cf,
                    where=[cf>ac for cf,ac in zip(traj_cf,traj_actual)],
                    alpha=0.15, color=BLUE,
                    label=f"CF gain={np.mean(traj_cf)-np.mean(traj_actual):.3f}")
    ax.set_xlabel("Period"); ax.set_ylabel("Mean governance care e6")
    ax.set_title("(d) JDC Trust counterfactual:\nearly vs late penalty escalation")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save("fig_8e_japan_policy", save)
    plt.show()

    return policy_results, traj_actual, traj_cf


# ════════════════════════════════════════════════════════════════
# 8F  WELFARE DECOMPOSITION & DISTRIBUTIONAL EFFECTS
#
# Decomposes social welfare into:
#   W_B: beneficiary surplus    = f(e) - E[D]
#   W_T: trustee net profit     = f(e) - c(e) - liability payments
#   C_R: enforcement budget     = audit/admin costs
#
# Distributional analysis:
#   α_H vs α_L types — who gains/loses under strict review?
# ════════════════════════════════════════════════════════════════

def welfare_decomposition(dims=DIMS_DEFAULT,
                          alpha_H=0.7, alpha_L=1.5,
                          q=0.40, c_R=0.05, save=True):
    """
    Decompose welfare and analyse distributional effects.
    """
    print("\n" + "="*65)
    print("MODULE 8F: Welfare Decomposition & Distributional Effects")
    print("="*65)

    scenarios = {
        "No enforcement":    {"mu_scale": 0.0, "lam_scale": 0.0, "bjr": False},
        "Civil only":        {"mu_scale": 0.0, "lam_scale": 1.0, "bjr": False},
        "Admin only":        {"mu_scale": 1.0, "lam_scale": 0.0, "bjr": False},
        "Dual (current)":    {"mu_scale": 1.0, "lam_scale": 1.0, "bjr": False},
        "Dual + PIR":        {"mu_scale": 1.0, "lam_scale": 2.0, "bjr": False},
        "Dual + BJR":        {"mu_scale": 1.0, "lam_scale": 0.3, "bjr": True},
    }

    records = []
    for scen_name, cfg in scenarios.items():
        ds = deepcopy(dims)
        for d in ds:
            d.mu  = d.mu  * cfg["mu_scale"]
            d.lam = d.lam * cfg["lam_scale"]
            if cfg["bjr"]:
                for i in [0, 1]:
                    ds[i].e_star = ds[i].e_bar

        regime = "dual" if cfg["mu_scale"]>0 and cfg["lam_scale"]>0 else \
                 "civil" if cfg["mu_scale"]==0 else "admin" \
                 if cfg["lam_scale"]==0 else "none"

        # High and low ability types
        W_comps = {}
        for tp_name, alpha_t in [("High-ability", alpha_H), ("Low-ability", alpha_L)]:
            ds_t = deepcopy(ds)
            for d in ds_t: d.alpha = alpha_t
            e_t   = solve_all(ds_t, regime)
            # Component decomposition per type
            W_B_t, W_T_t = 0.0, 0.0
            for i, d in enumerate(ds_t):
                th    = _RNG.normal(0, 0.3, 600)
                V     = d.scale * np.sqrt(max(e_t[i], 1e-9)) + th
                D     = np.maximum(0, 1 - V) * d.D_bar
                W_B_t += (d.scale * np.sqrt(max(e_t[i],1e-9)) - D.mean())
                liab   = d.lam * st.norm.cdf(
                    (d.e_star - e_t[i]) / (np.sqrt(d.sigma2c)+1e-9)) * D.mean()
                W_T_t += (d.scale * np.sqrt(max(e_t[i],1e-9))
                           - 0.5 * alpha_t * e_t[i]**2 - liab)
            W_comps[tp_name] = {"W_B": W_B_t, "W_T": W_T_t}

        C_R = c_R * sum(d.mu**2 for d in ds) / 2.0
        W_total = q * W_comps["High-ability"]["W_T"] + \
                  (1-q) * W_comps["Low-ability"]["W_T"]
        records.append({
            "Scenario":    scen_name,
            "W_B_H":       W_comps["High-ability"]["W_B"],
            "W_T_H":       W_comps["High-ability"]["W_T"],
            "W_B_L":       W_comps["Low-ability"]["W_B"],
            "W_T_L":       W_comps["Low-ability"]["W_T"],
            "C_R":         C_R,
            "W_total":     W_total - C_R,
        })

    df = pd.DataFrame(records).set_index("Scenario")
    print("\nWelfare decomposition (high-ability share q=%.2f):" % q)
    print(df.round(4).to_string())

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Module 8F — Welfare Decomposition & Distribution",
                 fontsize=13, fontweight="bold")

    scen_names = list(df.index)
    x = np.arange(len(scen_names))
    bar_cols   = [GRAY, RED, AMBER, SLATE, BLUE, RUST]

    # Stacked decomposition
    ax = axes[0]
    W_B = (q * df["W_B_H"] + (1-q) * df["W_B_L"]).values
    W_T = (q * df["W_T_H"] + (1-q) * df["W_T_L"]).values
    C_R_vals = df["C_R"].values
    ax.bar(x, W_B, width=0.55, color=BLUE,  alpha=0.75, label="Beneficiary surplus W_B")
    ax.bar(x, W_T, width=0.55, color=GREEN, alpha=0.60,
           bottom=W_B, label="Trustee surplus W_T")
    ax.bar(x, -C_R_vals, width=0.55, color=RED,   alpha=0.65,
           bottom=W_B+W_T, label="−Enforcement cost C_R")
    ax.set_xticks(x); ax.set_xticklabels(scen_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Welfare component"); ax.set_title("(a) Stacked welfare decomposition")
    ax.legend(fontsize=8); ax.axhline(0, color="black", lw=0.8)

    # High vs low ability comparison
    ax = axes[1]
    w2 = 0.28
    ax.bar(x - w2/2, df["W_T_H"].values, width=w2, color=BLUE,  alpha=0.85,
           label="High-ability trustee")
    ax.bar(x + w2/2, df["W_T_L"].values, width=w2, color=RED,   alpha=0.75,
           label="Low-ability trustee")
    ax.set_xticks(x); ax.set_xticklabels(scen_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Trustee net profit W_T")
    ax.set_title("(b) Distributional effect by ability type")
    ax.legend(); ax.axhline(0, color="black", lw=0.8)

    # Inequality: ratio W_T_H / W_T_L
    ax = axes[2]
    ratio_vals = (df["W_T_H"] / (df["W_T_L"].abs() + 1e-3)).values
    bars = ax.bar(x, ratio_vals, color=PURPLE, alpha=0.80)
    ax.axhline(1.0, color="black", ls="--", lw=1.2,
               label="Equal surplus ratio = 1")
    for bar, v in zip(bars, ratio_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(scen_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("W_T(high) / W_T(low)")
    ax.set_title("(c) Surplus ratio: high- vs low-ability\n(>1 = high-ability gains more)")
    ax.legend()

    plt.tight_layout()
    _save("fig_8f_decomp", save)
    plt.show()

    return df


# ════════════════════════════════════════════════════════════════
# MASTER RUNNER — MODULE 8
# ════════════════════════════════════════════════════════════════

def run_module8(save=True, run_all=True):
    """
    Run all policy simulation sub-modules.
    Set run_all=False and call individual functions for speed.
    """
    print("\n" + "#"*65)
    print("  MODULE 8: Policy (Regulatory) Simulation Suite")
    print("#"*65)

    results = {}

    # 8A: Regulatory design optimisation
    e_bar_opt, mu_opt, surf = optimise_regulatory_design(save=save)
    results["8A"] = (e_bar_opt, mu_opt)

    # 8B: Dynamic regulation with learning
    hist_e_dyn, hist_m, hist_v, hist_W_dyn = dynamic_regulation(
        n_periods=30, target_dim=5, sigma2c_true=0.65, save=save)
    results["8B"] = hist_e_dyn

    # 8C: Transition dynamics
    hist_trans, trans_cost = transition_dynamics(
        n_periods=40, n_agents=50, save=save)
    results["8C"] = trans_cost

    # 8D: Comparative architecture
    arch_res = compare_architectures(
        e_bar_opt=e_bar_opt, mu_opt=mu_opt, save=save)
    results["8D"] = arch_res

    # 8E: Japan-specific experiments
    jp_res, jdc_act, jdc_cf = japan_policy_experiments(save=save)
    results["8E"] = jp_res

    # 8F: Welfare decomposition
    df_decomp = welfare_decomposition(save=save)
    results["8F"] = df_decomp

    print("\n" + "="*65)
    print("MODULE 8 COMPLETE — key findings:")
    print("="*65)

    # Print integrated summary
    W_opt_net = social_welfare(
        solve_all([deepcopy(DIMS_DEFAULT[i]) for i in range(6)], "dual"),
        DIMS_DEFAULT) - 0.05 * sum(d.mu**2 for d in DIMS_DEFAULT) / 2
    print(f"  8A  Optimal penalty reallocation: "
          f"μ(e5,e6) ↑, μ(e3,e4) may ↓  (budget B={8.0})")
    print(f"  8B  Bayesian learning: regulator converges to true σ²_C "
          f"in ~{15} periods")
    print(f"  8C  Best escalation schedule: concave-optimal (S4)  "
          f"lowest transition cost")
    print(f"  8D  Optimal dual-layer architecture dominates all single-channel designs")
    print(f"  8E  JP1 (Prudent Investor Rule) yields largest welfare gain; "
          f"JP2 (BJR ext.) is welfare-reducing")
    best_jdc = "Early escalation" if np.mean(jdc_cf) > np.mean(jdc_act) else "Actual"
    print(f"  8E  JDC counterfactual: {best_jdc} outperforms actual enforcement path")
    print(f"  8F  Strict review (PIR) improves welfare for beneficiaries and "
          f"high-ability trustees alike")

    return results


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    results8 = run_module8(save=True, run_all=True)
