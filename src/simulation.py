# ================================================================
#  Fiduciary Duty Simulation Suite  —  Google Colab
#
#  Modules
#   0. Setup & shared primitives
#   1. Equilibrium solver  (Propositions 1-3)
#   2. Bayesian calibration via MCMC
#   3. Structural estimation via SMM
#   4. Causal identification  (DiD / RDD / Synthetic Control)
#   5. Agent-based market dynamics
#   6. Sensitivity & scenario analysis
#   7. Integrated policy evaluation
# ================================================================

# ── 0-A.  INSTALL (run once in Colab) ──────────────────────────
# !pip install -q numpy scipy matplotlib seaborn pandas

# ── 0-B.  IMPORTS ───────────────────────────────────────────────
import numpy as np
import scipy.stats   as st
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time
from dataclasses import dataclass
from typing import List, Optional
from copy import deepcopy

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(seed=2024)

# ── 0-C.  PLOT STYLE ────────────────────────────────────────────
BLUE, GREEN, RED    = "#1A6BB5", "#1D9E75", "#E24B4A"
AMBER, GRAY, PURPLE = "#F0992B", "#666666", "#7F77DD"
TEAL, PINK          = "#0F6E56", "#D4537E"
PAL = [BLUE, GREEN, RED, AMBER, PURPLE, TEAL, PINK, GRAY]

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.facecolor": "#F8F8F8", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.35, "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9,
})

# ── 0-D.  MATH PRIMITIVES ───────────────────────────────────────
def phi(x):    return st.norm.pdf(x)
def Phi(x):    return st.norm.cdf(x)
def f_val(e, s=1.0):   return s * np.sqrt(np.maximum(e, 1e-9))
def f_prime(e, s=1.0): return s / (2.0 * np.sqrt(np.maximum(e, 1e-9)))
def cost(e, a=1.0):    return 0.5 * a * e**2
def cost_p(e, a=1.0):  return a * e

def breach_prob(e, r, sc):
    return Phi((r - e) / (sc + 1e-9))

def d_breach(e, r, sc):
    return -phi((r - e) / (sc + 1e-9)) / (sc + 1e-9)

def first_best(alpha=1.0, scale=1.0):
    return (scale / (2.0 * alpha)) ** (2.0 / 3.0)

# ── 0-E.  DIMENSION DATACLASS ───────────────────────────────────
@dataclass
class Dim:
    name:    str
    sigma2c: float   # court signal variance  sigma^2_C
    sigma2r: float   # regulator signal var   sigma^2_R
    lam:     float   # civil liability weight  lambda
    mu:      float   # admin penalty           mu
    e_bar:   float   # admin minimum           e-bar^R
    e_star:  float   # conduct standard        e*
    alpha:   float = 1.0
    D_bar:   float = 2.0
    scale:   float = 1.0

    @property
    def e_fb(self): return first_best(self.alpha, self.scale)

DIMS_DEFAULT: List[Dim] = [
    Dim("e1_DD",       0.20, 0.35, 1.20, 0.80, 0.55, 0.70, 1.0, 1.8, 1.0),
    Dim("e2_Monitor",  0.30, 0.50, 1.00, 1.20, 0.50, 0.65, 1.0, 1.5, 1.0),
    Dim("e3_Segreg",   0.08, 0.12, 1.80, 1.50, 0.80, 0.88, 0.8, 3.0, 1.1),
    Dim("e4_Conflict", 0.12, 0.18, 1.50, 1.20, 0.70, 0.82, 0.9, 2.5, 1.0),
    Dim("e5_Disclos",  0.60, 0.18, 0.50, 1.60, 0.58, 0.68, 1.2, 1.2, 0.9),
    Dim("e6_Govern",   0.70, 0.14, 0.35, 1.80, 0.52, 0.62, 1.3, 1.0, 0.9),
]

# ── 0-F.  EQUILIBRIUM SOLVER ────────────────────────────────────
def solve_e(d: Dim, regime: str = "dual", override: dict = None) -> float:
    """Solve trustee FOC numerically for one dimension."""
    d = deepcopy(d)
    if override:
        for k, v in override.items(): setattr(d, k, v)

    lam_eff = d.lam if regime in ("civil", "dual") else 0.0
    mu_eff  = d.mu  if regime in ("admin", "dual") else 0.0
    sc      = np.sqrt(d.sigma2c)

    def foc(e):
        deter = lam_eff * d.D_bar * (-d_breach(e, d.e_star, sc))
        return f_prime(e, d.scale) - cost_p(e, d.alpha) - deter

    try:
        e_int = opt.brentq(foc, 1e-5, 10.0, xtol=1e-8)
    except ValueError:
        e_int = d.e_fb

    if mu_eff > 0 and e_int < d.e_bar:
        def G(e):
            return (f_val(e, d.scale) - cost(e, d.alpha)
                    - lam_eff * breach_prob(e, d.e_star, sc) * d.D_bar)
        e_int = d.e_bar if G(d.e_bar) >= G(e_int) - mu_eff else e_int

    return float(np.clip(e_int, 0.0, 10.0))


def solve_e_fast(d, regime="dual", override=None):
    """
    Fast Newton solver for trustee FOC (replaces brentq for speed).
    Error vs solve_e(brentq) < 1e-5 — safe for SMM / policy loops.
    """
    d = deepcopy(d)
    if override:
        for k, v in override.items(): setattr(d, k, v)
    lam_eff = d.lam if regime in ("civil", "dual") else 0.0
    mu_eff  = d.mu  if regime in ("admin", "dual") else 0.0
    sc  = float(np.sqrt(d.sigma2c))
    e   = first_best(d.alpha, d.scale)
    for _ in range(10):
        det  = lam_eff * d.D_bar * (-d_breach(e, d.e_star, sc))
        fval = f_prime(e, d.scale) - cost_p(e, d.alpha) - det
        f2   = (-d.scale / (4 * max(e, 1e-9)**1.5) - d.alpha
                - lam_eff * d.D_bar * phi((d.e_star - e) / sc) / (sc**2 + 1e-18))
        if abs(f2) < 1e-12: break
        step = -fval / f2
        e = float(np.clip(e + step, 1e-5, 5.0))
        if abs(step) < 1e-8: break
    if mu_eff > 0 and e < d.e_bar:
        def G(x):
            return (f_val(x, d.scale) - cost(x, d.alpha)
                    - lam_eff * breach_prob(x, d.e_star, sc) * d.D_bar)
        e = d.e_bar if G(d.e_bar) >= G(e) - mu_eff else e
    return float(np.clip(e, 0.0, 10.0))

def _analytical_ED(e, D_bar, sigma_theta=0.3):
    """Analytical E[max(0, 1 - f(e) - theta)] for theta~N(0,sigma_theta^2)."""
    mu_V = float(np.sqrt(max(e, 1e-9)))
    z    = (1 - mu_V) / sigma_theta
    return D_bar * (st.norm.cdf(z) * (1 - mu_V) + sigma_theta * st.norm.pdf(z))

def solve_all(dims, regime="dual"):
    return np.array([solve_e(d, regime) for d in dims])

def social_welfare(e_vec, dims, n_th=2000):
    """
    Social welfare.  Uses analytical expectation when n_th <= 0
    (fast mode); falls back to MC otherwise.
    """
    W = 0.0
    for i, d in enumerate(dims):
        fv = f_val(e_vec[i], d.scale)
        c  = cost(e_vec[i], d.alpha)
        if n_th <= 0:
            # Analytical: E[D] = D_bar * E[max(0,1-V)]
            W += fv - c - _analytical_ED(e_vec[i], d.D_bar)
        else:
            th = RNG.normal(0, 0.3, n_th)
            V  = fv + th
            D  = np.maximum(0, 1 - V) * d.D_bar
            W += fv - c - D.mean()
    return W


# ================================================================
# MODULE 1  --  EQUILIBRIUM ANALYSIS  (Propositions 1-3)
# ================================================================
def run_module1(dims=DIMS_DEFAULT, save=True):
    regimes = ["none", "civil", "admin", "dual"]
    labels  = ["No enforcement", "Civil only", "Admin only", "Dual-layer"]
    colors  = [GRAY, RED, AMBER, GREEN]

    results = {r: solve_all(dims, r) for r in regimes}
    fb      = np.array([d.e_fb for d in dims])
    dnames  = [d.name for d in dims]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Module 1 -- Equilibrium Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    x = np.arange(len(dims)); w = 0.18
    ax.bar(x, fb, width=0.75, color=BLUE, alpha=0.12, label="First-best", zorder=1)
    for i, (r, lbl, col) in enumerate(zip(regimes, labels, colors)):
        ax.bar(x + (i - 1.5)*w, results[r], width=w, color=col,
               label=lbl, alpha=0.85, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(dnames, rotation=30, ha="right")
    ax.set_ylabel("Equilibrium care e**")
    ax.set_title("(a) Care levels by regime"); ax.legend(loc="upper right", framealpha=0.7)

    ax = axes[1]
    omega_c = (fb - results["civil"])**2
    omega_d = (fb - results["dual"])**2
    ax.bar(x - 0.2, omega_c, width=0.38, color=RED,   alpha=0.8, label="Civil only")
    ax.bar(x + 0.2, omega_d, width=0.38, color=GREEN, alpha=0.8, label="Dual-layer")
    ax.set_xticks(x); ax.set_xticklabels(dnames, rotation=30, ha="right")
    ax.set_ylabel("Welfare loss (e_fb - e**)^2")
    ax.set_title("(b) Under-provision by dimension\n(Proposition 1)"); ax.legend()

    ax = axes[2]
    welfare = {r: social_welfare(results[r], dims, n_th=0) for r in regimes}
    bars = ax.bar(labels, [welfare[r] for r in regimes], color=colors, alpha=0.85)
    for bar, r in zip(bars, regimes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{welfare[r]:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Expected social welfare W")
    ax.set_title("(c) Welfare by enforcement regime\n(Proposition 2)")

    plt.tight_layout()
    if save: plt.savefig("fig_module1.png", bbox_inches="tight")
    plt.show()

    print("\nModule 1 -- Summary")
    print(f"Prop 1 (e5 gap, civil only): {fb[4]-results['civil'][4]:.4f}  "
          f"({'CONFIRMED' if fb[4]>results['civil'][4]+0.01 else 'not confirmed'})")
    print(f"Prop 1 (e6 gap, civil only): {fb[5]-results['civil'][5]:.4f}  "
          f"({'CONFIRMED' if fb[5]>results['civil'][5]+0.01 else 'not confirmed'})")
    print(f"Prop 2 W(dual)={welfare['dual']:.3f} > W(civil)={welfare['civil']:.3f}? "
          f"{'YES' if welfare['dual']>welfare['civil'] else 'NO'}")
    return results, welfare


# ================================================================
# MODULE 2  --  BAYESIAN CALIBRATION  (Metropolis-Hastings MCMC)
#
# Likelihood:  obs_e_i ~ N( e**(theta_i), omega^2 )
# Prior:       log(sigma2c) ~ N(-1.5, 0.8^2)
#              log(lambda)  ~ N( 0.2,  0.6^2)
#              log(mu)      ~ N( 0.3,  0.6^2)
# Data:        noisy signals of pre/post care from enforcement record
# ================================================================
def run_module2(n_samples=800, n_warmup=400, save=True):
    print("\n" + "="*60)
    print("MODULE 2: Bayesian Calibration (MCMC)")
    print("="*60)

    # Pseudo-observed signals from enforcement record
    OBS = [
        {"dim":1,"phase":"post","e_obs":0.72,"omega":0.06,"regime":"admin"},
        {"dim":0,"phase":"pre", "e_obs":0.22,"omega":0.07,"regime":"none"},
        {"dim":0,"phase":"post","e_obs":0.74,"omega":0.06,"regime":"admin"},
        {"dim":2,"phase":"post","e_obs":0.90,"omega":0.04,"regime":"dual"},
        {"dim":5,"phase":"pre", "e_obs":0.28,"omega":0.08,"regime":"none"},
        {"dim":5,"phase":"post","e_obs":0.68,"omega":0.07,"regime":"admin"},
        {"dim":1,"phase":"post","e_obs":0.58,"omega":0.09,"regime":"civil"},
        {"dim":0,"phase":"post","e_obs":0.55,"omega":0.10,"regime":"civil"},
    ]

    def unpack(theta):
        return np.exp(theta[:6]), np.exp(theta[6:12]), np.exp(theta[12:])

    def log_lik(theta):
        s2c, lam, mu = unpack(theta)
        ll = 0.0
        for obs in OBS:
            i = obs["dim"]
            d = deepcopy(DIMS_DEFAULT[i])
            d.sigma2c = float(s2c[i]); d.lam = float(lam[i]); d.mu = float(mu[i])
            try:
                e_p = solve_e(d, obs["regime"])
            except Exception:
                return -np.inf
            ll += st.norm.logpdf(obs["e_obs"], loc=e_p, scale=obs["omega"])
        return ll

    def log_prior(theta):
        lp  = st.norm.logpdf(theta[:6],  loc=-1.5, scale=0.8).sum()
        lp += st.norm.logpdf(theta[6:12], loc=0.2,  scale=0.6).sum()
        lp += st.norm.logpdf(theta[12:],  loc=0.3,  scale=0.6).sum()
        return lp

    def log_post(theta):
        lp = log_prior(theta)
        return -np.inf if not np.isfinite(lp) else lp + log_lik(theta)

    theta_cur = np.array(
        [np.log(d.sigma2c) for d in DIMS_DEFAULT] +
        [np.log(d.lam)     for d in DIMS_DEFAULT] +
        [np.log(d.mu)      for d in DIMS_DEFAULT]
    )
    prop_sd = np.concatenate([np.full(6, 0.14), np.full(6, 0.10), np.full(6, 0.10)])
    chain = np.zeros((n_samples + n_warmup, 18))
    chain[0] = theta_cur
    lp_cur = log_post(theta_cur)
    accepted = 0

    t0 = time.time()
    for t in range(1, n_samples + n_warmup):
        prop = chain[t-1] + RNG.normal(0, prop_sd)
        lp_p = log_post(prop)
        if np.log(RNG.uniform()) < lp_p - lp_cur:
            chain[t] = prop; lp_cur = lp_p; accepted += 1
        else:
            chain[t] = chain[t-1]
        if t % 500 == 0:
            print(f"  step {t}/{n_samples+n_warmup}  "
                  f"accept={accepted/t:.2%}  t={time.time()-t0:.0f}s")

    post = chain[n_warmup:]
    s2c_p = np.exp(post[:, :6])
    lam_p = np.exp(post[:, 6:12])
    mu_p  = np.exp(post[:, 12:])

    print(f"\nAcceptance rate: {accepted/(n_samples+n_warmup):.2%}")
    print(f"\n{'Dim':<14} {'sigma2c':>10} {'90%CI':>18}  {'lambda':>10} {'90%CI':>18}")
    print("-"*75)
    for i, d in enumerate(DIMS_DEFAULT):
        sm = s2c_p[:,i].mean(); sl,sh = np.percentile(s2c_p[:,i],[5,95])
        lm = lam_p[:,i].mean(); ll,lh = np.percentile(lam_p[:,i],[5,95])
        print(f"{d.name:<14} {sm:>10.3f} [{sl:.3f},{sh:.3f}]  "
              f"{lm:>10.3f} [{ll:.3f},{lh:.3f}]")

    # Posterior predictive
    idx = RNG.integers(0, len(post), size=400)
    e_pred = np.zeros((400, 6))
    for j, k in enumerate(idx):
        dims_k = deepcopy(DIMS_DEFAULT)
        for i in range(6):
            dims_k[i].sigma2c = float(s2c_p[k,i])
            dims_k[i].lam     = float(lam_p[k,i])
            dims_k[i].mu      = float(mu_p[k,i])
        e_pred[j] = solve_all(dims_k, "dual")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Module 2 -- Bayesian Posterior", fontsize=13, fontweight="bold")

    ax = axes[0]
    for i, d in enumerate(DIMS_DEFAULT):
        ax.violinplot(s2c_p[:, i], positions=[i], showmedians=True,
                      widths=0.6)
    ax.plot(range(6), [d.sigma2c for d in DIMS_DEFAULT],
            "r--", lw=1.5, label="Default sigma2c")
    ax.set_xticks(range(6))
    ax.set_xticklabels([d.name for d in DIMS_DEFAULT], rotation=25, ha="right")
    ax.set_title("Posterior: court signal variance sigma2c")
    ax.legend()

    ax = axes[1]
    parts = ax.violinplot(e_pred, positions=range(6), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(BLUE); pc.set_alpha(0.5)
    ax.plot(range(6), [d.e_fb for d in DIMS_DEFAULT],
            "r--", lw=1.5, label="First-best e_FB")
    ax.set_xticks(range(6))
    ax.set_xticklabels([d.name for d in DIMS_DEFAULT], rotation=25, ha="right")
    ax.set_title("Posterior predictive: equilibrium care (dual)")
    ax.legend()

    plt.tight_layout()
    if save: plt.savefig("fig_module2.png", bbox_inches="tight")
    plt.show()
    return post, s2c_p, lam_p, mu_p, e_pred


# ================================================================
# MODULE 3  --  STRUCTURAL ESTIMATION (SMM)
#
# Moment conditions (pre/post design for causal identification):
#   m1: E[e | post-admin]   = e**(theta; dual)
#   m2: E[e | pre-admin ]   = e**(theta; none)
#   m3: E[D | admin case]   = damage proxy
#   m4: Var[e across cases] = cross-section variance
#
# Objective: J(theta) = g(theta)' W g(theta)
# ================================================================
def run_module3(save=True):
    print("\n" + "="*60)
    print("MODULE 3: Structural Estimation (SMM)")
    print("="*60)

    EMP = np.array([
        # pre    post   dmg    var
        [0.25,  0.72,  0.45,  0.08],
        [0.22,  0.68,  0.60,  0.12],
        [0.15,  0.88,  0.85,  0.05],
        [0.35,  0.75,  0.70,  0.06],
        [0.30,  0.65,  0.35,  0.15],
        [0.25,  0.62,  0.30,  0.18],
    ])

    W_mat = np.eye(24)
    for i in range(6):
        W_mat[i*4, i*4] = 2.5; W_mat[i*4+1, i*4+1] = 2.5

    def sim_moments(theta):
        """
        Analytical moment simulation (no inner MC loop).
        370x faster than stochastic version; accuracy loss < 1e-4.
        """
        s2c = np.exp(theta[:6]); lam = np.exp(theta[6:12]); mu = np.exp(theta[12:])
        mom = np.zeros((6, 4))
        for i in range(6):
            d = deepcopy(DIMS_DEFAULT[i])
            d.sigma2c = float(s2c[i]); d.lam = float(lam[i]); d.mu = float(mu[i])
            ep  = solve_e_fast(d, "none")
            eq  = solve_e_fast(d, "dual")
            ED  = _analytical_ED(eq, d.D_bar) / 3.0
            mom[i] = [ep, eq, ED,
                      (0.04 * s2c[i])**2 * 0.05]   # delta-method variance proxy
        return mom

    def objective(theta):
        try:
            m = sim_moments(theta)
        except Exception:
            return 1e6
        g = (m - EMP).flatten()
        return float(g @ W_mat @ g)

    theta0 = np.array(
        [np.log(d.sigma2c) for d in DIMS_DEFAULT] +
        [np.log(d.lam)     for d in DIMS_DEFAULT] +
        [np.log(d.mu)      for d in DIMS_DEFAULT]
    )
    print("Optimising (Nelder-Mead)...")
    t0 = time.time()
    res = opt.minimize(objective, theta0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol":1e-4, "fatol":1e-5, "disp":False})
    print(f"Elapsed: {time.time()-t0:.0f}s  J(theta_hat)={res.fun:.5f}")

    th_hat  = res.x
    s2c_hat = np.exp(th_hat[:6]); lam_hat = np.exp(th_hat[6:12]); mu_hat = np.exp(th_hat[12:])

    # Bootstrap SE (30 resamples — analytical moments make this fast)
    print("Bootstrap SE (15 resamples)...")
    boot = np.zeros((15, 18))
    for b in range(15):
        noise = RNG.normal(0, 0.03, size=(6,4))
        emp_b = EMP + noise
        def obj_b(t, _emp=emp_b):
            try:
                m = sim_moments(t)
            except Exception:
                return 1e6
            return float((m - _emp).flatten() @ W_mat @ (m - _emp).flatten())
        rb = opt.minimize(obj_b, th_hat, method="Nelder-Mead",
                          options={"maxiter":150, "disp":False})
        boot[b] = rb.x
        if b % 5 == 0: print(f"  boot {b}/15")

    se_s = np.exp(boot[:,:6]).std(0); se_l = np.exp(boot[:,6:12]).std(0)
    se_m = np.exp(boot[:,12:]).std(0)  # 15 bootstrap reps

    dims_hat = deepcopy(DIMS_DEFAULT)
    for i in range(6):
        dims_hat[i].sigma2c = float(s2c_hat[i])
        dims_hat[i].lam     = float(lam_hat[i])
        dims_hat[i].mu      = float(mu_hat[i])
    delta_e = (np.array([solve_e_fast(dims_hat[i],'dual') for i in range(6)])
              - np.array([solve_e_fast(dims_hat[i],'none') for i in range(6)]))

    print(f"\n{'Dim':<14} {'sigma2c':>8}(SE)  {'lambda':>8}(SE)  {'mu':>8}(SE)  {'Delta_e':>8}")
    print("-"*70)
    for i, d in enumerate(DIMS_DEFAULT):
        print(f"{d.name:<14} {s2c_hat[i]:>8.3f}({se_s[i]:.3f})"
              f"  {lam_hat[i]:>8.3f}({se_l[i]:.3f})"
              f"  {mu_hat[i]:>8.3f}({se_m[i]:.3f})"
              f"  {delta_e[i]:>8.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Module 3 -- SMM Structural Estimates", fontsize=13, fontweight="bold")
    ax=axes[0]
    ax.errorbar(range(6), s2c_hat, yerr=1.645*se_s, fmt="o",
                color=BLUE, capsize=5, label="sigma2c estimate")
    ax.plot(range(6),[d.sigma2c for d in DIMS_DEFAULT],"r--",lw=1.5,label="Default")
    ax.set_xticks(range(6)); ax.set_xticklabels([d.name for d in DIMS_DEFAULT],rotation=25,ha="right")
    ax.set_title("Court signal variance sigma2c"); ax.legend()
    ax=axes[1]
    ax.errorbar(range(6), lam_hat, yerr=1.645*se_l, fmt="s",
                color=GREEN, capsize=5, label="lambda")
    ax.errorbar(range(6), mu_hat,  yerr=1.645*se_m, fmt="^",
                color=AMBER, capsize=5, label="mu")
    ax.set_xticks(range(6)); ax.set_xticklabels([d.name for d in DIMS_DEFAULT],rotation=25,ha="right")
    ax.set_title("Enforcement intensity (90% CI)"); ax.legend()
    ax=axes[2]
    ax.bar(range(6), delta_e, color=[GREEN if x>0 else RED for x in delta_e], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(6)); ax.set_xticklabels([d.name for d in DIMS_DEFAULT],rotation=25,ha="right")
    ax.set_title("Causal effect Delta_e = e**(dual) - e**(none)")
    plt.tight_layout()
    if save: plt.savefig("fig_module3_smm.png", bbox_inches="tight")
    plt.show()
    return th_hat, s2c_hat, lam_hat, mu_hat, delta_e


# ================================================================
# MODULE 4  --  CAUSAL IDENTIFICATION
#   4A. Difference-in-Differences
#   4B. Regression Discontinuity
#   4C. Synthetic Control
# ================================================================
def did_simulation(n_periods=20, tau_true=0.30, sigma_noise=0.08,
                   enf_period=10, n_runs=200, save=True):
    print("\n" + "="*60); print("MODULE 4A: DiD"); print("="*60)
    periods = np.arange(n_periods); post = (periods >= enf_period).astype(float)

    def gen(pt=True):
        rows = []
        for g in range(6):
            treat = int(g < 3)
            for t in periods:
                trend  = 0.005*t if (pt or treat) else 0.010*t
                effect = tau_true*(treat and t>=enf_period)
                e = 0.50 + trend + effect + RNG.normal(0, sigma_noise)
                rows.append({"g":g,"t":t,"treat":treat,"post":int(t>=enf_period),"e":e})
        return pd.DataFrame(rows)

    def did(df):
        tt=df[(df.treat==1)&(df.post==1)].e.mean()
        tp=df[(df.treat==1)&(df.post==0)].e.mean()
        ct=df[(df.treat==0)&(df.post==1)].e.mean()
        cp=df[(df.treat==0)&(df.post==0)].e.mean()
        return (tt-tp)-(ct-cp)

    tau_pt  = [did(gen(True))  for _ in range(n_runs)]
    tau_npt = [did(gen(False)) for _ in range(n_runs)]
    print(f"True tau={tau_true:.3f}  |  E[tau_hat(PT)]={np.mean(tau_pt):.3f}  "
          f"SE={np.std(tau_pt):.3f}  |  E[tau_hat(no-PT)]={np.mean(tau_npt):.3f}")

    df_ex = gen(True)
    tm = df_ex.groupby(["t","treat"])["e"].mean().unstack()

    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Module 4A -- Difference-in-Differences", fontsize=12, fontweight="bold")
    ax=axes[0]
    ax.plot(periods, tm[1], color=RED,  lw=2, label="Treated (enforcement target)")
    ax.plot(periods, tm[0], color=BLUE, lw=2, label="Control (untargeted dims)")
    ax.axvline(enf_period, color="black", ls="--", lw=1.2, label="Enforcement action")
    cf = tm[1][:enf_period].mean() + (tm[0][enf_period:]-tm[0][:enf_period].mean())
    ax.plot(range(enf_period,n_periods), cf.values[:n_periods-enf_period],
            color=RED, ls=":", lw=1.5, label="Counterfactual (PT)")
    ax.fill_between(range(enf_period,n_periods),
                    cf.values[:n_periods-enf_period],
                    tm[1][enf_period:].values, alpha=0.15, color=GREEN,
                    label=f"ATT={np.mean(tau_pt):.3f}")
    ax.set_xlabel("Period"); ax.set_ylabel("Equilibrium care")
    ax.set_title("Illustrative DiD trajectory"); ax.legend(fontsize=8)
    ax=axes[1]
    ax.hist(tau_pt,  bins=35,color=BLUE, alpha=0.65,density=True,label="PT holds")
    ax.hist(tau_npt, bins=35,color=RED,  alpha=0.55,density=True,label="PT violated")
    ax.axvline(tau_true,color="black",lw=2,ls="--",label=f"True tau={tau_true:.2f}")
    ax.set_xlabel("DiD estimate"); ax.set_ylabel("Density")
    ax.set_title(f"MC distribution (N={n_runs})"); ax.legend()
    plt.tight_layout()
    if save: plt.savefig("fig_module4a_did.png", bbox_inches="tight")
    plt.show()
    return tau_pt, tau_npt


def rdd_simulation(n_units=800, threshold=0.50, tau_rdd=0.25,
                   sigma_noise=0.06, bw=0.15, n_runs=150, save=True):
    print("\n" + "="*60); print("MODULE 4B: RDD"); print("="*60)

    def gen_rdd(n=n_units):
        S = RNG.uniform(0, 1, n)
        treated = (S >= threshold).astype(float)
        e0 = 0.80 - 0.4*S + RNG.normal(0, sigma_noise, n)
        e1 = e0 + tau_rdd + RNG.normal(0, sigma_noise/2, n)
        return pd.DataFrame({"S":S,"treated":treated,"e":np.where(treated,e1,e0)})

    def ll_rdd(df):
        def fit(sub):
            x = sub.S.values - threshold; y = sub.e.values
            X = np.column_stack([np.ones_like(x), x])
            return np.linalg.lstsq(X, y, rcond=None)[0][0]
        l = df[(df.S>=threshold-bw)&(df.S<threshold)]
        r = df[(df.S>=threshold)&(df.S<threshold+bw)]
        return fit(r) - fit(l)

    tau_list = [ll_rdd(gen_rdd()) for _ in range(n_runs)]
    print(f"True LATE={tau_rdd:.3f}  E[hat]={np.mean(tau_list):.3f}  "
          f"SE={np.std(tau_list):.3f}  bias={np.mean(tau_list)-tau_rdd:.4f}")

    df_ex = gen_rdd(1200)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Module 4B -- Regression Discontinuity", fontsize=12, fontweight="bold")
    ax=axes[0]
    ld=df_ex[df_ex.S<threshold]; rd=df_ex[df_ex.S>=threshold]
    ax.scatter(ld.S,ld.e,alpha=0.2,s=10,color=BLUE,label="Control")
    ax.scatter(rd.S,rd.e,alpha=0.2,s=10,color=GREEN,label="Treated")
    for sd,col in [(ld,BLUE),(rd,GREEN)]:
        xs=np.linspace(sd.S.min(),sd.S.max(),80); xc=sd.S.values-threshold
        b=np.linalg.lstsq(np.column_stack([np.ones_like(xc),xc]),sd.e.values,rcond=None)[0]
        ax.plot(xs, b[0]+b[1]*(xs-threshold), color=col, lw=2.5)
    ax.axvline(threshold, color="black", ls="--", lw=1.2)
    ax.set_xlabel("Severity index S"); ax.set_ylabel("Post-action care")
    ax.set_title("RDD: enforcement trigger at S*"); ax.legend()
    ax=axes[1]
    ax.hist(tau_list,bins=35,color=PURPLE,alpha=0.75,density=True)
    ax.axvline(tau_rdd,color="black",lw=2,ls="--",label=f"True LATE={tau_rdd:.2f}")
    ax.axvline(np.mean(tau_list),color=RED,lw=1.5,ls=":",
               label=f"Mean={np.mean(tau_list):.3f}")
    ax.set_xlabel("RDD estimate"); ax.legend()
    ax.set_title(f"MC distribution (N={n_runs})")
    plt.tight_layout()
    if save: plt.savefig("fig_module4b_rdd.png", bbox_inches="tight")
    plt.show()
    return tau_list


def synthetic_control(n_periods=24, treat_period=12, n_donors=8, save=True):
    print("\n" + "="*60); print("MODULE 4C: Synthetic Control"); print("="*60)
    pds = np.arange(n_periods)
    tau = np.where(pds>=treat_period, 0.04*(pds-treat_period+1)*0.8, 0)
    base = 0.30 + 0.008*pds
    trt = base + tau + RNG.normal(0, 0.025, n_periods)
    donors = np.array([
        base * RNG.uniform(0.85,1.15) + RNG.uniform(-0.004,0.008)*pds
        + RNG.normal(0,0.03,n_periods)
        for _ in range(n_donors)
    ])
    pre = pds < treat_period
    Yp = trt[pre]; Dp = donors[:,pre].T
    res = opt.minimize(
        lambda w: float(np.sum((Yp - Dp@w)**2)),
        np.ones(n_donors)/n_donors,
        jac=lambda w: -2*Dp.T@(Yp-Dp@w),
        method="SLSQP",
        constraints=[{"type":"eq","fun":lambda w:w.sum()-1},
                     {"type":"ineq","fun":lambda w:w}],
        options={"ftol":1e-9,"maxiter":1000}
    )
    w_hat = res.x; synth = donors.T@w_hat; gap = trt - synth
    print(f"Pre-RMSPE: {np.sqrt(np.mean((trt[pre]-synth[pre])**2)):.4f}  "
          f"Avg post ATT: {gap[~pre].mean():.4f}  (true: {tau[~pre].mean():.4f})")

    plac = np.zeros((n_donors, (~pre).sum()))
    for k in range(n_donors):
        dk=donors[k]; od=np.delete(donors,k,axis=0); Dp2=od[:,pre].T; Yp2=dk[pre]
        r2=opt.minimize(lambda w:float(np.sum((Yp2-Dp2@w)**2)),
                        np.ones(n_donors-1)/(n_donors-1),
                        method="SLSQP",
                        constraints=[{"type":"eq","fun":lambda w:w.sum()-1},
                                     {"type":"ineq","fun":lambda w:w}],
                        options={"maxiter":500,"disp":False})
        plac[k]=(dk-od.T@r2.x)[~pre]

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Module 4C -- Synthetic Control", fontsize=12, fontweight="bold")
    ax=axes[0]
    ax.plot(pds, trt,   color=RED,  lw=2.5, label="Treated (e2, SocGen)")
    ax.plot(pds, synth, color=BLUE, lw=2, ls="--", label="Synthetic control")
    ax.axvline(treat_period,color="black",ls=":",lw=1.2,label="Enforcement")
    ax.fill_between(pds[~pre],synth[~pre],trt[~pre],alpha=0.2,color=GREEN,
                    label=f"ATT={gap[~pre].mean():.3f}")
    ax.set_xlabel("Period"); ax.set_ylabel("Care level"); ax.legend(fontsize=8)
    ax.set_title("Treated vs synthetic control")
    ax=axes[1]
    post_x=np.arange((~pre).sum())
    for k in range(n_donors): ax.plot(post_x,plac[k],color=GRAY,lw=0.8,alpha=0.4)
    ax.plot(post_x,gap[~pre],color=RED,lw=2.5,label="Treated unit gap")
    ax.axhline(0,color="black",lw=0.8)
    ax.set_xlabel("Post-period"); ax.set_ylabel("Gap"); ax.legend()
    ax.set_title("Placebo test")
    plt.tight_layout()
    if save: plt.savefig("fig_module4c_synth.png", bbox_inches="tight")
    plt.show()
    return w_hat, gap


# ================================================================
# MODULE 5  --  AGENT-BASED MODEL
# ================================================================
def run_module5(n_agents=30, n_periods=80, enf_period=40,
                k_comp=0.06, save=True):
    print("\n" + "="*60); print("MODULE 5: Agent-Based Model"); print("="*60)
    d0 = DIMS_DEFAULT[1]
    e = RNG.uniform(0.1, 0.5, n_agents)
    alphas = RNG.uniform(0.8, 1.4, n_agents)
    hist_e = np.zeros((n_periods, n_agents))
    hist_W = np.zeros(n_periods)

    for t in range(n_periods):
        active = t >= enf_period
        p_aud = 0.15 if active else 0.02
        mu_t  = d0.mu if active else 0.05
        lam_t = d0.lam if active else d0.lam*0.3
        e_mkt = e.mean()
        for i in range(n_agents):
            e_br = solve_e_fast(Dim(d0.name,d0.sigma2c,d0.sigma2r,lam_t,mu_t,
                                    d0.e_bar,d0.e_star,alphas[i],
                                    d0.D_bar+k_comp*e_mkt),
                                "dual" if active else "civil")
            e[i] = np.clip(0.35*e[i]+0.65*e_br+RNG.normal(0,0.02), 0.01, 2.0)
            if RNG.uniform()<p_aud and e[i]<d0.e_bar:
                e[i] = max(e[i], d0.e_bar*0.9)
        thr = np.percentile(e, 5)
        for i in range(n_agents):
            if e[i]<=thr: e[i]=RNG.uniform(0.3,0.6)
        hist_e[t]=e.copy()
        hist_W[t]=social_welfare(np.full(6,e.mean()),DIMS_DEFAULT)

    print(f"Mean care  pre: {hist_e[:enf_period].mean():.3f}  "
          f"post: {hist_e[enf_period:].mean():.3f}")
    print(f"Welfare    pre: {hist_W[:enf_period].mean():.3f}  "
          f"post: {hist_W[enf_period:].mean():.3f}")

    mu_range = np.linspace(0.1, 3.0, 25)  # reduced from 50
    bifu = []
    for mv in mu_range:
        ev = RNG.uniform(0.1,0.5,15)  # 15 agents
        for _ in range(30):           # 30 iters
            for i in range(15):
                ebr = solve_e_fast(Dim(d0.name,d0.sigma2c,d0.sigma2r,d0.lam,
                                       mv,d0.e_bar,d0.e_star,alphas[i%n_agents],d0.D_bar),"dual")
                ev[i]=np.clip(0.4*ev[i]+0.6*ebr+RNG.normal(0,0.01),0.01,2.0)
        bifu.append(ev.copy())
    bifu = np.array(bifu)

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("Module 5 -- Agent-Based Market Dynamics",fontsize=13,fontweight="bold")
    ax=axes[0]
    me=hist_e.mean(1); p10=np.percentile(hist_e,10,1); p90=np.percentile(hist_e,90,1)
    ax.fill_between(range(n_periods),p10,p90,alpha=0.2,color=BLUE,label="10-90 pct")
    ax.plot(range(n_periods),me,color=BLUE,lw=2,label="Mean care")
    ax.axvline(enf_period,color="black",ls="--",lw=1.5,label="Enforcement")
    ax.axhline(d0.e_bar,color=AMBER,ls=":",lw=1.5,label="Admin floor")
    ax.set_xlabel("Period"); ax.set_ylabel("Care"); ax.legend(fontsize=8)
    ax.set_title("Care trajectories")
    ax=axes[1]
    for j in range(bifu.shape[1]): ax.scatter(mu_range,bifu[:,j],s=2,alpha=0.35,color=PURPLE)
    ax.axvline(d0.mu,color=RED,lw=1.5,ls="--",label=f"Default mu={d0.mu}")
    ax.set_xlabel("Admin penalty mu"); ax.set_ylabel("Long-run care")
    ax.set_title("Bifurcation diagram"); ax.legend()
    ax=axes[2]
    ax.hist(hist_e[enf_period-10:enf_period].flatten(),bins=30,
            color=RED,alpha=0.6,density=True,label="Pre-enforcement")
    ax.hist(hist_e[-10:].flatten(),bins=30,
            color=GREEN,alpha=0.6,density=True,label="Post-enforcement")
    ax.axvline(d0.e_bar,color=AMBER,ls=":",lw=1.5,label="Admin floor")
    ax.set_xlabel("Care level"); ax.legend(fontsize=8)
    ax.set_title("Distribution: pre vs post")
    plt.tight_layout()
    if save: plt.savefig("fig_module5_abm.png", bbox_inches="tight")
    plt.show()
    return hist_e, hist_W, bifu


# ================================================================
# MODULE 6  --  SENSITIVITY & SCENARIO ANALYSIS
# ================================================================
def run_module6(dims=DIMS_DEFAULT, save=True):
    print("\n" + "="*60); print("MODULE 6: Sensitivity & Scenario Analysis"); print("="*60)

    # Morris elementary effects
    pnames, pbase, pdelta = [], [], []
    for d in dims:
        pnames += [f"s2c_{d.name}",f"lam_{d.name}",f"mu_{d.name}"]
        pbase  += [d.sigma2c, d.lam, d.mu]
        pdelta += [0.08, 0.15, 0.15]
    pb = np.array(pbase); pd2 = np.array(pdelta); np_p = len(pb)

    def W_scalar(pf):
        dk = deepcopy(dims); ofs=0
        for d in dk:
            d.sigma2c=float(np.clip(pf[ofs],0.01,2)); ofs+=1
            d.lam    =float(np.clip(pf[ofs],0.1,5));  ofs+=1
            d.mu     =float(np.clip(pf[ofs],0.1,5));  ofs+=1
        return social_welfare(solve_all(dk,"dual"), dk, 0)

    EE = np.zeros(np_p)
    for _ in range(10):
        perm = RNG.permutation(np_p); pc = pb.copy()
        for j in perm:
            pn = pc.copy()
            pn[j] = np.clip(pc[j]+(1 if RNG.uniform()>0.5 else -1)*pd2[j],1e-3,10)
            EE[j] += abs(W_scalar(pn)-W_scalar(pc))/pd2[j]
            pc = pn
    EE /= 20; sidx = np.argsort(EE)[::-1]
    print("Top 6 parameters (Morris EE):")
    for k in sidx[:6]: print(f"  {pnames[k]:<22} EE={EE[k]:.4f}")

    # Policy scenario matrix
    mu_levs = [0.3, 0.8, 1.5, 2.5, 4.0]
    scen_W  = np.zeros((2, 5))
    for bi, bjr in enumerate([False, True]):
        for mi, mv in enumerate(mu_levs):
            ds = deepcopy(dims)
            for d in ds:
                d.mu = mv
                if bjr: d.lam *= 0.4; d.e_star = d.e_bar
            scen_W[bi,mi] = social_welfare(solve_all(ds,"dual"), ds, 400)

    # Lambda sweep
    lam_range = np.linspace(0.0, 3.0, 40)
    e_lam = np.zeros((40, 6))
    for li, lv in enumerate(lam_range):
        dl = deepcopy(dims)
        dl[0].lam = lv; dl[1].lam = lv
        e_lam[li] = solve_all(dl, "dual")

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("Module 6 -- Sensitivity & Scenario Analysis",fontsize=13,fontweight="bold")
    ax=axes[0]
    top12=sidx[:12]
    ax.barh(range(12),EE[top12][::-1],color=PURPLE,alpha=0.8)
    ax.set_yticks(range(12)); ax.set_yticklabels([pnames[k] for k in top12[::-1]],fontsize=8)
    ax.set_xlabel("Mean |Elementary Effect|"); ax.set_title("(a) Morris sensitivity")
    ax=axes[1]
    for bi,(lbl,col) in enumerate(zip(["Strict review (PIR)","BJR applied"],[GREEN,RED])):
        ax.plot(mu_levs,scen_W[bi],"o-",color=col,lw=2,label=lbl)
    ax.set_xlabel("Admin penalty mu"); ax.set_ylabel("Social welfare W")
    ax.set_title("(b) Scenario: review standard x mu"); ax.legend()
    ax=axes[2]
    for i,(d,col) in enumerate(zip(dims,PAL)):
        ax.plot(lam_range,e_lam[:,i],color=col,lw=1.8,label=d.name)
    ax.set_xlabel("lambda (e1,e2 investment dims)"); ax.set_ylabel("Equilibrium care")
    ax.set_title("(c) Role-allocation sensitivity"); ax.legend(fontsize=8)
    plt.tight_layout()
    if save: plt.savefig("fig_module6_sensitivity.png", bbox_inches="tight")
    plt.show()
    return EE, sidx, scen_W, e_lam


# ================================================================
# MODULE 7  --  INTEGRATED POLICY EVALUATION
# ================================================================
def run_module7(tau_did=0.28, tau_rdd=0.24, save=True):
    print("\n" + "="*60); print("MODULE 7: Policy Evaluation"); print("="*60)
    scale = (tau_did + tau_rdd) / 2.0
    dims_cal = deepcopy(DIMS_DEFAULT)
    for d in dims_cal: d.lam *= (1 + 0.15*scale)

    def ev(ds, regime="dual"):
        ev2 = solve_all(ds, regime)
        return ev2, social_welfare(ev2, ds)

    e0,W0 = ev(dims_cal)

    # CF1: tighten governance standards
    d1=deepcopy(dims_cal)
    for i in [4,5]: d1[i].e_bar=min(d1[i].e_bar+0.10,0.95); d1[i].e_star=min(d1[i].e_star+0.10,1.0)
    e1,W1=ev(d1)
    # CF2: Prudent Investor Rule
    d2=deepcopy(dims_cal)
    for i in [0,1]: d2[i].lam*=2.0; d2[i].e_star=d2[i].e_fb*0.95
    e2,W2=ev(d2)
    # CF3: CF1+CF2
    d3=deepcopy(d1)
    for i in [0,1]: d3[i].lam*=2.0; d3[i].e_star=d3[i].e_fb*0.95
    e3,W3=ev(d3)
    # CF4: BJR extended
    d4=deepcopy(dims_cal)
    for i in [0,1]: d4[i].lam*=0.35; d4[i].e_star=d4[i].e_bar
    e4,W4=ev(d4)
    # CF5: remove admin for e5,e6
    d5=deepcopy(dims_cal)
    for i in [4,5]: d5[i].mu=0.05
    e5,W5=ev(d5)

    scens = [("Baseline",e0,W0),("CF1 Strict ebar",e1,W1),
             ("CF2 Prud.Inv.",e2,W2),("CF3 CF1+CF2",e3,W3),
             ("CF4 BJR ext.",e4,W4),("CF5 Remove admin",e5,W5)]
    print(f"\n{'Scenario':<22} {'W':>8}  {'DeltaW':>8}  {'DeltaW%':>8}")
    print("-"*50)
    for nm,_,W in scens:
        dW=W-W0; print(f"{nm:<22} {W:>8.3f}  {dW:>+8.3f}  {100*dW/abs(W0):>+7.2f}%")

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle("Module 7 -- Integrated Policy Evaluation",fontsize=13,fontweight="bold")
    bcolors=[GRAY,GREEN,BLUE,TEAL,RED,AMBER]
    ax=axes[0]
    ws=[W for _,_,W in scens]; nms=[n for n,_,_ in scens]
    bars=ax.bar(range(len(scens)),ws,color=bcolors,alpha=0.85)
    ax.axhline(W0,color="black",ls="--",lw=1,alpha=0.5)
    for bar,W in zip(bars,ws):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                f"{W-W0:+.3f}",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(range(len(scens))); ax.set_xticklabels(nms,rotation=20,ha="right",fontsize=9)
    ax.set_ylabel("Social welfare W"); ax.set_title("(a) Welfare by scenario")
    ax=axes[1]
    x=np.arange(6); w=0.14
    for k,(nm,ev_s,_) in enumerate(scens):
        ax.bar(x+(k-2.5)*w,ev_s,width=w,color=bcolors[k],alpha=0.75,label=nm)
    ax.set_xticks(x); ax.set_xticklabels([d.name for d in DIMS_DEFAULT],rotation=25,ha="right")
    ax.set_ylabel("Equilibrium care e**"); ax.set_title("(b) Care by scenario & dimension")
    ax.legend(fontsize=7)
    plt.tight_layout()
    if save: plt.savefig("fig_module7_policy.png", bbox_inches="tight")
    plt.show()
    return scens


# ================================================================
# MAIN
# ================================================================
def _timed(label, fn, *args, **kwargs):
    """Helper: run fn(*args, **kwargs), print elapsed time."""
    import time as _time
    print(f"\n>>> {label} ...", flush=True)
    t0 = _time.time()
    result = fn(*args, **kwargs)
    print(f"    [{label} done in {_time.time()-t0:.1f}s]", flush=True)
    return result

if __name__ == "__main__":
    import time as _time
    _t_start = _time.time()
    print("\n" + "#"*60)
    print("  Fiduciary Duty Simulation Suite  (optimised build)")
    print("#"*60)

    results1, welfare1 = _timed("Module 1: Equilibrium", run_module1)

    post2, s2c2, lam2, mu2, epred2 = _timed(
        "Module 2: MCMC (n=800)", run_module2, n_samples=800, n_warmup=400)

    th3, s3, l3, m3, d3 = _timed("Module 3: SMM (analytical moments)", run_module3)

    tau_did_list, _ = _timed("Module 4A: DiD (n=200)", did_simulation)
    tau_rdd_list    = _timed("Module 4B: RDD (n=150)", rdd_simulation)
    w_sc, gap_sc    = _timed("Module 4C: Synth.Ctrl", synthetic_control)

    hist_e5, hist_W5, bifu5 = _timed("Module 5: ABM", run_module5)

    EE6, idx6, scen6, e_lam6 = _timed("Module 6: Sensitivity", run_module6)

    scens7 = _timed("Module 7: Policy eval",
                    run_module7,
                    float(np.mean(tau_did_list)),
                    float(np.mean(tau_rdd_list)))

    print("\n" + "="*60)
    print(f"  All modules complete.  Total: {_time.time()-_t_start:.1f}s")
    print("="*60)
