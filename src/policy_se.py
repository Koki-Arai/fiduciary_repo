# ============================================================
# policy_se.py
#
# Standard error estimation for policy simulation outputs.
# Propagates SMM parameter uncertainty (σ²_C, λ, μ) into:
#
#   Module 8E  Japan policy experiments  (ΔW for JP1–JP5)
#   Module 8D  Regulatory architectures  (W_net for 6 designs)
#   Module 8F  Welfare decomposition     (W_B, W_T_H, W_T_L)
#
# Method: Parametric bootstrap over θ=(log σ²_C, log λ, log μ)
#   For b=1..B:
#     (a) Perturb empirical moments → emp_b
#     (b) Re-estimate θ_b = argmin J(θ; emp_b)
#     (c) Update DIMS parameters from θ_b
#     (d) Run each policy module → record welfare outcomes
#   CI = percentile(5,95) across bootstrap draws
#
# Usage (Google Colab):
#   !pip install -q numpy scipy matplotlib pandas
#   # Run smm_se.py first (defines model primitives)
#   %run policy_se.py
#
# Or standalone:
#   exec(open("smm_se.py").read())  # loads model
#   theta_hat, boot_draws = run_policy_bootstrap(B=200)
# ============================================================

import numpy as np
import scipy.stats  as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import warnings, time
from copy import deepcopy

# ── Model primitives (from smm_se.py) ─────────────
# If running standalone, execute smm_se.py first.
try:
    _ = DIMS          # already loaded
except NameError:
    exec(open("smm_se.py").read())

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi":130,"figure.facecolor":"white",
    "axes.facecolor":"#F9F9F9","axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.4,
    "font.size":11,"axes.labelsize":11,"axes.titlesize":12,
    "legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,
})
BLUE="#1A6BB5"; GREEN="#1D9E75"; RED="#E24B4A"
AMBER="#F0992B"; GRAY="#888888"; NAVY="#0A2342"; PURPLE="#7F77DD"


# ──────────────────────────────────────────────────────────────
# 1.  Core: apply policy and compute welfare
#     (streamlined versions of Modules 8D, 8E, 8F)
# ──────────────────────────────────────────────────────────────

def _solve_all(dims_list, regime):
    return np.array([solve_e(d, regime) for d in dims_list])

def _W(e_vec, dims_list):
    """Analytical social welfare (fast, no MC)."""
    return sum(
        f(e_vec[i]) - c(e_vec[i], dims_list[i].alpha)
        - ED(e_vec[i], dims_list[i].D_bar)
        for i in range(len(dims_list))
    )

# ── 8E: Japan policy experiments ──────────────────────────────

def _apply_policy_8E(base, policy):
    """Apply one of the five Japan policy experiments."""
    ds = deepcopy(base)
    if policy == "JP1_PIR":
        for i in [0, 1]:
            ds[i].e_bar  = min(ds[i].e_bar  + 0.17, 0.90)
            ds[i].e_star = min(ds[i].e_star + 0.10, 0.92)
            ds[i].mu     = min(ds[i].mu * 1.20, 3.0)
    elif policy == "JP2_BJR":
        for i in [0, 1]:
            ds[i].e_bar  = ds[i].e_bar  * 0.78
            ds[i].e_star = ds[i].e_bar  * 0.78
            ds[i].mu     = ds[i].mu     * 0.58
    elif policy == "JP3_ROLE":
        for i in [0, 1]:
            ds[i].e_bar  = min(ds[i].e_bar  + 0.10, 0.85)
            ds[i].e_star = min(ds[i].e_star + 0.08, 0.90)
            ds[i].mu     = min(ds[i].mu     * 1.15, 3.0)
    elif policy == "JP4_JDC":
        for i in [4, 5]:
            ds[i].mu    = min(ds[i].mu * 2.5, 3.5)
            ds[i].e_bar = min(ds[i].e_bar + 0.08, 0.90)
    elif policy == "JP5_REFORM":
        for i in [0, 1]:
            ds[i].e_bar  = min(ds[i].e_bar  + 0.30, 0.93)
            ds[i].e_star = min(ds[i].e_star + 0.22, 0.96)
            ds[i].mu     = min(ds[i].mu * 1.50, 3.5)
        for i in [2, 3, 4, 5]:
            ds[i].mu    = min(ds[i].mu * 2.2, 3.5)
            ds[i].e_bar = min(ds[i].e_bar + 0.08, 0.93)
    return ds

POLICIES_8E = ["JP1_PIR","JP2_BJR","JP3_ROLE","JP4_JDC","JP5_REFORM"]
LABELS_8E   = ["JP1: PIR","JP2: BJR","JP3: Role","JP4: JDC","JP5: Reform"]

def eval_8E(dims_list):
    """Evaluate all 8E policy experiments. Returns dict of ΔW."""
    e_base = _solve_all(dims_list, "dual")
    W_base = _W(e_base, dims_list)
    results = {}
    for pol in POLICIES_8E:
        ds_p = _apply_policy_8E(dims_list, pol)
        e_p  = _solve_all(ds_p, "dual")
        W_p  = _W(e_p, ds_p)
        results[pol] = {"W": W_p, "dW": W_p - W_base,
                        "dW_pct": 100*(W_p-W_base)/max(abs(W_base),1e-9)}
    results["W_base"] = W_base
    return results

# ── 8D: Regulatory architectures ──────────────────────────────

ARCHS_8D = {
    "Civil only":   {"mu_s": 0.0, "lam_s": 1.0},
    "Admin only":   {"mu_s": 1.0, "lam_s": 0.0},
    "Dual current": {"mu_s": 1.0, "lam_s": 1.0},
    "Dual + PIR":   {"mu_s": 1.0, "lam_s": 2.0},
    "Dual + BJR":   {"mu_s": 1.0, "lam_s": 0.3, "bjr": True},
    "Optimal dual": {"mu_s": 1.2, "lam_s": 1.5},
}

def eval_8D(dims_list, c_R=0.05):
    """Evaluate 6 regulatory architectures. Returns dict of W_net."""
    results = {}
    for name, cfg in ARCHS_8D.items():
        ds = deepcopy(dims_list)
        for d in ds:
            d.mu  *= cfg["mu_s"]
            d.lam *= cfg["lam_s"]
        if cfg.get("bjr", False):
            for i in [0,1]: ds[i].e_star = ds[i].e_bar
        regime = ("dual" if cfg["mu_s"]>0 and cfg["lam_s"]>0
                  else "civil" if cfg["mu_s"]==0 else "admin")
        e = _solve_all(ds, regime)
        W = _W(e, ds)
        C = c_R * sum(d.mu**2 for d in ds) / 2.0
        results[name] = {"W": W, "C": C, "W_net": W - C}
    return results


def eval_8D_v2(dims_list, c_R=0.05):
    """
    Corrected architecture comparison.
    Architectures are defined by changes to (e_bar, e_star, regime)
    rather than just lam/mu scaling (which are ineffective when
    the admin floor is the binding constraint).
    """
    baseline_e  = _solve_all(dims_list, "dual")
    W0          = _W(baseline_e, dims_list)
    C0          = c_R * sum(d.mu**2 for d in dims_list) / 2.0

    def run_arch(dims_mod, regime, extra_mu_scale=1.0):
        ds = deepcopy(dims_mod)
        for d in ds: d.mu *= extra_mu_scale
        e  = _solve_all(ds, regime)
        W  = _W(e, ds)
        C  = c_R * sum(d.mu**2 for d in ds) / 2.0
        return {"W":W, "C":C, "W_net":W-C, "e":e}

    results = {}

    # A1: Civil only — remove admin (mu=0, e_bar=0)
    ds_a1 = deepcopy(dims_list)
    for d in ds_a1: d.mu=0.0; d.e_bar=0.0
    results["Civil only"] = run_arch(ds_a1, "civil")

    # A2: Admin only — remove civil liability (lam=0)
    ds_a2 = deepcopy(dims_list)
    for d in ds_a2: d.lam=0.0
    results["Admin only"] = run_arch(ds_a2, "admin")

    # A3: Dual current — baseline
    results["Dual current"] = run_arch(deepcopy(dims_list), "dual")

    # A4: Dual + PIR — raise e_bar, e_star for investment dims
    ds_a4 = deepcopy(dims_list)
    for i in [0,1]:
        ds_a4[i].e_bar  = min(ds_a4[i].e_bar  + 0.17, 0.90)
        ds_a4[i].e_star = min(ds_a4[i].e_star + 0.10, 0.92)
    results["Dual + PIR"] = run_arch(ds_a4, "dual")

    # A5: Dual + BJR — lower e_bar, e_star for investment dims
    ds_a5 = deepcopy(dims_list)
    for i in [0,1]:
        ds_a5[i].e_bar  *= 0.78
        ds_a5[i].e_star  = ds_a5[i].e_bar
    results["Dual + BJR"] = run_arch(ds_a5, "dual")

    # A6: Optimal dual — raise all e_bar toward first-best, reduce mu
    ds_a6 = deepcopy(dims_list)
    for d in ds_a6:
        d.e_bar  = min(d.e_bar + 0.20, 0.93)
        d.e_star = min(d.e_star + 0.15, 0.95)
    results["Optimal dual"] = run_arch(ds_a6, "dual", extra_mu_scale=0.5)

    return results
# ── 8F: Welfare decomposition ──────────────────────────────────

def eval_8F(dims_list, alpha_H=0.7, alpha_L=1.5, q=0.40, c_R=0.05):
    """
    Decompose welfare into beneficiary (W_B) and trustee (W_T) components
    under the six scenarios used in Module 8F.
    Returns dict of {scenario: {W_B, W_T_H, W_T_L, C_R, W_total}}.
    """
    scenarios = {
        "No enforcement": {"mu_s":0.0,"lam_s":0.0,"bjr":False},
        "Civil only":     {"mu_s":0.0,"lam_s":1.0,"bjr":False},
        "Admin only":     {"mu_s":1.0,"lam_s":0.0,"bjr":False},
        "Dual (current)": {"mu_s":1.0,"lam_s":1.0,"bjr":False},
        "Dual + PIR":     {"mu_s":1.0,"lam_s":2.0,"bjr":False},
        "Dual + BJR":     {"mu_s":1.0,"lam_s":0.3,"bjr":True},
    }
    records = {}
    for scen, cfg in scenarios.items():
        ds = deepcopy(dims_list)
        for d in ds: d.mu*=cfg["mu_s"]; d.lam*=cfg["lam_s"]
        if cfg["bjr"]:
            for i in [0,1]: ds[i].e_star=ds[i].e_bar
        regime=("dual" if cfg["mu_s"]>0 and cfg["lam_s"]>0
                else "civil" if cfg["mu_s"]==0 else "admin")
        W_T_H=W_T_L=W_B=0.0
        for alpha_t,key in [(alpha_H,"H"),(alpha_L,"L")]:
            ds_t=deepcopy(ds)
            for d in ds_t: d.alpha=alpha_t
            e_t=_solve_all(ds_t,regime)
            for i,d in enumerate(ds_t):
                fv=f(e_t[i]); ct=c(e_t[i],alpha_t)
                EDi=ED(e_t[i],d.D_bar)
                sc=max(np.sqrt(d.sigma2c),1e-9)
                liab=d.lam*Phi((d.e_star-e_t[i])/sc)*EDi
                if key=="H": W_T_H+=fv-ct-liab; W_B+=fv-EDi
                else:        W_T_L+=fv-ct-liab
        C_R=c_R*sum(d.mu**2 for d in ds)/2.0
        W_total=q*W_T_H+(1-q)*W_T_L-C_R
        records[scen]={"W_B":W_B/2,"W_T_H":W_T_H,
                       "W_T_L":W_T_L,"C_R":C_R,"W_total":W_total}
    return records


# ──────────────────────────────────────────────────────────────
# 2.  Bootstrap: propagate parameter uncertainty
#
# For b = 1..B:
#   (a) Perturb empirical moments eps_b ~ N(0, Σ_emp)/√n_eff
#   (b) Re-estimate θ_b = argmin J(θ; EMP + eps_b)
#   (c) Update DIMS from θ_b
#   (d) Evaluate 8E, 8D, 8F → store outcomes
# ──────────────────────────────────────────────────────────────

def policy_bootstrap(B=200, n_eff=13, verbose=True):
    """
    Propagate SMM parameter uncertainty into policy welfare outcomes.

    Parameters
    ----------
    B     : bootstrap replications
    n_eff : effective sample size (13 enforcement events)

    Returns
    -------
    dict with keys:
      'theta_hat'  : (18,) point estimate
      '8E'         : {policy: (B,) array of ΔW}
      '8D'         : {arch:   (B,) array of W_net}
      '8F'         : {scen:   {component: (B,) array}}
      'se_8E', 'ci90_8E', etc.
    """
    if verbose:
        print("="*62)
        print("POLICY SIMULATION — PARAMETRIC BOOTSTRAP SE")
        print(f"n_eff={n_eff}  |  B={B}")
        print("="*62)
        t_total = time.time()

    # ── First-stage SMM ────────────────────────────────────────
    theta_hat = first_stage(verbose=True)

    # ── Baseline policy outcomes at point estimates ────────────
    dims_base = _dims_from_theta(theta_hat)
    res8E_pt  = eval_8E(dims_base)
    res8D_pt  = eval_8D_v2(dims_base)
    res8F_pt  = eval_8F(dims_base)

    if verbose:
        print("\n── Point estimates (baseline) ──")
        print(f"  W_base = {res8E_pt['W_base']:.4f}")
        for p,lab in zip(POLICIES_8E,LABELS_8E):
            r=res8E_pt[p]
            print(f"  {lab:<14}: W={r['W']:.4f}  ΔW={r['dW']:+.4f}  ({r['dW_pct']:+.1f}%)")

    # ── Bootstrap ─────────────────────────────────────────────
    emp_flat = EMP.flatten()
    sig_d    = (0.15 * np.maximum(np.abs(emp_flat), 0.05))**2
    L        = np.diag(np.sqrt(sig_d))

    # Storage
    store_8E  = {p: np.zeros(B) for p in POLICIES_8E}
    store_8E["W_base"] = np.zeros(B)
    arch_keys = list(eval_8D_v2(dims_base).keys())
    store_8D  = {a: np.zeros(B) for a in arch_keys}
    store_8F  = {s: {k: np.zeros(B) for k in ["W_B","W_T_H","W_T_L","C_R","W_total"]}
                 for s in ["No enforcement","Civil only","Admin only",
                            "Dual (current)","Dual + PIR","Dual + BJR"]}

    failed = 0
    if verbose:
        print(f"\n── Bootstrap ({B} reps) ──")
        t0 = time.time()

    for b in range(B):
        # (a) Perturb moments
        eps   = (L @ RNG.standard_normal(N_DIM*N_MOM)) / np.sqrt(n_eff)
        emp_b = np.clip((emp_flat + eps).reshape(N_DIM, N_MOM), 0.01, 2.0)

        # (b) Re-estimate θ
        def Jb(th, _e=emp_b):
            try: g=(sim_moments(th)-_e).flatten(); return float(g@W@g)
            except: return 1e6
        rb = opt.minimize(Jb, theta_hat, method="Nelder-Mead",
                          options={"maxiter":350,"xatol":1e-4,"fatol":1e-5,"disp":False})
        th_b = rb.x if rb.fun < 20.0 else theta_hat
        if rb.fun >= 20.0: failed += 1

        # (c) Update DIMS
        dims_b = _dims_from_theta(th_b)

        # (d) Evaluate modules
        r8E = eval_8E(dims_b)
        r8D = eval_8D_v2(dims_b)
        r8F = eval_8F(dims_b)

        store_8E["W_base"][b] = r8E["W_base"]
        for p in POLICIES_8E:
            store_8E[p][b] = r8E[p]["dW"]
        for a in arch_keys:
            store_8D[a][b] = r8D[a]["W_net"]
        for s in store_8F:
            for k in store_8F[s]:
                store_8F[s][k][b] = r8F[s][k]

        if verbose and (b+1) % 50 == 0:
            el  = time.time()-t0
            eta = el/(b+1)*(B-b-1)
            print(f"  {b+1}/{B}  [{el:.0f}s  ETA {eta:.0f}s  failed {failed}]")

    if verbose:
        print(f"  Done. failed={failed}/{B}  total {time.time()-t_total:.0f}s")

    # ── Compute SE and CI ──────────────────────────────────────
    se_8E  = {p: store_8E[p].std()  for p in POLICIES_8E}
    ci_8E  = {p: np.percentile(store_8E[p],[5,95]) for p in POLICIES_8E}
    pct_8E = {p: 100*store_8E[p]/np.maximum(np.abs(store_8E["W_base"]),1e-9)
              for p in POLICIES_8E}
    ci_pct_8E = {p: np.percentile(pct_8E[p],[5,95]) for p in POLICIES_8E}

    se_8D  = {a: store_8D[a].std()  for a in arch_keys}
    ci_8D  = {a: np.percentile(store_8D[a],[5,95]) for a in arch_keys}

    return {
        "theta_hat":  theta_hat,
        "dims_base":  dims_base,
        # Point estimates
        "pt_8E": res8E_pt, "pt_8D": res8D_pt, "pt_8F": res8F_pt,
        # Bootstrap draws
        "store_8E": store_8E, "store_8D": store_8D, "store_8F": store_8F,
        # SE and CI
        "se_8E":  se_8E,  "ci_8E":  ci_8E,
        "ci_pct_8E": ci_pct_8E,
        "se_8D":  se_8D,  "ci_8D":  ci_8D,
        "failed": failed, "B": B,
    }


def _dims_from_theta(theta):
    """Build DIMS list from log-scale parameter vector θ."""
    s2c, lam, mu = unpack(theta)
    ds = deepcopy(DIMS)
    for i in range(N_DIM):
        ds[i].sigma2c = float(s2c[i])
        ds[i].lam     = float(lam[i])
        ds[i].mu      = float(mu[i])
    return ds


# ──────────────────────────────────────────────────────────────
# 3.  Print results table
# ──────────────────────────────────────────────────────────────

def print_results(res):
    pt   = res["pt_8E"]
    se   = res["se_8E"]
    ci   = res["ci_8E"]
    cipct= res["ci_pct_8E"]

    print("\n" + "="*72)
    print("MODULE 8E — JAPAN POLICY EXPERIMENTS: WELFARE EFFECTS WITH 90% CI")
    print("="*72)
    print(f"  W_base = {pt['W_base']:.4f}")
    print(f"\n  {'Policy':<18} {'ΔW(pt)':>9} {'SE(ΔW)':>9} "
          f"{'90% CI (ΔW)':>22}  {'ΔW% 90% CI':>22}")
    print("  " + "-"*82)
    for p,lab in zip(POLICIES_8E, LABELS_8E):
        dw_pt = pt[p]["dW"]
        pct_pt= pt[p]["dW_pct"]
        cil, cih = ci[p]
        pl, ph   = cipct[p]
        sig = "*" if cil*cih > 0 else " "  # CI excludes zero
        print(f"  {lab:<18} {dw_pt:>+9.4f} {se[p]:>9.4f} "
              f"[{cil:>+7.4f},{cih:>+7.4f}]{sig}  "
              f"[{pl:>+6.1f}%,{ph:>+6.1f}%]{sig}")
    print("\n  * = 90% CI excludes zero (direction robust to parameter uncertainty)")

    print("\n" + "="*72)
    print("MODULE 8D — REGULATORY ARCHITECTURE: W_NET WITH 90% CI")
    print("="*72)
    pt8D = res["pt_8D"]
    ci8D = res["ci_8D"]
    se8D = res["se_8D"]
    W0   = pt8D["Dual current"]["W_net"]
    print(f"  {'Architecture':<20} {'W_net(pt)':>10} {'SE':>8} "
          f"{'90% CI':>22}  {'ΔW_net vs dual':>16}")
    print("  " + "-"*78)
    for a in res['pt_8D']:
        wn = pt8D[a]["W_net"]
        cil,cih = ci8D[a]
        dw = wn-W0
        sig="*" if (cil-W0)*(cih-W0)>0 else " "
        print(f"  {a:<20} {wn:>10.4f} {se8D[a]:>8.4f} "
              f"[{cil:>7.4f},{cih:>7.4f}]  {dw:>+10.4f}{sig}")
    print("\n  * = CI excludes Dual current (architecture distinction is robust)")


# ──────────────────────────────────────────────────────────────
# 4.  Figures
# ──────────────────────────────────────────────────────────────

def plot_8E(res, save=True):
    """Panel figure: 8E policy experiments with 90% CI bands."""
    pt  = res["pt_8E"]
    ci  = res["ci_8E"]
    se  = res["se_8E"]
    st  = res["store_8E"]
    B   = res["B"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Module 8E — Japan Policy Welfare Effects: 90% CI",
                 fontsize=13, fontweight="bold")

    # Panel A: ΔW with 90% CI
    ax   = axes[0]
    dws  = [pt[p]["dW"]  for p in POLICIES_8E]
    cols = [GREEN if dw>0 else RED for dw in dws]
    x    = np.arange(len(POLICIES_8E))
    bars = ax.bar(x, dws, color=cols, alpha=0.75, width=0.55)
    # Error bars from 90% CI
    lo   = [pt[p]["dW"] - ci[p][0] for p in POLICIES_8E]
    hi   = [ci[p][1] - pt[p]["dW"] for p in POLICIES_8E]
    ax.errorbar(x, dws, yerr=[lo,hi], fmt="none",
                color="black", capsize=5, lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_8E, rotation=25, ha="right")
    ax.set_ylabel("ΔW  (welfare change vs baseline)")
    ax.set_title("(a) ΔW with bootstrap 90% CI")
    # Annotate significance
    for i,p in enumerate(POLICIES_8E):
        cil,cih=ci[p]
        if cil*cih>0:
            ax.text(i, max(abs(dws[i])*1.05,0.01)*np.sign(dws[i]),
                    "*",ha="center",fontsize=14,color="black")

    # Panel B: Bootstrap distributions of ΔW
    ax = axes[1]
    cols_b=[GREEN,RED,BLUE,AMBER,PURPLE]
    for i,(p,lab,col) in enumerate(zip(POLICIES_8E,LABELS_8E,cols_b)):
        ax.hist(st[p], bins=25, alpha=0.55, density=True,
                color=col, label=lab)
        ax.axvline(pt[p]["dW"], color=col, lw=2, ls="--")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ΔW"); ax.set_ylabel("Density")
    ax.set_title("(b) Bootstrap distributions of ΔW")
    ax.legend(fontsize=8)

    # Panel C: ΔW% with CI — ordered by point estimate
    ax   = axes[2]
    pcts_pt = [pt[p]["dW_pct"] for p in POLICIES_8E]
    order   = np.argsort(pcts_pt)[::-1]
    x2      = np.arange(len(POLICIES_8E))
    dws_ord = [pcts_pt[o]          for o in order]
    labs_ord= [LABELS_8E[o]        for o in order]
    pol_ord = [POLICIES_8E[o]      for o in order]
    ci_lo_ord= [pcts_pt[o] - res["ci_pct_8E"][POLICIES_8E[o]][0] for o in order]
    ci_hi_ord= [res["ci_pct_8E"][POLICIES_8E[o]][1] - pcts_pt[o] for o in order]
    cols_ord = [GREEN if v>0 else RED for v in dws_ord]
    ax.barh(x2, dws_ord, color=cols_ord, alpha=0.75, height=0.55)
    ax.errorbar(dws_ord, x2, xerr=[ci_lo_ord,ci_hi_ord],
                fmt="none",color="black",capsize=4,lw=1.3)
    ax.axvline(0, color="black",lw=0.8)
    ax.set_yticks(x2); ax.set_yticklabels(labs_ord)
    ax.set_xlabel("ΔW%  (% change vs baseline)")
    ax.set_title("(c) ΔW% — ordered, with 90% CI\n(* = CI excludes zero)")
    for i,(p,v) in enumerate(zip(pol_ord,dws_ord)):
        cil,cih=res["ci_pct_8E"][p]
        if cil*cih>0:
            ax.text(v+0.3*(1 if v>0 else -1),i,"*",
                    ha="center",va="center",fontsize=13)

    plt.tight_layout()
    if save: plt.savefig("policy_se_8E.png", bbox_inches="tight", dpi=130)
    plt.show()


def plot_8D(res, save=True):
    """Architecture comparison with 90% CI."""
    pt  = res["pt_8D"]
    ci  = res["ci_8D"]
    se  = res["se_8D"]
    st  = res["store_8D"]

    arch_names = list(ARCHS_8D.keys()) if False else list(res['pt_8D'].keys())
    W_nets     = [pt[a]["W_net"] for a in arch_names]
    W0         = pt["Dual current"]["W_net"]
    cols       = [RED,AMBER,BLUE,GREEN,RED,NAVY]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Module 8D — Regulatory Architecture W_net: 90% CI",
                 fontsize=13, fontweight="bold")

    # Panel A: W_net with CI
    ax = axes[0]
    x  = np.arange(len(arch_names))
    bars=ax.bar(x, W_nets, color=cols, alpha=0.75, width=0.55)
    lo_err=[W_nets[i]-ci[a][0] for i,a in enumerate(arch_names)]
    hi_err=[ci[a][1]-W_nets[i] for i,a in enumerate(arch_names)]
    ax.errorbar(x, W_nets, yerr=[lo_err,hi_err],
                fmt="none",color="black",capsize=5,lw=1.5)
    ax.axhline(W0, color="gray", ls="--", lw=1.2, label="Dual current")
    ax.set_xticks(x); ax.set_xticklabels(arch_names,rotation=25,ha="right")
    ax.set_ylabel("Net welfare W_net"); ax.set_title("(a) W_net by architecture")
    ax.legend()
    for i,(a,wn) in enumerate(zip(arch_names,W_nets)):
        cil,cih=ci[a]
        if (cil-W0)*(cih-W0)>0:
            ax.text(i,wn+0.01,"*",ha="center",fontsize=13)

    # Panel B: Bootstrap dist of dual vs civil-only gap
    ax=axes[1]
    gap_boot = st["Dual current"] - st["Civil only"]
    ax.hist(gap_boot, bins=30, color=BLUE, alpha=0.7, density=True,
            label="W_net(Dual) − W_net(Civil only)")
    ax.axvline(gap_boot.mean(), color=BLUE, lw=2, ls="--")
    ax.axvline(0, color="black", lw=1.2)
    ci_gap=np.percentile(gap_boot,[5,95])
    ax.axvline(ci_gap[0],color=BLUE,lw=1,ls=":",alpha=0.7)
    ax.axvline(ci_gap[1],color=BLUE,lw=1,ls=":",alpha=0.7)
    ax.set_xlabel("W_net(Dual) − W_net(Civil only)")
    ax.set_ylabel("Density")
    ax.set_title(f"(b) Bootstrap: dual advantage\n90% CI = [{ci_gap[0]:.3f},{ci_gap[1]:.3f}]")
    ax.legend()

    plt.tight_layout()
    if save: plt.savefig("policy_se_8D.png", bbox_inches="tight", dpi=130)
    plt.show()


def plot_8F(res, save=True):
    """Welfare decomposition with 90% CI bands."""
    st8F = res["store_8F"]
    pt8F = res["pt_8F"]
    scens= list(st8F.keys())
    x    = np.arange(len(scens))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Module 8F — Welfare Decomposition: Bootstrap 90% CI",
                 fontsize=13, fontweight="bold")

    for ax_idx, (comp, label, col) in enumerate([
        ("W_T_H", "W_T (High-ability trustee)", BLUE),
        ("W_T_L", "W_T (Low-ability trustee)",  AMBER),
        ("W_total","W_total (net)",              GREEN),
    ]):
        ax = axes[ax_idx]
        pts = [pt8F[s][comp] for s in scens]
        lo  = [pt8F[s][comp]-np.percentile(st8F[s][comp],5) for s in scens]
        hi  = [np.percentile(st8F[s][comp],95)-pt8F[s][comp] for s in scens]
        ax.bar(x, pts, color=col, alpha=0.75, width=0.55)
        ax.errorbar(x, pts, yerr=[lo,hi],
                    fmt="none",color="black",capsize=4,lw=1.3)
        ax.axhline(0,color="black",lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(scens,rotation=25,ha="right",fontsize=8)
        ax.set_ylabel(label); ax.set_title(f"({chr(97+ax_idx)}) {label}\nwith 90% CI")

    plt.tight_layout()
    if save: plt.savefig("policy_se_8F.png", bbox_inches="tight", dpi=130)
    plt.show()


# ──────────────────────────────────────────────────────────────
# 5.  Save to CSV
# ──────────────────────────────────────────────────────────────

def save_csv(res):
    # 8E table
    rows_8E=[]
    pt=res["pt_8E"]
    for p,lab in zip(POLICIES_8E,LABELS_8E):
        dw=pt[p]["dW"]; pct=pt[p]["dW_pct"]
        cil,cih=res["ci_8E"][p]
        pl,ph=res["ci_pct_8E"][p]
        rows_8E.append({
            "Policy":lab,"dW_pt":dw,"SE_dW":res["se_8E"][p],
            "CI90_lo_dW":cil,"CI90_hi_dW":cih,
            "dW_pct_pt":pct,"CI90_lo_pct":pl,"CI90_hi_pct":ph,
            "sign_robust":int(cil*cih>0),
        })
    pd.DataFrame(rows_8E).to_csv("policy_se_8E.csv",index=False,float_format="%.4f")

    # 8D table
    rows_8D=[]
    pt8D=res["pt_8D"]; W0=pt8D["Dual current"]["W_net"]
    for a in res['pt_8D']:
        cil,cih=res["ci_8D"][a]
        rows_8D.append({
            "Architecture":a,
            "W_net_pt":pt8D[a]["W_net"],
            "SE_Wnet":res["se_8D"][a],
            "CI90_lo":cil,"CI90_hi":cih,
            "dW_vs_dual":pt8D[a]["W_net"]-W0,
        })
    pd.DataFrame(rows_8D).to_csv("policy_se_8D.csv",index=False,float_format="%.4f")

    print("Saved: policy_se_8E.csv, policy_se_8D.csv")


# ──────────────────────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────────────────────

def run_all(B=200, n_eff=13, save=True):
    res = policy_bootstrap(B=B, n_eff=n_eff, verbose=True)
    print_results(res)
    plot_8E(res, save=save)
    plot_8D(res, save=save)
    plot_8F(res, save=save)
    save_csv(res)
    if save:
        print("Saved: policy_se_8E.png, policy_se_8D.png, policy_se_8F.png")
    return res


if __name__ == "__main__":
    res = run_all(B=200, n_eff=13)

# ──────────────────────────────────────────────────────────────
# CORRECTED 8D: architectures defined by (e_bar, e_star) shifts
# not just lam/mu scaling (which have no effect when floor binds)
# ──────────────────────────────────────────────────────────────

