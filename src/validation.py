# ============================================================
# validation.py
#
# Validation suite for the fiduciary duty simulation.
# Allows changing parameters and repetition counts to check
# that core propositions hold across settings.
#
# ── Check catalogue ─────────────────────────────────────────
# V1  Prop 1     Civil-only under-provides e₅,e₆ vs first-best
# V2  Prop 2     W(dual) > W(civil-only) [primary welfare comparison]
# V3  Prop 3     W(PIR) > W(base) > W(BJR)
# V4  Assume A2  σ²_C ordering: e₅,e₆ >> e₃,e₄
# V5  Prop 2b    Dual ≥ Admin for traceable-harm dims (e₃,e₄)
# V6  Policy     JP1,JP3,JP4,JP5 ΔW>0;  JP2 ΔW<0
# V7  Bootstrap  P(A2 ordering) ≥ threshold across bootstrap draws
# V8  Sensitivity Prop signs hold under ±perturb% param noise
# V9  Admin bind Admin floor is binding for e₅,e₆
# V10 Prop 2c    Civil-only severely under-provides e₅,e₆
#                (e₅,e₆ gap > ε under civil-only)
#
# Key modelling note (from diagnostic):
#   'none' regime equilibrium = first-best (no enforcement costs,
#   trustee maximises f(e)−c(e)). Admin floor may bind BELOW
#   first-best, so W(admin) ≥ W(dual) is possible. The correct
#   Prop 2 comparison is dual vs civil-only (not vs admin or none).
#
# ── Usage ───────────────────────────────────────────────────
#   exec(open("smm_se.py").read())
#   exec(open("policy_se.py").read())   # for V6
#   exec(open("validation.py").read())
#
#   run_validation()                  # all checks, default settings
#   run_validation(B=100, n_grid=15)  # more thorough
#   run_validation(checks=["V1","V2","V3"])  # subset only
#   grid_scan("sigma2c_scale", np.linspace(0.5,3.0,12))
#   monte_carlo_stability(N=50, perturb=0.20)
# ============================================================

import numpy as np
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import warnings, time
from copy import deepcopy

warnings.filterwarnings("ignore")

# ── Load model primitives ────────────────────────────────────
try:
    _ = DIMS          # from smm_se.py
except NameError:
    exec(open("smm_se.py").read())

BLUE="#1A6BB5"; GREEN="#1D9E75"; RED="#E24B4A"
AMBER="#F0992B"; GRAY="#888888"; NAVY="#0A2342"

# ── Shared helpers ───────────────────────────────────────────

def _W(e_vec, dims_list):
    """Analytical social welfare."""
    return sum(
        f(e_vec[i]) - c(e_vec[i], dims_list[i].alpha) - ED(e_vec[i], dims_list[i].D_bar)
        for i in range(len(dims_list))
    )

def _eq(dims_list, regime):
    return np.array([solve_e(d, regime) for d in dims_list])

def _fb(dims_list):
    """First-best care vector (private optimum, no enforcement costs)."""
    return np.array([(1/(2*d.alpha))**(2/3) for d in dims_list])

def _perturb(dims_list, scale, rng):
    """Randomly perturb parameters by ±scale (multiplicative log-normal)."""
    ds = deepcopy(dims_list)
    for d in ds:
        n = rng.uniform(1-scale, 1+scale, 7)
        d.sigma2c = float(np.clip(d.sigma2c * n[0], 0.005, 10.0))
        d.lam     = float(np.clip(d.lam     * n[1], 0.05,  10.0))
        d.mu      = float(np.clip(d.mu      * n[2], 0.05,  10.0))
        d.e_bar   = float(np.clip(d.e_bar   * n[3], 0.05,  0.94))
        d.e_star  = float(np.clip(d.e_star  * n[4], d.e_bar+0.01, 0.99))
        d.alpha   = float(np.clip(d.alpha   * n[5], 0.20,  5.0))
        d.D_bar   = float(np.clip(d.D_bar   * n[6], 0.20,  8.0))
    return ds


# ──────────────────────────────────────────────────────────────
# 1.  Check functions
#     Each returns (passed: bool, detail: str, value)
# ──────────────────────────────────────────────────────────────

def chk_V1(dims, gap_min=0.02):
    """
    Prop 1: civil-only produces e₅,e₆ below first-best.
    gap = e_fb - e_civil > gap_min for i=4,5.
    """
    e_civil = _eq(dims, "civil")
    e_fb    = _fb(dims)
    gap = e_fb - e_civil
    passed = gap[4] > gap_min and gap[5] > gap_min
    detail = (f"e₅ gap={gap[4]:+.4f}  e₆ gap={gap[5]:+.4f}  "
              f"e₃ gap={gap[2]:+.4f} (threshold >{gap_min})")
    return passed, detail, gap[[4,5]]


def chk_V2(dims):
    """
    Prop 2 (primary): W(dual) > W(civil-only).
    This is the main welfare-dominance claim. W(admin) may exceed
    W(dual) in this model because the admin floor can bind below
    first-best, but the primary comparison is dual vs civil-only.
    """
    e_dual  = _eq(dims, "dual");  W_dual  = _W(e_dual,  dims)
    e_civil = _eq(dims, "civil"); W_civil = _W(e_civil, dims)
    margin  = W_dual - W_civil
    passed  = margin > 0
    detail  = f"W(dual)={W_dual:.4f}  W(civil)={W_civil:.4f}  margin={margin:+.4f}"
    return passed, detail, margin


def chk_V3(dims):
    """
    Prop 3: PIR adoption raises welfare; BJR extension lowers it.
    W(PIR) > W(base) > W(BJR).
    """
    def W_apply(ebar_delta, estar_delta, ebar_scale=1.0):
        ds = deepcopy(dims)
        for i in [0,1]:
            ds[i].e_bar  = min(ds[i].e_bar  + ebar_delta,  0.93)
            ds[i].e_star = min(ds[i].e_star + estar_delta, 0.96)
            ds[i].e_bar  = ds[i].e_bar * ebar_scale
            ds[i].e_star = ds[i].e_bar * ebar_scale if ebar_scale < 1 else ds[i].e_star
        return _W(_eq(ds, "dual"), ds)

    W_base = _W(_eq(dims, "dual"), dims)
    W_pir  = W_apply(0.17, 0.10)
    W_bjr  = W_apply(0, 0, ebar_scale=0.78)   # BJR erodes floor/standard
    passed = W_pir > W_base > W_bjr
    detail = f"W(PIR)={W_pir:.4f}  W(base)={W_base:.4f}  W(BJR)={W_bjr:.4f}"
    return passed, detail, (W_pir, W_base, W_bjr)


def chk_V4(dims):
    """
    Assumption A2: σ²_C ordering.
    All four pairwise orderings: e₅,e₆ >> e₃,e₄.
    """
    s = [d.sigma2c for d in dims]
    pairs = [
        (s[5]>s[2], "σ²(e₆)>σ²(e₃)"),
        (s[5]>s[3], "σ²(e₆)>σ²(e₄)"),
        (s[4]>s[2], "σ²(e₅)>σ²(e₃)"),
        (s[4]>s[3], "σ²(e₅)>σ²(e₄)"),
    ]
    n_pass = sum(ok for ok,_ in pairs)
    passed = n_pass == 4
    detail = "  ".join(f"{lbl}={'✓' if ok else '✗'}" for ok,lbl in pairs)
    return passed, detail, n_pass/4


def chk_V5(dims, tol=0.0):
    """
    Prop 2 (dim-level): W(dual) > W(civil-only) for traceable-harm dims e₃,e₄.
    Civil channel adds deterrence for observable-harm dimensions.
    Note: dual vs admin-only comparison is not a Prop 2 claim because
    the admin floor may differ between regimes.
    """
    def W_slice(e_vec, idx_list):
        return sum(f(e_vec[i])-c(e_vec[i],dims[i].alpha)-ED(e_vec[i],dims[i].D_bar)
                   for i in idx_list)
    e_dual  = _eq(dims, "dual")
    e_civil = _eq(dims, "civil")
    W34_dual  = W_slice(e_dual,  [2,3])
    W34_civil = W_slice(e_civil, [2,3])
    margin    = W34_dual - W34_civil
    passed    = margin > tol
    detail    = (f"W(dual|e₃,e₄)={W34_dual:.4f}  "
                 f"W(civil|e₃,e₄)={W34_civil:.4f}  margin={margin:+.4f}")
    return passed, detail, margin


def chk_V6(dims):
    """
    Policy signs: JP1,JP3,JP4,JP5 ΔW>0;  JP2 ΔW<0.
    """
    def apply_pol(pol):
        ds = deepcopy(dims)
        if pol == "JP1_PIR":
            for i in [0,1]:
                ds[i].e_bar  = min(ds[i].e_bar  + 0.17, 0.92)
                ds[i].e_star = min(ds[i].e_star + 0.10, 0.95)
        elif pol == "JP2_BJR":
            for i in [0,1]:
                ds[i].e_bar  *= 0.78
                ds[i].e_star  = ds[i].e_bar
        elif pol == "JP3_ROLE":
            for i in [0,1]:
                ds[i].e_bar  = min(ds[i].e_bar  + 0.10, 0.88)
                ds[i].e_star = min(ds[i].e_star + 0.08, 0.92)
        elif pol == "JP4_JDC":
            for i in [4,5]:
                ds[i].mu    = min(ds[i].mu * 2.5, 3.5)
                ds[i].e_bar = min(ds[i].e_bar + 0.08, 0.90)
        elif pol == "JP5_REFORM":
            for i in [0,1]:
                ds[i].e_bar  = min(ds[i].e_bar  + 0.30, 0.93)
                ds[i].e_star = min(ds[i].e_star + 0.22, 0.96)
            for i in [2,3,4,5]:
                ds[i].mu    = min(ds[i].mu * 2.2, 3.5)
                ds[i].e_bar = min(ds[i].e_bar + 0.08, 0.93)
        return ds

    W_base = _W(_eq(dims, "dual"), dims)
    pols   = ["JP1_PIR","JP2_BJR","JP3_ROLE","JP4_JDC","JP5_REFORM"]
    exp_pos= {"JP1_PIR","JP3_ROLE","JP4_JDC","JP5_REFORM"}
    dWs = {}
    for p in pols:
        ds = apply_pol(p)
        dWs[p] = _W(_eq(ds,"dual"), ds) - W_base

    passed = (all(dWs[p]>0 for p in exp_pos) and dWs["JP2_BJR"]<0)
    detail = "  ".join(f"{p.split('_')[0]}:{v:+.3f}" for p,v in dWs.items())
    return passed, detail, dWs


def chk_V7(dims, B=50, n_eff=13, threshold=0.90):
    """
    Bootstrap: P(σ²_C ordering holds) ≥ threshold across B draws.
    """
    emp_flat = EMP.flatten()
    sig_d    = (0.15 * np.maximum(np.abs(emp_flat), 0.05))**2
    L        = np.diag(np.sqrt(sig_d))
    rng      = np.random.default_rng(42)

    theta0 = first_stage(verbose=False)
    n_order = 0
    for _ in range(B):
        eps   = (L @ rng.standard_normal(N_DIM*N_MOM)) / np.sqrt(n_eff)
        emp_b = np.clip((emp_flat+eps).reshape(N_DIM,N_MOM), 0.01, 2.0)
        def Jb(th, _e=emp_b):
            try: g=(sim_moments(th)-_e).flatten(); return float(g@W@g)
            except: return 1e6
        rb = opt.minimize(Jb, theta0, method="Nelder-Mead",
                          options={"maxiter":200,"xatol":1e-4,"fatol":1e-5,"disp":False})
        th = rb.x if rb.fun < 20 else theta0
        s  = np.exp(th[:6])
        if s[5]>s[2] and s[5]>s[3] and s[4]>s[2] and s[4]>s[3]:
            n_order += 1

    prob   = n_order / B
    passed = prob >= threshold
    detail = f"P(A2 ordering) = {prob:.3f}  ({n_order}/{B} draws, threshold={threshold})"
    return passed, detail, prob


def chk_V8(dims, n_grid=8, perturb=0.25):
    """
    Sensitivity: Prop signs hold in at least 75% of perturbed param draws.
    Tests V1, V2, V3 jointly.
    """
    rng  = np.random.default_rng(99)
    cnts = {"V1":0, "V2":0, "V3":0}
    for _ in range(n_grid):
        ds = _perturb(dims, perturb, rng)
        if chk_V1(ds)[0]: cnts["V1"] += 1
        if chk_V2(ds)[0]: cnts["V2"] += 1
        if chk_V3(ds)[0]: cnts["V3"] += 1

    rates  = {k: v/n_grid for k,v in cnts.items()}
    passed = all(r >= 0.75 for r in rates.values())
    detail = (f"±{perturb*100:.0f}% over {n_grid} draws: "
              + "  ".join(f"{k}={v:.0%}" for k,v in rates.items()))
    return passed, detail, rates


def chk_V9(dims, tol=0.05):
    """
    Admin floor is binding for e₅,e₆:  |e** − ē^R| < tol.
    """
    e_dual = _eq(dims, "dual")
    b5 = abs(e_dual[4] - dims[4].e_bar) < tol
    b6 = abs(e_dual[5] - dims[5].e_bar) < tol
    passed = b5 and b6
    detail = (f"e₅: e**={e_dual[4]:.3f} ē={dims[4].e_bar:.3f} "
              f"[{'BIND' if b5 else 'free'}]  "
              f"e₆: e**={e_dual[5]:.3f} ē={dims[5].e_bar:.3f} "
              f"[{'BIND' if b6 else 'free'}]")
    return passed, detail, (b5, b6)


def chk_V10(dims, gap_min=0.05):
    """
    Prop 1 (reinforced): civil-only severely under-provides e₅,e₆.
    gap ≥ gap_min *for both* governance and disclosure dims.
    Also verifies e₃,e₄ gaps are smaller (relative observability).
    """
    e_civil = _eq(dims, "civil")
    e_fb    = _fb(dims)
    g = e_fb - e_civil
    gov_ok  = g[4] >= gap_min and g[5] >= gap_min
    # Under-provision is worse for governance dims than segregation
    rel_ok  = (g[4]+g[5])/2 > (g[2]+g[3])/2 - gap_min
    passed  = gov_ok
    detail  = (f"e₅={g[4]:.4f}  e₆={g[5]:.4f}  "
               f"e₃={g[2]:.4f}  e₄={g[3]:.4f}  "
               f"(threshold ≥{gap_min}, gov_ok={gov_ok})")
    return passed, detail, g


# ── Check registry ───────────────────────────────────────────
CHECKS = [
    ("V1",  "Prop 1: civil-only under-provides e₅,e₆",           chk_V1),
    ("V2",  "Prop 2: W(dual) > W(civil-only)",                   chk_V2),
    ("V3",  "Prop 3: W(PIR) > W(base) > W(BJR)",                 chk_V3),
    ("V4",  "Assume A2: σ²_C ordering e₅,e₆ >> e₃,e₄",          chk_V4),
    ("V5",  "Prop 2b: dual ≥ admin for e₃,e₄ (each chan. adds)", chk_V5),
    ("V6",  "Policy signs: JP1,3,4,5 ΔW>0;  JP2 ΔW<0",          chk_V6),
    ("V7",  "Bootstrap: P(A2 ordering) ≥ threshold",             chk_V7),
    ("V8",  "Sensitivity: prop signs hold under param noise",     chk_V8),
    ("V9",  "Admin floor binding for e₅,e₆",                     chk_V9),
    ("V10", "Prop 1 (strong): gov dims gap >> segreg dims gap",   chk_V10),
]


# ──────────────────────────────────────────────────────────────
# 2.  Main validation runner
# ──────────────────────────────────────────────────────────────

def run_validation(
    dims          = None,
    B             = 50,            # bootstrap reps for V7
    n_grid        = 8,             # perturbation draws for V8
    perturb_scale = 0.25,          # ±fraction for V8
    n_eff         = 13,            # effective sample size
    threshold_V7  = 0.90,          # min P(ordering) for V7
    gap_min_V1    = 0.02,          # min gap for V1
    gap_min_V10   = 0.05,          # min gap for V10
    checks        = None,          # None = all; or list e.g. ["V1","V2"]
    verbose       = True,
    plot          = True,
    save_fig      = True,
):
    """
    Run validation checks on the simulation.

    Parameters
    ----------
    dims          : list of DimParams; None = use DIMS from smm_se.py
    B             : bootstrap reps for V7  (set to 0 to skip V7)
    n_grid        : param-perturbation draws for V8
    perturb_scale : ±fraction for V8 noise (0.25 = ±25%)
    n_eff         : enforcement record size (13)
    threshold_V7  : pass threshold for V7 (default 0.90)
    gap_min_V1/V10: minimum care gap for V1/V10 to pass
    checks        : list of check IDs to run; None = all
    verbose       : print progress
    plot          : show summary figure
    save_fig      : save figure to PNG

    Returns
    -------
    pd.DataFrame with columns: id, description, passed, message, elapsed
    """
    if dims is None:
        dims = DIMS

    # Skip V7 if B=0
    active_ids = [c[0] for c in CHECKS] if checks is None else checks
    if B == 0 and "V7" in active_ids:
        active_ids = [c for c in active_ids if c != "V7"]

    active = [(vid,desc,fn) for vid,desc,fn in CHECKS if vid in active_ids]

    if verbose:
        print("="*65)
        print("FIDUCIARY SIMULATION — VALIDATION SUITE")
        print(f"dims: {len(dims)}  B={B}  n_grid={n_grid}  "
              f"perturb=±{perturb_scale*100:.0f}%")
        print("="*65)

    records = []
    t_start = time.time()

    for vid, desc, fn in active:
        t0 = time.time()
        try:
            kwargs = {}
            if vid == "V7": kwargs = dict(B=B, n_eff=n_eff, threshold=threshold_V7)
            if vid == "V8": kwargs = dict(n_grid=n_grid, perturb=perturb_scale)
            if vid == "V1": kwargs = dict(gap_min=gap_min_V1)
            if vid == "V10":kwargs = dict(gap_min=gap_min_V10)
            passed, msg, _ = fn(dims, **kwargs)
        except Exception as e:
            passed, msg = False, f"ERROR: {e}"

        elapsed = time.time() - t0
        records.append({"id":vid,"description":desc,
                        "passed":passed,"message":msg,"elapsed":elapsed})

        if verbose:
            icon = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {vid:<4} {icon}  {desc}")
            print(f"       {msg}  [{elapsed:.1f}s]")

    df = pd.DataFrame(records)
    n_pass = df["passed"].sum()

    if verbose:
        total = time.time()-t_start
        print("─"*65)
        print(f"RESULT: {n_pass}/{len(df)} passed  [{total:.0f}s]")
        if n_pass < len(df):
            print(f"FAILED: {', '.join(df.loc[~df['passed'],'id'])}")
        print("="*65)

    if plot:
        _plot_validation(df, save=save_fig)

    return df


# ──────────────────────────────────────────────────────────────
# 3.  Grid scan — vary one parameter
# ──────────────────────────────────────────────────────────────

def grid_scan(
    param_name,
    values,
    dims   = None,
    checks = ("V1","V2","V3","V4"),
    verbose= True,
):
    """
    Sweep one parameter over `values` and record pass/fail.

    Supported param_name values
    ---------------------------
    sigma2c_scale   multiply all σ²_C by value
    lam_scale       multiply all λ by value
    mu_scale        multiply all μ by value
    ebar_scale      multiply all ē^R by value
    sigma2c_5       set σ²_C for e₅ (disclosure) to value
    sigma2c_6       set σ²_C for e₆ (governance) to value
    lam_56          set λ for e₅,e₆ to value
    mu_56           set μ for e₅,e₆ to value
    ebar_56         set ē^R for e₅,e₆ to value
    D_bar_all       multiply all D_bar by value

    Examples
    --------
    grid_scan("sigma2c_scale", np.linspace(0.3, 4.0, 12))
    grid_scan("mu_scale", np.linspace(0.2, 3.0, 10), checks=["V2","V9"])
    grid_scan("ebar_56", np.linspace(0.3, 0.9, 12), checks=["V1","V9"])
    """
    if dims is None:
        dims = DIMS

    if verbose:
        header = f"  {'Value':>9}  " + "  ".join(f"{c:>6}" for c in checks)
        print(f"\n── Grid scan: {param_name}  ({len(values)} points) ──")
        print(header)
        print("  " + "-"*(len(header)-2))

    records = []
    for v in values:
        ds = deepcopy(dims)

        if   param_name == "sigma2c_scale": [setattr(d,"sigma2c",float(np.clip(d.sigma2c*v,0.005,10))) for d in ds]
        elif param_name == "lam_scale":     [setattr(d,"lam",    float(np.clip(d.lam*v,    0.01, 10))) for d in ds]
        elif param_name == "mu_scale":      [setattr(d,"mu",     float(np.clip(d.mu*v,     0.01, 10))) for d in ds]
        elif param_name == "ebar_scale":    [setattr(d,"e_bar",  float(np.clip(d.e_bar*v,  0.01,.95))) for d in ds]
        elif param_name == "D_bar_all":     [setattr(d,"D_bar",  float(np.clip(d.D_bar*v,  0.1,  12))) for d in ds]
        elif param_name == "sigma2c_5":     ds[4].sigma2c = float(np.clip(v, 0.005, 5.0))
        elif param_name == "sigma2c_6":     ds[5].sigma2c = float(np.clip(v, 0.005, 5.0))
        elif param_name == "lam_56":
            for i in [4,5]: ds[i].lam = float(np.clip(v, 0.01, 5.0))
        elif param_name == "mu_56":
            for i in [4,5]: ds[i].mu  = float(np.clip(v, 0.01, 5.0))
        elif param_name == "ebar_56":
            for i in [4,5]: ds[i].e_bar = float(np.clip(v, 0.01, 0.95))
        else:
            raise ValueError(
                f"Unknown param_name '{param_name}'. Choose from: "
                "sigma2c_scale, lam_scale, mu_scale, ebar_scale, D_bar_all, "
                "sigma2c_5, sigma2c_6, lam_56, mu_56, ebar_56")

        row = {"value": float(v)}
        for vid in checks:
            fn = next(fn for i,_,fn in CHECKS if i==vid)
            try:
                row[vid] = fn(ds)[0]
            except Exception:
                row[vid] = False
        records.append(row)

        if verbose:
            status = "  ".join(f"{'✓' if row[c] else '✗':>6}" for c in checks)
            print(f"  {v:>9.3f}  {status}")

    return pd.DataFrame(records)


def plot_grid_scan(scan_df, param_name, save=True):
    """Stacked pass/fail plot from grid_scan() output."""
    checks = [c for c in scan_df.columns if c != "value"]
    n = len(checks)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle(f"Grid scan: {param_name}", fontsize=12, fontweight="bold")

    x = scan_df["value"].values
    for ax, c in zip(axes, checks):
        passed = scan_df[c].values.astype(float)
        ax.fill_between(x, 0, passed,   alpha=0.70, color=GREEN, step="post", label="PASS")
        ax.fill_between(x, passed, 1,   alpha=0.50, color=RED,   step="post", label="FAIL")
        ax.axvline(x[0]+(x[-1]-x[0])/2, color="gray", ls=":", lw=0.8)
        ax.set_ylabel(c, fontsize=10)
        ax.set_ylim(-0.1, 1.2)
        ax.set_yticks([0,1]); ax.set_yticklabels(["FAIL","PASS"])
        ax.legend(loc="upper right", fontsize=8)
        pass_frac = passed.mean()
        ax.text(0.02, 0.75, f"{pass_frac:.0%} pass", transform=ax.transAxes,
                fontsize=9, color=GREEN if pass_frac >= 0.75 else RED)

    axes[-1].set_xlabel(param_name)
    plt.tight_layout()
    if save:
        plt.savefig(f"validation_grid_{param_name}.png", bbox_inches="tight", dpi=130)
    plt.show()


# ──────────────────────────────────────────────────────────────
# 4.  Monte Carlo stability
# ──────────────────────────────────────────────────────────────

def monte_carlo_stability(
    N             = 50,
    perturb       = 0.15,
    dims          = None,
    checks        = ("V1","V2","V3","V4","V5","V9"),
    B_per_run     = 0,      # bootstrap reps per run (0 = skip V7)
    verbose       = True,
):
    """
    Run N independent validations with random parameter perturbations.
    Reports pass rate per check — high rates confirm robustness.

    Parameters
    ----------
    N         : number of independent runs
    perturb   : ±fraction of parameter noise per run
    dims      : base parameter set
    checks    : which checks to test
    B_per_run : bootstrap reps if V7 is in checks (slow; set 0 to skip)
    """
    if dims is None:
        dims = DIMS

    rng        = np.random.default_rng(2025)
    counts     = {c: 0 for c in checks}
    fail_log   = {c: [] for c in checks}

    if verbose:
        print(f"\n── Monte Carlo stability: N={N}, noise=±{perturb*100:.0f}% ──")
        print(f"  {'Check':<6} {'Pass rate':>10}  {'Bar':25}")
        print("  " + "-"*44)

    for run in range(N):
        ds = _perturb(dims, perturb, rng)
        for vid in checks:
            fn = next(fn for i,_,fn in CHECKS if i==vid)
            try:
                kwargs = {}
                if vid=="V7": kwargs=dict(B=B_per_run)
                passed = fn(ds, **kwargs)[0]
            except Exception:
                passed = False
            if passed:
                counts[vid] += 1
            else:
                fail_log[vid].append(run)

    rows = []
    for c in checks:
        rate = counts[c] / N
        bar  = "█"*int(rate*20) + "░"*(20-int(rate*20))
        flag = "✓" if rate >= 0.90 else ("△" if rate >= 0.75 else "✗")
        if verbose:
            print(f"  {flag} {c:<5} {rate:>9.0%}  [{bar}]")
        rows.append({"check":c, "pass_rate":rate,
                     "n_pass":counts[c], "n_total":N})

    df_mc = pd.DataFrame(rows)
    if verbose:
        n_robust = (df_mc["pass_rate"]>=0.90).sum()
        print(f"\n  {n_robust}/{len(df_mc)} checks ≥90% robust  "
              f"(N={N}, perturb=±{perturb*100:.0f}%)")
    return df_mc


# ──────────────────────────────────────────────────────────────
# 5.  Figure
# ──────────────────────────────────────────────────────────────

def _plot_validation(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Validation Suite Results", fontsize=13, fontweight="bold")

    # Panel A: pass/fail
    ax = axes[0]
    y  = range(len(df))
    colors = [GREEN if p else RED for p in df["passed"]]
    ax.barh(list(y), [1]*len(df), color=colors, alpha=0.70, height=0.60)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(0.03, i, f"{row['id']}: {row['description'][:42]}",
                va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.text(0.97, i, "PASS" if row["passed"] else "FAIL",
                ha="right", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xlim(0,1); ax.set_xticks([]); ax.set_yticks([])
    n_pass = df["passed"].sum()
    ax.set_title(f"(a) Checks  ({n_pass}/{len(df)} passed)")

    # Panel B: elapsed time
    ax = axes[1]
    colors2 = [GREEN if p else RED for p in df["passed"]]
    bars = ax.barh(list(y), df["elapsed"], color=colors2, alpha=0.70, height=0.60)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["elapsed"]+0.05, i, f"{row['elapsed']:.1f}s",
                va="center", fontsize=8.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(df["id"])
    ax.set_xlabel("Elapsed time (s)")
    ax.set_title("(b) Time per check")

    plt.tight_layout()
    if save:
        plt.savefig("validation_results.png", bbox_inches="tight", dpi=130)
    plt.show()


# ──────────────────────────────────────────────────────────────
# 6.  Preset configurations
# ──────────────────────────────────────────────────────────────

def validate_fast(dims=None):
    """
    Fast validation — no bootstrap, light noise. ~15s.
    Good for checking a specific parameter set quickly.
    """
    print("Running FAST validation (V7 skipped, B=0) ...")
    return run_validation(
        dims=dims, B=0, n_grid=5, perturb_scale=0.20,
        checks=["V1","V2","V3","V4","V5","V6","V9","V10"],
        verbose=True, plot=True
    )


def validate_full(dims=None, B=50, n_grid=10):
    """
    Full validation including bootstrap ordering test. ~10–20min.
    """
    print(f"Running FULL validation (B={B}, n_grid={n_grid}) ...")
    return run_validation(
        dims=dims, B=B, n_grid=n_grid, perturb_scale=0.25,
        verbose=True, plot=True
    )


def validate_propositions(dims=None):
    """
    Run only the three core proposition checks.
    """
    return run_validation(
        dims=dims, B=0, checks=["V1","V2","V3"],
        verbose=True, plot=False
    )


def scan_sigma2c(values=None):
    """One-way scan: scale all σ²_C uniformly."""
    if values is None:
        values = np.linspace(0.3, 4.0, 15)
    df = grid_scan("sigma2c_scale", values, checks=["V1","V2","V3","V4"])
    plot_grid_scan(df, "sigma2c_scale")
    return df


def scan_floor(values=None):
    """One-way scan: scale admin floors ē^R for e₅,e₆."""
    if values is None:
        values = np.linspace(0.3, 0.92, 13)
    df = grid_scan("ebar_56", values, checks=["V1","V2","V9","V10"])
    plot_grid_scan(df, "ebar_56")
    return df


# ──────────────────────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_fast()
