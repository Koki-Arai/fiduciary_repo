# ============================================================
# smm_se.py  —  SMM Standard Error Estimation
#
# Computes SEs for the 18 calibrated SMM parameters:
#   θ = (log σ²_C, log λ, log μ) × 6 obligation dimensions
#
# Identification note
# -------------------
# With n_eff = 13 enforcement events and 18 parameters,
# the system is substantially under-identified:  only the
# 6 pre-enforcement care moments are informative (post-
# enforcement care is pinned to the admin floor, giving
# zero gradient for most parameters).  SEs therefore
# reflect calibration uncertainty given the moment targets,
# not sampling precision from a large dataset.
#
# Three SE methods:
#   1. Delta method on informative moments (6×2 block system)
#   2. Hessian diagonal (marginal curvature of J per param)
#   3. Parametric bootstrap (Pakes-Pollard 1989)
#
# Usage (Google Colab / local):
#   !pip install -q numpy scipy matplotlib pandas
#   %run smm_se.py     # → B=200 bootstrap reps
#   # or: theta, df = run_all(B=200, n_eff=13)
#
# References:
#   Pakes & Pollard (1989, Econometrica)
#   Newey & McFadden (1994, Handbook of Econometrics)
# ============================================================

import numpy as np
import scipy.stats  as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import warnings, time
from copy import deepcopy
from dataclasses import dataclass

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi":130,"figure.facecolor":"white",
    "axes.facecolor":"#F9F9F9","axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.4,
    "font.size":11,"axes.labelsize":11,"axes.titlesize":12,
    "legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,
})
BLUE="#1A6BB5"; GREEN="#1D9E75"; RED="#E24B4A"; AMBER="#F0992B"; GRAY="#888888"

# ── 0. Model ──────────────────────────────────────────────

@dataclass
class DimParams:
    name:str; sigma2c:float; sigma2r:float; lam:float; mu:float
    e_bar:float; e_star:float; alpha:float=1.0; D_bar:float=2.0

DIMS = [
    DimParams("e1_DD",      0.20,0.35,1.20,0.80,0.55,0.70,1.0,1.8),
    DimParams("e2_Monitor", 0.30,0.50,1.00,1.20,0.50,0.65,1.0,1.5),
    DimParams("e3_Segreg",  0.08,0.12,1.80,1.50,0.80,0.88,0.8,3.0),
    DimParams("e4_Conflict",0.12,0.18,1.50,1.20,0.70,0.82,0.9,2.5),
    DimParams("e5_Disclos", 0.60,0.18,0.50,1.60,0.58,0.68,1.2,1.2),
    DimParams("e6_Govern",  0.70,0.14,0.35,1.80,0.52,0.62,1.3,1.0),
]
N_DIM=6; N_MOM=4; N_PAR=18
phi=st.norm.pdf; Phi=st.norm.cdf

def breach(e,r,sc): return Phi((r-e)/max(sc,1e-9))
def dbreach(e,r,sc): sc=max(sc,1e-9); return -phi((r-e)/sc)/sc
def f(e): return np.sqrt(max(e,1e-9))
def fp(e): return 0.5/np.sqrt(max(e,1e-9))
def c(e,a): return a*e*e/2
def cp(e,a): return a*e

def ED(e,D,s=0.3):
    mv=np.sqrt(max(e,1e-9)); z=(1-mv)/s
    return D*(Phi(z)*(1-mv)+s*phi(z))

def solve_e(d:DimParams, regime:str)->float:
    lam=d.lam if regime in("civil","dual") else 0.0
    mu =d.mu  if regime in("admin","dual") else 0.0
    ebar=d.e_bar if mu>0 else 0.0
    sc=np.sqrt(max(d.sigma2c,1e-9))
    def foc(e): return fp(e)-cp(e,d.alpha)-lam*d.D_bar*(-dbreach(e,d.e_star,sc))
    def G(e):   return f(e)-c(e,d.alpha)-lam*breach(e,d.e_star,sc)*d.D_bar
    try: ei=opt.brentq(foc,1e-4,5.0,xtol=1e-7)
    except: ei=(1/(2*d.alpha))**(2/3)
    if mu>0 and ei<ebar:
        ei=ebar if G(ebar)>=G(ei)-mu else ei
    return float(np.clip(ei,0,5))

# ── 1. Moments & objective ─────────────────────────────────
#
# Empirical targets calibrated to the 13-event enforcement record.
# m_pre  = E[e | civil only, no admin floor]  (pre-enforcement)
# m_post = E[e | dual regime]                 (post-enforcement)
# m_dmg  = E[D | dual] / D_bar               (normalised)
# m_var  = cross-section variance proxy

EMP = np.array([
    # pre    post   dmg    var
    [0.18,  0.55,  0.20,  0.040],
    [0.22,  0.50,  0.28,  0.060],
    [0.27,  0.80,  0.50,  0.020],
    [0.20,  0.70,  0.40,  0.030],
    [0.40,  0.58,  0.12,  0.070],
    [0.44,  0.52,  0.10,  0.090],
])

W=np.eye(N_DIM*N_MOM)
for i in range(N_DIM):
    W[i*4,i*4]=2.5; W[i*4+1,i*4+1]=2.5

def unpack(theta):
    return np.exp(theta[:6]),np.exp(theta[6:12]),np.exp(theta[12:])

def sim_moments(theta)->np.ndarray:
    s2c,lam,mu=unpack(theta)
    mom=np.zeros((N_DIM,N_MOM))
    for i in range(N_DIM):
        d=deepcopy(DIMS[i])
        d.sigma2c=float(s2c[i]); d.lam=float(lam[i]); d.mu=float(mu[i])
        dp=deepcopy(d); dp.mu=0.0; dp.e_bar=0.0   # pre: civil only, no floor
        ep=solve_e(dp,"civil"); eq=solve_e(d,"dual")
        mom[i]=[ep, eq, ED(eq,d.D_bar)/3.0, (0.04*s2c[i])**2*0.05]
    return mom

def gap(theta,emp=EMP): return (sim_moments(theta)-emp).flatten()
def J(theta,emp=EMP):
    try: g=gap(theta,emp); return float(g@W@g)
    except: return 1e6

# ── 2. First-stage estimation ──────────────────────────────

def first_stage(verbose=True)->np.ndarray:
    theta0=np.array(
        [np.log(d.sigma2c) for d in DIMS]+
        [np.log(d.lam)     for d in DIMS]+
        [np.log(d.mu)      for d in DIMS])
    if verbose:
        print("="*62)
        print("FIRST-STAGE SMM ESTIMATION")
        print("="*62)
        t0=time.time()
    res=opt.minimize(J,theta0,method="Nelder-Mead",
                     options={"maxiter":800,"xatol":1e-5,"fatol":1e-6,"disp":False})
    s2c,lam,mu=unpack(res.x)
    if verbose:
        print(f"  J(θ̂)={res.fun:.5f}  [{time.time()-t0:.1f}s]")
        print(f"  σ²_C: {np.round(s2c,3)}")
        print(f"  λ   : {np.round(lam,3)}")
        print(f"  μ   : {np.round(mu, 3)}")
        ok=s2c[4]>s2c[2] and s2c[5]>s2c[3]
        print(f"  Ordering A2 (e₅,e₆ >> e₃,e₄): {'✓' if ok else '✗'}")
    return res.x

# ── 3. METHOD 1: Delta method (dimension-wise) ─────────────
#
# Exploiting the block-diagonal identification structure:
# for each dim i, only the pre-care moment (m1_i) is informative,
# and it identifies only (σ²_C_i, λ_i).
#
# For dim i with informative gradient g_i(σ²_C_i, λ_i):
#   SE(σ²_C_i) ≈ σ_emp_i / |∂g_i/∂σ²_C_i| × σ²_C_i   (delta method)
#   σ_emp_i = 0.15 × |m_pre_i|  (conservative 15% moment error)

def delta_se(theta_hat, n_eff=13, verbose=True)->dict:
    """Dimension-wise delta method on pre-care moments."""
    if verbose: print("\n── Method 1: Delta method (block-diagonal) ──")
    s2c,lam,mu=unpack(theta_hat)
    h=1e-4
    se_s2c=np.zeros(6); se_lam=np.zeros(6); se_mu=np.zeros(6)

    for i in range(6):
        # Gradient wrt log(σ²_C_i) for pre-care moment
        th_p=theta_hat.copy(); th_p[i]+=h
        th_m=theta_hat.copy(); th_m[i]-=h
        g_p=gap(th_p); g_m=gap(th_m)
        dg_ds2c=(g_p[i*4]-g_m[i*4])/(2*h)     # pre-care row for dim i

        # Gradient wrt log(λ_i)
        th_p=theta_hat.copy(); th_p[6+i]+=h
        th_m=theta_hat.copy(); th_m[6+i]-=h
        g_p=gap(th_p); g_m=gap(th_m)
        dg_dlam=(g_p[i*4]-g_m[i*4])/(2*h)

        # Moment sampling error: σ_emp = 15% of moment magnitude / √n_eff
        sigma_emp=0.15*max(abs(EMP[i,0]),0.02)/np.sqrt(n_eff)

        # Delta method: SE(log σ²_C) = σ_emp / |∂g/∂log σ²_C|
        se_ls2c = sigma_emp/max(abs(dg_ds2c),1e-8)
        se_llam = sigma_emp/max(abs(dg_dlam),1e-8)

        # Convert log-space SE to level-space (delta method: SE_level ≈ |param|*SE_log)
        se_s2c[i]=s2c[i]*se_ls2c
        se_lam[i]=lam[i]*se_llam
        # μ is not identified from pre-care moment → report NaN (identified from admin floor binding)
        se_mu[i]=np.nan

    if verbose:
        print(f"  Note: μ SEs not identified from pre-care moments (see bootstrap).")
    return {"se_s2c":se_s2c,"se_lam":se_lam,"se_mu":se_mu,
            "se_level":np.concatenate([se_s2c,se_lam,se_mu])}

# ── 4. METHOD 2: Hessian diagonal ─────────────────────────

def hessian_diag_se(theta_hat, n_eff=13, h=5e-4, verbose=True)->dict:
    """
    Diagonal of numerical Hessian H = ∂²J/∂θᵢ².
    SE_i = 1 / sqrt(H_ii × n_eff)  (marginal curvature approximation).
    Appropriate when off-diagonal terms are small (block-diagonal structure).
    """
    if verbose: print("\n── Method 2: Hessian diagonal ──")
    t0=time.time()
    f0=J(theta_hat); diag_H=np.zeros(N_PAR)
    for i in range(N_PAR):
        hi=max(abs(theta_hat[i])*h,h)
        tp=theta_hat.copy(); tp[i]+=hi
        tm=theta_hat.copy(); tm[i]-=hi
        d2=(J(tp)-2*f0+J(tm))/hi**2
        diag_H[i]=max(d2,1e-10)   # clip negatives (flat objective)

    se_log=1.0/np.sqrt(diag_H*n_eff+1e-10)
    se_lev=np.exp(theta_hat)*se_log
    if verbose:
        print(f"  Done [{time.time()-t0:.1f}s]")
        neg=(J(theta_hat+1e-4*np.random.randn(N_PAR))<J(theta_hat))
        flat=(diag_H<1e-6).sum()
        if flat>0: print(f"  {flat} parameters at flat region (SE inflated)")
    return {"diag_H":diag_H,"se_log":se_log,"se_level":se_lev}

# ── 5. METHOD 3: Parametric bootstrap ──────────────────────

def param_bootstrap(theta_hat, B=200, n_eff=13, verbose=True)->dict:
    """
    Pakes-Pollard (1989) parametric bootstrap.

    Algorithm:
      For b = 1..B:
        (a) ε_b ~ N(0, Σ_emp) / √n_eff    — moment sampling noise
        (b) emp_b = EMP + ε_b.reshape(6,4) — perturbed targets
        (c) θ_b = argmin J(θ; emp_b)       — re-estimate
      SE = std({exp(θ_b)})

    Σ_emp: diagonal, 15% of each moment magnitude
    Interpretation: SE reflects 'how much would θ̂ change
    if the moment targets shifted by a typical calibration
    error', scaled by 1/√n_eff.
    """
    if verbose:
        print(f"\n── Method 3: Parametric bootstrap  (B={B}) ──")
        t0=time.time()
    emp_flat=EMP.flatten()
    sig_d=(0.15*np.maximum(np.abs(emp_flat),0.05))**2
    L=np.diag(np.sqrt(sig_d))

    boot=np.zeros((B,N_PAR)); failed=0
    for b in range(B):
        eps=(L@RNG.standard_normal(N_DIM*N_MOM))/np.sqrt(n_eff)
        emp_b=np.clip((emp_flat+eps).reshape(N_DIM,N_MOM),0.01,2.0)
        def Jb(th,_e=emp_b):
            try: g=(sim_moments(th)-_e).flatten(); return float(g@W@g)
            except: return 1e6
        rb=opt.minimize(Jb,theta_hat,method="Nelder-Mead",
                        options={"maxiter":350,"xatol":1e-4,"fatol":1e-5,"disp":False})
        boot[b]=rb.x if rb.fun<20 else theta_hat
        if rb.fun>=20: failed+=1
        if verbose and (b+1)%50==0:
            el=time.time()-t0; eta=el/(b+1)*(B-b-1)
            print(f"  {b+1}/{B}  [{el:.0f}s, ETA {eta:.0f}s, failed {failed}]")

    bl=np.exp(boot)
    se_log=boot.std(0); se_lev=bl.std(0)
    ci90=np.percentile(bl,[5,95],axis=0).T
    if verbose:
        print(f"  Done. failed={failed}/{B}  total {time.time()-t0:.0f}s")
    return {"boot":boot,"boot_lev":bl,"se_log":se_log,
            "se_level":se_lev,"ci90":ci90}

# ── 6. Ordering test ───────────────────────────────────────

def ordering_test(boot_res, theta_hat):
    print("\n"+"="*62)
    print("ORDERING TEST  (Assumption A2 — Proposition 1 foundation)")
    print("="*62)
    bl=boot_res["boot_lev"]
    pairs=[("σ²_C(e₆)>σ²_C(e₃)",5,2),("σ²_C(e₆)>σ²_C(e₄)",5,3),
           ("σ²_C(e₅)>σ²_C(e₃)",4,2),("σ²_C(e₅)>σ²_C(e₄)",4,3),
           ("σ²_C(e₅)>σ²_C(e₁)",4,0),("σ²_C(e₆)>σ²_C(e₁)",5,0)]
    all_ok=True
    for label,hi,lo in pairs:
        prob=(bl[:,hi]>bl[:,lo]).mean()
        ratio=np.exp(theta_hat[hi])/np.exp(theta_hat[lo])
        ok=prob>=0.90; all_ok=all_ok and ok
        print(f"  {'✓' if ok else '✗'} P({label}) = {prob:.3f}   "
              f"[point ratio: {ratio:.1f}×]")
    print(f"\n  Conclusion: A2 ordering is {'robust' if all_ok else 'FRAGILE'} "
          f"across bootstrap draws.")
    print("  Note: exact ratios (e.g. 3000×) should not be over-interpreted;")
    print("  the ordinal ranking is the key inferential object.")

# ── 7. Results ─────────────────────────────────────────────

PNAMES=([f"σ²_C({d.name})" for d in DIMS]+
        [f"λ({d.name})"    for d in DIMS]+
        [f"μ({d.name})"    for d in DIMS])

def results_table(theta_hat,delta,hess,boot)->pd.DataFrame:
    lev=np.exp(theta_hat)
    rows=[]
    for j in range(N_PAR):
        se_d=delta["se_level"][j]
        rows.append({
            "Parameter": PNAMES[j],
            "Estimate":  lev[j],
            "SE_delta":  se_d,
            "SE_hessian":hess["se_level"][j],
            "SE_boot":   boot["se_level"][j],
            "CI90_lo":   boot["ci90"][j,0] if j<N_PAR else np.nan,
            "CI90_hi":   boot["ci90"][j,1] if j<N_PAR else np.nan,
        })
    return pd.DataFrame(rows)

def print_table(df):
    print("\n"+"="*72)
    print("SMM ESTIMATES AND STANDARD ERRORS")
    print(f"{'':>16}{'Estimate':>10}{'SE(delta)':>11}"
          f"{'SE(hess)':>11}{'SE(boot)':>11}  {'90% CI (boot)':>20}")
    slices=[("σ²_C",slice(0,6)),("λ",slice(6,12)),("μ",slice(12,18))]
    for glabel,gs in slices:
        print(f"\n  ── {glabel} ──")
        for _,r in df.iloc[gs].iterrows():
            dim=r["Parameter"].split("(")[1].rstrip(")")
            sed=f"{r['SE_delta']:>10.3f}" if not np.isnan(r['SE_delta']) else f"{'n/a':>10}"
            print(f"  {dim:<14}{r['Estimate']:>10.3f}{sed}"
                  f"{r['SE_hessian']:>11.3f}{r['SE_boot']:>11.3f}"
                  f"  [{r['CI90_lo']:>6.3f},{r['CI90_hi']:>6.3f}]")
    print("\n  Note: SE(delta) for μ = n/a (μ identified from admin floor, not pre-care moment).")
    print("  SE(boot) is the primary uncertainty measure; SE(delta/hess) are diagnostics.")

# ── 8. Figures ─────────────────────────────────────────────

def plot_main(df, theta_hat, boot, save=True):
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("SMM Parameter Estimates — Standard Errors",
                 fontsize=13,fontweight="bold")

    # Panel A: σ²_C with 90% CI
    ax=axes[0]
    est=np.exp(theta_hat[:6])
    lo=boot["ci90"][:6,0]; hi=boot["ci90"][:6,1]
    names=[d.name for d in DIMS]
    ax.errorbar(range(6),est,yerr=[est-lo,hi-est],fmt="o",
                color=BLUE,capsize=5,label="Estimate ± 90% CI (boot)")
    ax.set_xticks(range(6)); ax.set_xticklabels(names,rotation=30,ha="right")
    ax.set_title("σ²_C — court signal variance\n(bootstrap 90% CI)")
    ax.set_ylabel("σ²_C"); ax.legend(fontsize=8)
    # Add ordering annotation
    ax.annotate("A2: e₅,e₆ >> e₃",xy=(4,est[4]),xytext=(2.5,max(est)*0.8),
                fontsize=8,color=BLUE,
                arrowprops=dict(arrowstyle="->",color=BLUE,lw=0.8))

    # Panel B: SE comparison across methods
    ax=axes[1]
    x=np.arange(N_PAR)
    ax.scatter(x, df["SE_delta"].fillna(0),  s=22,color=BLUE, alpha=0.8,label="Delta method")
    ax.scatter(x, df["SE_hessian"],           s=22,color=RED,  alpha=0.8,marker="s",label="Hessian")
    ax.scatter(x, df["SE_boot"],              s=22,color=GREEN,alpha=0.8,marker="^",label="Bootstrap")
    for v in [5.5,11.5]: ax.axvline(v,color="gray",lw=0.8,ls="--")
    ax.set_xticks([2.5,8.5,14.5]); ax.set_xticklabels(["σ²_C","λ","μ"],fontsize=10)
    ax.set_title("SE comparison: all 18 parameters"); ax.set_ylabel("SE (level)")
    ax.set_yscale("log"); ax.legend(fontsize=8)

    # Panel C: Bootstrap dist σ²_C ordering
    ax=axes[2]
    bl=boot["boot_lev"]
    ax.hist(bl[:,5],bins=30,color=BLUE,alpha=0.6,density=True,label="σ²_C(e₆)")
    ax.hist(bl[:,4],bins=30,color=PURPLE if "PURPLE" in dir() else AMBER,
            alpha=0.6,density=True,label="σ²_C(e₅)")
    ax.hist(bl[:,2],bins=30,color=RED,  alpha=0.6,density=True,label="σ²_C(e₃)")
    for i,col in [(5,BLUE),(4,AMBER),(2,RED)]:
        ax.axvline(np.exp(theta_hat[i]),color=col,lw=2,ls="--")
    ax.set_title("Bootstrap: σ²_C ordering\n(Assumption A2 robustness)")
    ax.set_xlabel("σ²_C"); ax.set_ylabel("Density"); ax.legend(fontsize=8)

    plt.tight_layout()
    if save: plt.savefig("smm_se_main.png",bbox_inches="tight",dpi=130)
    plt.show()

def plot_boot_grid(boot, theta_hat, save=True):
    bl=boot["boot_lev"]; ci90=boot["ci90"]; lev=np.exp(theta_hat)
    groups=[("σ²_C",bl[:,:6],lev[:6],ci90[:6],BLUE),
            ("λ",   bl[:,6:12],lev[6:12],ci90[6:12],GREEN),
            ("μ",   bl[:,12:],lev[12:],ci90[12:],AMBER)]
    names=[d.name for d in DIMS]
    fig,axes=plt.subplots(3,6,figsize=(18,9),sharey="row")
    fig.suptitle("Bootstrap Distributions — All 18 SMM Parameters",
                 fontsize=13,fontweight="bold")
    for row,(pname,samp,pts,cis,col) in enumerate(groups):
        for j in range(6):
            ax=axes[row,j]
            ax.hist(samp[:,j],bins=30,color=col,alpha=0.75,density=True)
            ax.axvline(pts[j],color="black",lw=1.5)
            ax.axvline(cis[j,0],color="gray",lw=1,ls="--")
            ax.axvline(cis[j,1],color="gray",lw=1,ls="--")
            if row==0: ax.set_title(names[j],fontsize=9)
            if j==0:   ax.set_ylabel(pname,fontsize=9)
            ax.tick_params(labelsize=7)
    plt.tight_layout()
    if save: plt.savefig("smm_se_boot_grid.png",bbox_inches="tight",dpi=130)
    plt.show()

# ── 9. Main ────────────────────────────────────────────────

def run_all(B=200, n_eff=13, save=True):
    print("="*62)
    print("SMM STANDARD ERROR ESTIMATION")
    print(f"n_eff={n_eff}  |  bootstrap B={B}")
    print("─"*62)
    print("Identification note:")
    print("  Only pre-enforcement care moments are informative (6 moments")
    print("  for 18 params). SEs reflect calibration uncertainty, not")
    print("  sampling precision from a large micro-data panel.")
    print("="*62)

    theta_hat = first_stage(verbose=True)
    delta = delta_se(theta_hat, n_eff=n_eff, verbose=True)
    hess  = hessian_diag_se(theta_hat, n_eff=n_eff, verbose=True)
    boot  = param_bootstrap(theta_hat, B=B, n_eff=n_eff, verbose=True)

    df = results_table(theta_hat, delta, hess, boot)
    print_table(df)
    ordering_test(boot, theta_hat)

    plot_main(df, theta_hat, boot, save=save)
    plot_boot_grid(boot, theta_hat, save=save)

    df.to_csv("smm_se_results.csv",index=False,float_format="%.4f")
    print("\nSaved: smm_se_results.csv")
    if save: print("Saved: smm_se_main.png, smm_se_boot_grid.png")
    return theta_hat, df, delta, hess, boot

PURPLE="#7F77DD"  # needed for panel C

if __name__ == "__main__":
    theta_hat, df, delta, hess, boot = run_all(B=200, n_eff=13)
