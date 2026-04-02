# Fiduciary Duty of Care — Dual-Layer Enforcement Simulation

Replication code for:

> Arai, K. (2025). "Fiduciary Duty of Care in Trust Administration: A Dual-Layer Framework from Administrative Enforcement and Private Law." *Working paper*, Kyoritsu Women's University.
>
> JSPS KAKENHI Grant Number 23K01404

---

## Overview

This repository contains the Python simulation suite used to verify the paper's three propositions and generate all quantitative results. The simulation combines:

- **Equilibrium analysis** — trustee care-level solver across four enforcement regimes
- **Structural parameter estimation** — SMM-style calibration and Bayesian MCMC
- **Causal identification** — DiD, RDD, and synthetic control specification checks
- **Agent-based market dynamics** — bifurcation and low-care trap analysis
- **Policy counterfactuals** — six Japan-specific reform scenarios (Module 8)
- **Standard error estimation** — parametric bootstrap for SMM parameters and policy welfare effects
- **Validation suite** — ten checks covering all three propositions

### Three propositions verified

| Proposition | Content | Key result |
|---|---|---|
| **Prop 1** | Civil-only enforcement under-provides governance and disclosure care | e₅ gap = 0.145, e₆ gap = 0.079 under civil-only |
| **Prop 2** | Dual-layer dominates both single-channel alternatives | W(dual) = 0.748 >> W(civil only) = −2.696 |
| **Prop 3** | PIR raises welfare; BJR extension reduces it | W(PIR) > W(base) > W(BJR) across all σ²_C values |

---

## Repository structure

```
fiduciary_repo/
├── src/
│   ├── simulation.py         # Modules 1–7: core simulation suite
│   ├── policy_simulation.py  # Module 8: Japan policy experiments (8A–8F)
│   ├── smm_se.py             # SMM standard error estimation (3 methods)
│   ├── policy_se.py          # Policy welfare SE via parametric bootstrap
│   ├── policy_sensitivity.py # Policy design sensitivity analysis
│   └── validation.py         # Validation suite (V1–V10)
├── results/                  # Output figures and CSV (generated at runtime)
├── docs/
│   └── parameter_table.md    # Parameter calibration protocol (Table F.0)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick start

### Google Colab (recommended)

Upload all files in `src/` to your Colab session, then:

```python
# Step 1: Install dependencies (run once)
!pip install numpy scipy matplotlib pandas

# Step 2: Run the core simulation (Modules 1–7)
exec(open("simulation.py").read())
run_module1()   # Equilibrium analysis → Fig 4.1
run_module2()   # Bayesian MCMC calibration
run_module3()   # SMM estimation → Table 4.1 (preliminary)

# Step 3: Run policy simulation (Module 8)
exec(open("policy_simulation.py").read())
run_module8()   # All sub-modules (8A–8F) → Figs 4.3, 4.4

# Step 4: Bootstrap standard errors for SMM parameters
exec(open("smm_se.py").read())
theta_hat, df, delta, hess, boot = run_all(B=200, n_eff=13)
# → Table 4.1 (final), smm_se_main.png, smm_se_boot_grid.png

# Step 5: Bootstrap SE for policy welfare effects
exec(open("policy_se.py").read())
res = run_all(B=200, n_eff=13)
# → Table 4.4, policy_se_8E.png, policy_se_8D.png

# Step 6: Policy design sensitivity analysis
exec(open("policy_sensitivity.py").read())
run_sensitivity()
# → policy_sensitivity_oneway.png, policy_sensitivity_twoway.png

# Step 7: Validation suite
exec(open("validation.py").read())
validate_fast()           # ~15 seconds, V7 skipped
validate_full(B=50)       # ~10 minutes, all 10 checks
```

### Local environment

```bash
pip install -r requirements.txt
python src/smm_se.py        # SMM standard errors (standalone)
python src/validation.py    # Fast validation
```

---

## File descriptions

### `src/simulation.py` — Core simulation (Modules 1–7)

| Module | Function | Content |
|---|---|---|
| 1 | `run_module1()` | Equilibrium care solver × 4 regimes; Propositions 1–3 |
| 2 | `run_module2()` | Bayesian MCMC posterior for σ²_C, λ, μ |
| 3 | `run_module3()` | SMM estimation; σ²_C ordering; Assumption A2 |
| 4A–4C | `run_module4*()` | DiD, RDD, synthetic control specification checks |
| 5 | `run_module5()` | Agent-based model; bifurcation diagram; low-care trap |
| 6 | `run_module6()` | Morris sensitivity; welfare surface |
| 7 | `run_module7()` | Integrated policy counterfactuals |

**Key parameters** (in `DIMS_DEFAULT`):

| Dim | σ²_C | λ | μ | ē^R | Interpretation |
|---|---|---|---|---|---|
| e₁ DD | 0.20 | 1.20 | 0.80 | 0.55 | Pre-acceptance due diligence |
| e₂ Monitor | 0.30 | 1.00 | 1.20 | 0.50 | Post-acceptance monitoring |
| e₃ Segreg | 0.08 | 1.80 | 1.50 | 0.80 | Asset segregation |
| e₄ Conflict | 0.12 | 1.50 | 1.20 | 0.70 | Conflict-of-interest management |
| e₅ Disclos | 0.60 | 0.50 | 1.60 | 0.58 | Disclosure / information provision |
| e₆ Govern | 0.70 | 0.35 | 1.80 | 0.52 | Organisational governance |

### `src/policy_simulation.py` — Module 8

| Sub-module | Function | Content |
|---|---|---|
| 8A | `regulatory_design_optimisation()` | Optimal (ē^R, μ) via SLSQP |
| 8B | `dynamic_regulation()` | Bayesian learning dynamics |
| 8C | `transition_dynamics()` | Escalation schedule comparison |
| 8D | `compare_architectures()` | Six institutional designs × W_net |
| 8E | `japan_policy_experiments()` | JP1–JP5 counterfactuals |
| 8F | `welfare_decomposition()` | W_B, W_T_H, W_T_L decomposition |

**Japan policy experiments (JP1–JP5):**

| Code | Policy | ΔW | Direction |
|---|---|---|---|
| JP1 | Prudent Investor Rule adoption | +0.287 [+0.164, +0.371] | positive |
| JP2 | Business Judgment Rule extension | −0.300 [−0.138, −0.488] | negative |
| JP3 | Role-allocation reform | +0.189 [+0.102, +0.261] | positive |
| JP4 | JDC Trust early escalation | +0.076 [+0.042, +0.101] | positive |
| JP5 | Comprehensive reform | +0.620 [+0.379, +0.738] | positive |

### `src/smm_se.py` — SMM standard errors

Three SE estimation methods:

| Method | Description |
|---|---|
| Delta method | Dimension-wise, block-diagonal Jacobian |
| Hessian diagonal | Numerical second derivative of J(θ) |
| Parametric bootstrap | Pakes–Pollard (1989), B=200 |

**Key identification note:** Only the six pre-enforcement care moments are informative (post-enforcement care is pinned to the admin floor). SEs reflect calibration uncertainty given the moment targets, not sampling precision from a large panel.

**SMM estimates (bootstrap 90% CI, B=200):**

| Dim | σ²_C | 90% CI | λ | 90% CI | μ | 90% CI |
|---|---|---|---|---|---|---|
| e₁ DD | 0.323 | [0.262, 0.384] | 1.199 | [1.151, 1.282] | 0.873 | [0.850, 0.889] |
| e₂ Monitor | 0.200 | [0.172, 0.350] | 1.003 | [1.003, 1.004] | 1.152 | [1.115, 1.181] |
| e₃ Segreg | 0.082 | [0.070, 0.094] | 1.734 | [1.534, 1.879] | 1.381 | [1.308, 1.448] |
| e₄ Conflict | 0.141 | [0.122, 0.163] | 1.380 | [1.314, 1.451] | 1.239 | [1.208, 1.283] |
| e₅ Disclos | 0.516 | [0.473, 0.596] | 0.503 | [0.403, 0.598] | 1.852 | [1.655, 1.967] |
| e₆ Govern | 0.960 | [0.955, 0.966] | 0.453 | [0.295, 0.634] | 2.275 | [1.962, 2.500] |

Ordering test: P(σ²_C(e₆) > σ²_C(e₃)) = **1.000** across all 200 bootstrap draws.

### `src/policy_se.py` — Policy welfare SE

Propagates SMM parameter uncertainty (θ) into Module 8D/8E/8F welfare outcomes via parametric bootstrap. Key result: **SE(θ) = 0** for all Module 8E policies — a structural property (not a bug): when the admin floor binds, ΔW depends only on policy design parameters (Δē^R, Δe*), not on (σ²_C, λ, μ).

### `src/policy_sensitivity.py` — Design sensitivity

One-way and two-way sensitivity of Module 8E welfare effects over policy shift magnitude Δ ∈ [0.5×, 1.5×] baseline. All five policy directions are robust across this range (breakeven = 0 for JP1, JP3, JP4, JP5; negative for all tested JP2 magnitudes).

### `src/validation.py` — Validation suite

| Check | Description | Result |
|---|---|---|
| V1 | Prop 1: civil-only under-provides e₅,e₆ | PASS (gap=0.149, 0.080) |
| V2 | Prop 2: W(dual) > W(civil-only) | PASS (margin=+3.40) |
| V3 | Prop 3: W(PIR) > W(base) > W(BJR) | PASS |
| V4 | Assumption A2: σ²_C ordering | PASS (4/4 pairs) |
| V5 | Prop 2b: dual ≥ civil for e₃,e₄ welfare | PASS |
| V6 | Policy signs: JP1,3,4,5 pos; JP2 neg | PASS |
| V7 | Bootstrap: P(A2 ordering) ≥ 0.90 | PASS (P=1.000, B=200) |
| V8 | Sensitivity: prop signs under ±25% noise | PASS (100% / 15 draws) |
| V9 | Admin floor binding for e₅,e₆ | PASS |
| V10 | Prop 1 strong: gov dims gap > segreg dims | PASS |

---

## Computational notes

### Runtime (Google Colab, CPU)

| Step | Function | Approx. time |
|---|---|---|
| Modules 1–7 | `run_module1()` … `run_module7()` | ~5 min total |
| Module 8 | `run_module8()` | ~10 min |
| SMM SE (B=200) | `smm_se.run_all(B=200)` | ~35 min |
| Policy SE (B=200) | `policy_se.run_all(B=200)` | ~35 min |
| Validation full | `validate_full(B=50)` | ~10 min |

For faster runs during development, use B=20–50 for bootstrap steps.

### Reproducibility

All random number generators are seeded (`np.random.default_rng(seed=42)` or `seed=2025`). Re-running the same script produces identical numerical output.

---

## Data

The model is calibrated to a **census of 13 enforcement events** (n_eff=13):
- 7 FSA administrative enforcement actions (2002–2018): Shinsei Trust (2006), JP Morgan Trust (2006), State Street Trust (2006), SocGen Trust (2008), Rites Trust (2010–2016), AIJ Investment (2012), JDC Trust (2016–2018)
- 6 judicial decisions (2002–2018) involving trustee duty-of-care claims

No micro-level data files are required; all empirical moments are hard-coded in `smm_se.py` (the `EMP` array). The full calibration protocol mapping each enforcement event to parameter values is in `docs/parameter_table.md`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{arai2025fiduciary,
  author  = {Arai, Koki},
  title   = {Fiduciary Duty of Care in Trust Administration:
             A Dual-Layer Framework from Administrative Enforcement
             and Private Law},
  year    = {2025},
  note    = {Working paper, Kyoritsu Women's University.
             JSPS KAKENHI 23K01404}
}
```

---

## License

MIT License. See `LICENSE` for details.
