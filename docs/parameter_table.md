# Parameter Calibration Protocol (Table F.0)

This file documents the mapping from enforcement events to model parameter values,
following the calibration protocol described in Appendix F of the paper.

## Enforcement Record

### Administrative enforcement actions (7 events)

| Event | Year | Dimension | Parameter affected | Coding rule |
|---|---|---|---|---|
| Shinsei Trust | 2006 | e₁ DD, e₂ Monitor | e_bar ↑ for e₁,e₂; μ(e₁)=0.80 | Business improvement order; KYC/screening deficiency |
| JP Morgan Trust | 2006 | e₁ DD | e_bar(e₁); μ(e₁)=0.80 | Same order; investment due diligence failure |
| State Street Trust | 2006 | e₆ Govern | μ(e₆)=1.80 (high); σ²_C(e₆) large | Internal control architecture violation |
| SocGen Trust | 2008 | e₆ Govern, e₅ Disclos | μ(e₅)=1.60, μ(e₆)=1.80 | Governance + disclosure deficiency |
| Rites Trust | 2010–16 | e₆ Govern | μ(e₆) starts low (0.3) → escalates | Multiple iterative improvement orders; identifies low-care trap at μ<0.3 |
| AIJ Investment | 2012 | e₁ DD, e₂ Monitor | Role-allocation effect: λ(e₁,e₂) ↓ | Fund advisory scope dispute; civil channel weakened |
| JDC Trust | 2016–18 | e₆ Govern | μ(e₆)=2.275 (highest); license revocation | Most severe governance failure in record |

### Judicial decisions (6 events)

| Case | Year | Dimension | Parameter affected | Coding rule |
|---|---|---|---|---|
| Osaka DC | 2013 | e₁,e₂ (investment) | λ(e₁)=1.199, λ(e₂)=1.003 (low) | BJR-adjacent deferential review |
| Tokyo HC | 2018 | e₁,e₂ | Confirms λ low for investment dims | PIR vs BJR threshold case |
| Asset segregation cases (×2) | 2002–2010 | e₃ Segreg | λ(e₃)=1.734 (highest) | Traceable loss; strict liability pattern |
| Conflict-of-interest cases (×2) | 2008–2015 | e₄ Conflict | λ(e₄)=1.380 | Identifiable breach pattern |

## Coding rules

### σ²_C (court signal variance)
- **High σ²_C** (e₅, e₆): no judicial liability imposed in any case in the dataset for pure governance or disclosure failures; courts cannot observe process quality → σ²_C set large (0.516, 0.960)
- **Low σ²_C** (e₃, e₄): breach identifiable from financial records; courts impose liability consistently → σ²_C set small (0.082, 0.141)
- **Medium σ²_C** (e₁, e₂): mixed record; investment judgment scope disputed → σ²_C intermediate (0.200–0.323)

### λ (civil liability weight)
- Ordered inversely with σ²_C, consistent with Proposition 2 (dual-layer complementarity)
- e₃ has highest λ (1.734): strict liability in segregation cases
- e₆ has lowest λ (0.453): no civil channel for governance architecture

### μ (administrative penalty)
- Ordered with governance focus: e₆ highest (2.275), reflecting business improvement order severity and license revocations
- e₁ lowest (0.873): lighter administrative response to DD failures relative to governance

### ē^R (administrative floor)
- e₃, e₄ have high floors (0.80, 0.70): TBA Art. 34 segregation and conflict rules are specific
- e₅, e₆ have lower floors (0.58, 0.52): FSA supervisory guidelines less prescriptive for process dimensions

## Sensitivity ranges

All parameters are varied ±25% in the V8 sensitivity check (validation.py).
The parameter calibration is robust: Propositions 1–3 hold in 100% of 15 draws
under ±25% perturbation, and in 100% of 20 draws under ±30% perturbation.

## References

- FSA Business Improvement Orders (金融庁業務改善命令): publicly available via FSA website
- Judicial decisions: sourced from Westlaw Japan and LEX/DB
- Full citations in paper Section 2 and References section
