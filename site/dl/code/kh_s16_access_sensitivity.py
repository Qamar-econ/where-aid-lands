"""How much does the road-access result depend on how the index is built?
Same outcome (activities 2017-2020), same PPML with province FE and ln population, same inference as kh_s04
(randomisation inference is skipped for the variants; p_max = larger of clustered and Conley 50/100 km p-values).

  Weighting      PC1 (main) | equal weights | 7 components without the sparse surface/wet/steep tags | travel times only
                 | one measure (time to town >= 20k) | leave one component out (10 variants)
  Speeds         unpaved multiplier 0.5 / 0.9 (main 0.7) | minor-road speeds x0.8 / x1.2  (kh_s03 with KH_UNPAVED, KH_MINOR_SCALE)
  Network date   OSM 1 Jan 2017 (main) | OSM 2026
Output: outputs/tables/kh_access_sensitivity.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])          # DF, A, outcome(), ppml(), conley_se() as in the main script
PROCD = ROOT / "data/processed/cambodia"


def components(a):
    c = pd.DataFrame({"neg_ln_tt_town_dry": -np.log(a.tt_town_dry), "neg_ln_tt_town_wet": -np.log(a.tt_town_wet), "neg_ln_tt_hq": -np.log(a.tt_hq_dry),
                      "neg_ln_tt_hosp": -np.log(a.tt_hosp_dry), "neg_ln_walk": -np.log(a.walk_min), "share_walk30": a.share_walk30,
                      "ln_road_density": np.log(a.road_density), "paved_share": a.paved_share, "neg_wet_penalty": -a.wet_penalty_share,
                      "neg_steep_share": -a.steep_share}, index=a.index).replace([np.inf, -np.inf], np.nan)
    c = c.fillna(c.median())
    return (c - c.mean()) / c.std()


def pc1(Z):
    w, V = np.linalg.eigh(np.cov(Z.T)); v = V[:, -1] * np.sign(V[:, -1].sum()); return pd.Series(Z.values @ v, index=Z.index)


zs = lambda s: (s - s.mean()) / s.std()
acc = pd.read_csv(PROCD / "kh_access.csv").set_index("k")
Z = components(acc)
TT = ["neg_ln_tt_town_dry", "neg_ln_tt_town_wet", "neg_ln_tt_hq", "neg_ln_tt_hosp", "neg_ln_walk"]
V = {"main: PC1 of 10 components (2017 OSM)": pc1(Z),
     "equal weights, 10 components": Z.mean(axis=1),
     "PC1 of 7 components (no paved/wet/steep)": pc1(Z.drop(columns=["paved_share", "neg_wet_penalty", "neg_steep_share"])),
     "travel times only (PC1 of 5)": pc1(Z[TT]),
     "one measure: -ln time to town >= 20k": Z["neg_ln_tt_town_dry"]}
for c in Z.columns:
    V[f"leave out: {c}"] = pc1(Z.drop(columns=c))
for tag, lab in [("unp05", "unpaved speed x0.5 (main 0.7)"), ("unp09", "unpaved speed x0.9"), ("minor08", "minor-road speeds x0.8"),
                 ("minor12", "minor-road speeds x1.2"), ("latest", "network: OSM 2026")]:
    V[lab] = pd.read_csv(PROCD / f"kh_access_{tag}.csv").set_index("k").access_pc1

d0 = outcome(A[A.level == "adm2"])
rows = []
for lab, s in V.items():
    d = d0.assign(xv=zs(d0.unit.map(s)))
    r = ppml("N ~ xv + lnpop + C(prov)", d); b = r.params["xv"]
    ps = [r.pvalues["xv"]] + [2 * norm.sf(abs(b / conley_se(r, d, "xv", c))) for c in (50, 100)]
    rows.append({"variant": lab, "pct": 100 * (np.exp(b) - 1), "p_cluster": ps[0], "p_max": max(ps),
                 "corr_with_main": d.xv.corr(zs(d0.unit.map(V["main: PC1 of 10 components (2017 OSM)"])))})
R = pd.DataFrame(rows)
R.to_csv(T / "kh_access_sensitivity.csv", index=False)
print(R.round(3).to_string(index=False))
lo = R[R.variant.str.startswith("leave out")]
print(f"\nleave-one-out range: {lo.pct.min():.0f}% to {lo.pct.max():.0f}% | all variants: {R.pct.min():.0f}% to {R.pct.max():.0f}%, "
      f"largest p_max {R.p_max.max():.3f}")

# input-audit corrections (kh_s19 findings), kept in their own table so the 20-variant grid above is unchanged
rows = []
for tag, lab in [("towncap", "towns = 25 provincial capitals"), ("secpaved", "untagged secondary roads paved"),
                 ("grade185", "grade over >= 185 m"), ("allfix", "all three corrections")]:
    s = pd.read_csv(PROCD / f"kh_access_{tag}.csv").set_index("k").access_pc1
    d = d0.assign(xv=zs(d0.unit.map(s)))
    r = ppml("N ~ xv + lnpop + C(prov)", d); b = r.params["xv"]
    ps = [r.pvalues["xv"]] + [2 * norm.sf(abs(b / conley_se(r, d, "xv", c))) for c in (50, 100)]
    rows.append({"variant": lab, "pct": 100 * (np.exp(b) - 1), "p_cluster": ps[0], "p_max": max(ps),
                 "corr_with_main": d.xv.corr(zs(d0.unit.map(V["main: PC1 of 10 components (2017 OSM)"])))})
RA = pd.DataFrame(rows); RA.to_csv(T / "kh_access_audit_variants.csv", index=False)
print("\ninput-audit corrections:\n" + RA.round(3).to_string(index=False))
