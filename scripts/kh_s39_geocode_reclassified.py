"""Re-estimates the headline single-factor models after hand-classifying town- and province-name geocodes (spec 9-26).

Classes (kh_geocode_classified.csv, from titles and descriptions only):
  TOWN      kept where geocoded
  PROVINCE  weight at the flagged district spread over the province's districts by population (Phnom Penh: its khans)
  UNCLEAR   main: as PROVINCE
  NATIONAL  (added while classifying) country-wide programmes geocoded to the embassy or headquarters city: main = dropped
Main estimates get the most conservative of clustered, Conley 50/100 km and Freedman-Lane randomisation inference
(999 draws); sensitivity scenarios report clustered p.
Output: outputs/tables/kh_geocode_reclassified.csv, .txt
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

S24 = Path(__file__).with_name("kh_s24_conservative_p.py")
if not S24.exists():
    S24 = next(Path(__file__).parent.glob("kh_s24*.py"))
ns = {"__file__": str(S24)}
exec(S24.read_text(encoding="utf-8").split("rows = []")[0], ns)
DF, cur, outcome, prep, conley_p, ri_p, ROOT = (ns[k] for k in ["DF", "cur", "outcome", "prep", "conley_p", "ri_p", "ROOT"])
PROCD, T = ROOT / "data/processed/cambodia", ROOT / "outputs/tables"
CL = pd.read_csv(PROCD / "kh_geocode_classified.csv")[["project_id", "class"]]
FU = pd.read_csv(PROCD / "kh_geocode_flagged_units.csv").merge(CL, on="project_id")
D = DF.set_index("unit"); CAP = set(DF[DF.capital].unit)
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))


def realloc(c, move=("PROVINCE", "UNCLEAR"), drop=("NATIONAL",), how="pop"):
    key = FU.set_index(["project_id", "unit"])["class"]
    cls = pd.Series([key.get((p, u)) for p, u in zip(c.project_id, c.unit)], index=c.index)
    keep = c[~cls.isin(list(move) + list(drop))]
    mv = c[cls.isin(move)]
    out = [keep]
    for r in mv.itertuples():
        pool = DF[DF.unit.isin(CAP)] if r.unit in CAP else DF[DF.prov == D.loc[r.unit, "prov"]]
        sh = (pool.pop_2019 / pool.pop_2019.sum()).values if how == "pop" else \
             (pool.access == pool.access.min()).astype(float).values if how == "worst" else np.full(len(pool), 1 / len(pool))
        out.append(pd.DataFrame({"project_id": r.project_id, "unit": pool.unit.values, "w": r.w * sh, "comm": 0.0, "family": r.family}))
    return pd.concat(out, ignore_index=True)


VARS = [("access", "physical access"), ("need", "poverty"), ("lights", "night lights"), ("conn", "mobile speed")]


def fit(c, full=False):
    d = prep(outcome(c)); res = {}
    for v, lab in VARS:
        dd = d.dropna(subset=[v]).reset_index(drop=True); f = f"N ~ {v} + lnpop + C(prov)"
        r = smf.glm(f, data=dd, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]}, maxiter=300)
        b, s = r.params[v], r.bse[v]; p = {"p_cluster": r.pvalues[v]}
        if full:
            p.update(p_conley50=conley_p(r, dd, v, 50), p_conley100=conley_p(r, dd, v, 100), p_ri=ri_p(dd, v, [], f, b))
        res[v] = dict(pct=100 * (np.exp(b) - 1), lo=100 * (np.exp(b - 1.96 * s) - 1), hi=100 * (np.exp(b + 1.96 * s) - 1), **p,
                      p_conservative=max(p.values()), projects=dd.N.sum())
    return res


rows = []
say("classes:", CL["class"].value_counts().to_dict(), "| flagged district-activity pairs:", len(FU))
for lab, c, full in [("original (as published)", cur, True),
                     ("MAIN: PROVINCE+UNCLEAR spread by population, NATIONAL dropped", realloc(cur), True),
                     ("UNCLEAR kept as town", realloc(cur, move=("PROVINCE",)), False),
                     ("UNCLEAR dropped", realloc(cur, move=("PROVINCE",), drop=("NATIONAL", "UNCLEAR")), False),
                     ("NATIONAL kept where geocoded", realloc(cur, drop=()), False),
                     ("NATIONAL spread over Phnom Penh khans as PROVINCE", realloc(cur, move=("PROVINCE", "UNCLEAR", "NATIONAL"), drop=()), False),
                     ("spread equally within province", realloc(cur, how="equal"), False),
                     ("worst case: PROVINCE+UNCLEAR to least accessible district", realloc(cur, how="worst"), False)]:
    R = fit(c, full)
    say(f"== {lab}")
    for v, nm in VARS:
        r = R[v]; rows.append({"scenario": lab, "var": v, **r})
        say(f"   {nm:<16s} {r['pct']:+5.0f}% (95% CI {r['lo']:+.0f}% to {r['hi']:+.0f}%) | p cluster {r['p_cluster']:.3f}" +
            (f" | Conley50 {r['p_conley50']:.3f} Conley100 {r['p_conley100']:.3f} RI {r['p_ri']:.3f} | most conservative {r['p_conservative']:.3f}" if full else "") +
            f" | projects {r['projects']:.1f}")
pd.DataFrame(rows).to_csv(T / "kh_geocode_reclassified.csv", index=False)
(T / "kh_geocode_reclassified.txt").write_text("\n".join(log), encoding="utf-8")
