"""Does infrastructure aid pull in later aid? District-year panel, 2000-2020 (exploratory, added after the first results).

Uses every district-located GODAD activity of every year, not only the first infrastructure project, so far more events
inform the estimate than in the stacked event study (kh_s08).
  treatment   infra_{t-l}: 1 if a transport or energy activity started in the district in year t-l (l = 1..5),
              plus leads t+1, t+2 as a placebo (aid should not respond to infrastructure that has not started)
  benchmark   social_{t-l}: the same for social-sector activities; if social projects predict later aid as much as
              infrastructure does, the pattern is persistence of place, not infrastructure opening the door
  outcomes    (a) new non-infrastructure activities  (b) donors new to the district (first activity of that donor
              there)  (c) new social activities
PPML with district and year fixed effects, errors clustered by district. Sum of the five lags = cumulative response.
Optional: --major adds hand-coded major road projects of donors missing from GODAD (kh_major_roads_units.csv).
Output: outputs/tables/kh_infra_panel.csv, kh_infra_panel.txt (suffix _major with --major)
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

MAJOR = "--major" in sys.argv or "--major-only" in sys.argv; ONLY = "--major-only" in sys.argv; NOPP = "--no-pp" in sys.argv
SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, K, ROOT = ns["DF"], ns["L"], ns["K"], ns["ROOT"]
T, PROCD = ROOT / "outputs/tables", ROOT / "data/processed/cambodia"
SUF = ("_majoronly" if ONLY else "_major" if MAJOR else "") + ("_nopp" if NOPP else "")
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

L = L.copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce")
L["donor"] = L.project_id.map(K.groupby("project_id").donor.first())
L = L[L.unit.isin(DF.unit) & L.yr.between(1995, 2020)]
infra_ev = L[L.family == "infra"][["unit", "yr"]].assign(src="GODAD")
if ONLY:
    infra_ev = infra_ev.iloc[0:0]          # treatment = the hand-coded major roads alone
if MAJOR:
    mj = pd.read_csv(PROCD / "kh_major_roads_units.csv")
    infra_ev = pd.concat([infra_ev, mj[["unit", "start_year"]].rename(columns={"start_year": "yr"}).assign(src="major roads")])
    say(f"major road projects added: {mj.project.nunique()} projects, {len(mj)} district-project pairs")

years = range(2000, 2021)
grid = pd.MultiIndex.from_product([DF.unit, range(1995, 2023)], names=["unit", "yr"])
any_ = lambda frame: frame.groupby(["unit", "yr"]).size().reindex(grid, fill_value=0).clip(upper=1)
I, S = any_(infra_ev), any_(L[L.family == "social"])
non = L[~L.family.isin(["infra", "ict"])]
first = L.sort_values("yr").drop_duplicates(["unit", "donor"])          # first activity of each donor in each district
first = first[~first.family.isin(["infra", "ict"])]
Y = pd.DataFrame({"noninfra": non.groupby(["unit", "yr"]).w.sum(), "newdonor": first.groupby(["unit", "yr"]).size(),
                  "social": L[L.family == "social"].groupby(["unit", "yr"]).w.sum()}).reindex(grid, fill_value=0)
P = Y.reset_index()
for l in range(1, 6):
    P[f"i_l{l}"] = I.groupby(level=0).shift(l).values; P[f"s_l{l}"] = S.groupby(level=0).shift(l).values
for l in (1, 2):
    P[f"i_f{l}"] = I.groupby(level=0).shift(-l).values
P = P[P.yr.isin(years)].fillna(0).reset_index(drop=True)
if NOPP:
    P = P[P.unit.map(DF.set_index("unit").prov) != DF.loc[DF.pop_2019.idxmax(), "prov"]].reset_index(drop=True)   # drop the capital's province
say(f"panel: {P.unit.nunique()} districts x {P.yr.nunique()} years = {len(P)} obs | district-years with a new infrastructure "
    f"activity (as lag 1): {int(P.i_l1.sum())} | with a new social activity: {int(P.s_l1.sum())}")

IL, SL, LEAD = [f"i_l{l}" for l in range(1, 6)], [f"s_l{l}" for l in range(1, 6)], ["i_f1", "i_f2"]
rows = []
for y in ["noninfra", "newdonor", "social"]:
    d = P[P.groupby("unit")[y].transform("sum") > 0]                    # districts with no event never identify anything
    for spec, X in [("infra lags + leads", IL + LEAD), ("infra and social lags", IL + SL)]:
        m = smf.glm(f"{y} ~ " + " + ".join(X) + " + C(unit) + C(yr)", data=d, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]}, maxiter=500)
        for grp, vs in [("infra, sum of lags 1-5", IL), ("social, sum of lags 1-5", SL), ("infra leads (placebo)", LEAD)]:
            vs = [v for v in vs if v in X]
            if not vs:
                continue
            b = m.params[vs].sum(); se = float(np.sqrt(np.ones(len(vs)) @ m.cov_params().loc[vs, vs].values @ np.ones(len(vs))))
            rows.append({"outcome": y, "spec": spec, "term": grp, "pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * se) - 1),
                         "ci_hi": 100 * (np.exp(b + 1.96 * se) - 1), "p": 2 * norm.sf(abs(b / se)), "districts": d.unit.nunique()})
R = pd.DataFrame(rows); R.to_csv(T / f"kh_infra_panel{SUF}.csv", index=False)
say(R.round(3).to_string(index=False))
(T / f"kh_infra_panel{SUF}.txt").write_text("\n".join(log), encoding="utf-8")
