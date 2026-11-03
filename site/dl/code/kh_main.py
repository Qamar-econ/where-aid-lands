"""Main analysis (spec 9-27): within a province, do aid projects go where it is easy to reach, or where people are poor?

Outcome    N_d = new GODAD activities starting 2017-2020 located in district d (split 1/K over K districts), ICT excluded,
           town- and province-name geocodes reclassified (spec 9-26)
Reach      log2 population-weighted travel time to the nearest town of 20,000+ (dry season, 2017 OSM network):
           the coefficient is the change in projects when travel time doubles
Poverty    -Meta Relative Wealth Index, standardised (per SD)
Model      PPML with province fixed effects and ln population: reach alone, poverty alone, both
Inference  Freedman-Lane randomisation inference (residuals permuted within province), primary; clustered and Conley 100 km
           reported; Holm over the two single-factor tests
Then five robustness checks, donor groups, and one descriptive figure.
Output: outputs/tables/kh_main_simple.csv, kh_main_simple.txt; outputs/figures/cambodia/M1_reach_quintiles.png
"""
import sys, numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from scipy.stats import norm
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kh_common import reclassify_geocodes

ROOT = Path(__file__).resolve().parents[1]
PROC, T, FIG = ROOT / "data/processed/cambodia", ROOT / "outputs/tables", ROOT / "outputs/figures/cambodia"
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
RNG = np.random.default_rng(20261103)
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

# ---------------- districts ----------------
DF = pd.read_csv(PROC / "kh_covariates.csv").merge(pd.read_csv(PROC / "kh_access.csv").rename(columns={"k": "unit"}), on="unit", how="left")
DF = DF[~DF.exclude].reset_index(drop=True)
U = gpd.read_file(PROC / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"}).to_crs(32648); U["unit"] = U.unit.astype("int64")
cen = U.set_index("unit").geometry.centroid
DF["x_km"], DF["y_km"] = DF.unit.map(cen.x) / 1000, DF.unit.map(cen.y) / 1000
DF["prov"] = DF.PARENT.astype(str); DF["lnpop"] = np.log(DF.pop_2019)
DF["capital"] = DF.unit.isin({116008004, 116008005, 116008007, 116008011})
DF["access"] = DF.access_pc1
DF["reach"] = np.log2(DF.tt_town_dry)                                     # +1 = travel time doubles
rwi = pd.read_csv(PROC / "kh_rwi.csv").set_index("unit").rwi
DF["poverty"] = -DF.unit.map(rwi); DF["poverty"] = (DF.poverty - DF.poverty.mean()) / DF.poverty.std()
DF["index"] = (DF.access_pc1 - DF.access_pc1.mean()) / DF.access_pc1.std()
weiss = pd.read_csv(PROC / "kh_weiss_district.csv").set_index("unit").w_city
DF["reach_weiss"] = np.log2(DF.unit.map(weiss))

# ---------------- projects ----------------
A0 = pd.read_csv(PROC / "kh_assign.csv").merge(pd.read_csv(PROC / "kh_projects.csv")[["project_id", "dgroup", "sector_code"]], on="project_id")
A0 = A0[pd.to_numeric(A0.sector_code.astype(str).str.extract(r"(\d{3})")[0], errors="coerce").ne(220)]
ADM2 = A0[(A0.level == "adm2") & A0.unit.isin(DF.unit)]
ADM1 = A0[A0.level == "adm1_spread"]
MAIN = reclassify_geocodes(ADM2, DF, PROC)


def counts(a):
    return DF.assign(N=DF.unit.map(a.groupby("unit").w.sum()).fillna(0))


# ---------------- estimation ----------------
def fit(d, xs):
    f = "N ~ " + " + ".join(xs + ["lnpop", "C(prov)"])
    return f, smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)


def conley_p(r, d, v, cutoff=100):
    X = r.model.exog; mu = r.fittedvalues.values; S = X * (r.model.endog - mu)[:, None]
    Dm = np.hypot(d.x_km.values[:, None] - d.x_km.values[None, :], d.y_km.values[:, None] - d.y_km.values[None, :])   # from coordinates: safe for subsets
    K = np.clip(1 - Dm / cutoff, 0, None); B = np.linalg.pinv(X.T @ (X * mu[:, None]))
    j = r.model.exog_names.index(v)
    return 2 * norm.sf(abs(r.params[v] / np.sqrt((B @ (S.T @ K @ S) @ B)[j, j])))


def ri_p(d, v, others, f, b, draws):
    """Freedman-Lane: v = fit on (others, ln pop, province) + residual; residuals permuted within province."""
    aux = smf.ols(f"{v} ~ " + " + ".join(others + ["lnpop", "C(prov)"]), data=d).fit()
    fv, e0, g = aux.fittedvalues.values, aux.resid.values, d.prov.values; idx = [np.where(g == k)[0] for k in np.unique(g)]; bs = []
    for _ in range(draws):
        e = e0.copy()
        for ix in idx:
            e[ix] = RNG.permutation(e0[ix])
        bs.append(smf.glm(f, data=d.assign(**{v: fv + e}), family=sm.families.Poisson()).fit(maxiter=100).params[v])
    return (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs))


rows = []


def estimate(label, a, xs, draws=DRAWS, extra=None):
    d = (extra if extra is not None else counts(a)).dropna(subset=xs).reset_index(drop=True)
    f, r = fit(d, xs)
    for v in xs:
        b, s = r.params[v], r.bse[v]
        row = {"model": label, "var": v, "pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * s) - 1), "ci_hi": 100 * (np.exp(b + 1.96 * s) - 1),
               "p_ri": ri_p(d, v, [x for x in xs if x != v], f, b, draws) if draws else np.nan, "p_cluster": r.pvalues[v], "p_conley100": conley_p(r, d, v),
               "projects": d.N.sum(), "districts": len(d)}
        rows.append(row)
        say(f"  {label:<44s} {v:<12s} {row['pct']:+5.0f}% (95% CI {row['ci_lo']:+.0f}% to {row['ci_hi']:+.0f}%) | p RI {row['p_ri']:.3f} | "
            f"cluster {row['p_cluster']:.3f} | Conley {row['p_conley100']:.3f} | projects {row['projects']:.1f}")
    return r


say(f"districts {len(DF)} | projects {MAIN.project_id.nunique()} (weight {MAIN.w.sum():.1f}) | corr(reach, poverty) = {DF.reach.corr(DF.poverty):.2f} | "
    f"median travel time {DF.tt_town_dry.median():.0f} min (IQR {DF.tt_town_dry.quantile(.25):.0f}-{DF.tt_town_dry.quantile(.75):.0f})")
say("== MAIN (reach: per doubling of travel time; poverty: per SD)")
estimate("reach alone", MAIN, ["reach"]); estimate("poverty alone", MAIN, ["poverty"]); estimate("both", MAIN, ["reach", "poverty"])
R = pd.DataFrame(rows)
p2 = R[R.model.isin(["reach alone", "poverty alone"])].sort_values("p_ri"); run = 0; holm = {}
for i, (ix, r) in enumerate(p2.iterrows()):
    run = max(run, min(1, (2 - i) * r.p_ri)); holm[r.model] = run
say("  Holm-adjusted RI p: " + ", ".join(f"{k} {v:.3f}" for k, v in holm.items()))

import os
CEN = Path(os.environ.get("KH_CENSUS_IN", PROC / "kh_census.csv"))   # built by kh_census_build.py once the IPUMS extract arrives
if CEN.exists():
    c = pd.read_csv(CEN); c19 = c[c.year == 2019].set_index("unit")
    DF["deprivation"] = DF.unit.map(c19.deprivation); DF["deprivation"] = (DF.deprivation - DF.deprivation.mean()) / DF.deprivation.std()
    say(f"== CENSUS 2019 (pre-registered H-C1: deprivation per SD; {int(c19.n_indicators.iloc[0])} indicators) | corr(reach, deprivation) = {DF.reach.corr(DF.deprivation):.2f}")
    estimate("H-C1 deprivation alone", MAIN, ["deprivation"]); estimate("H-C1b reach + deprivation", MAIN, ["reach", "deprivation"])
else:
    say(f"== CENSUS: {CEN.name} not found, H-C1 skipped (IPUMS extract pending)")

say("== WHERE THE GRADIENT COMES FROM (exploratory, spec 9-28): capital khans, provincial towns (Krong), rural districts")
DF["type"] = np.where(DF.capital, "capital", np.where(DF.ADMIN_NAME.str.contains("Krong"), "town", "rural"))
DF["capital_d"], DF["town_d"] = (DF.type == "capital").astype(int), (DF.type == "town").astype(int)
dt = counts(MAIN); tot_n, tot_p = dt.N.sum(), dt.pop_2019.sum()
for t, g in dt.groupby("type"):
    say(f"  {t:<8s} districts {len(g):>3d} | projects {g.N.sum():6.1f} ({100 * g.N.sum() / tot_n:.0f}%) | population {100 * g.pop_2019.sum() / tot_p:.0f}% | "
        f"projects per 100,000 people {1e5 * g.N.sum() / g.pop_2019.sum():.2f} | median travel time {g.tt_town_dry.median():.0f} min")
estimate("district type (vs rural)", MAIN, ["capital_d", "town_d"])
RUR = counts(MAIN)[lambda x: x.type == "rural"]
estimate("rural districts: reach alone", MAIN, ["reach"], extra=RUR)
estimate("rural districts: poverty alone", MAIN, ["poverty"], extra=RUR)
estimate("rural districts: both", MAIN, ["reach", "poverty"], extra=RUR)

say("== ROBUSTNESS (reach alone)")
estimate("R1a geocodes as published (no reclassification)", ADM2, ["reach"], draws=999)
estimate("R1b worst case: province-level geocodes to remotest district", reclassify_geocodes(ADM2, DF.assign(access=-DF.tt_town_dry), PROC, how="worst"), ["reach"], draws=999)
for lab, pick in [("R2a province-only projects added at remotest district", "max"), ("R2b province-only projects added at best-reached district", "min")]:
    add = []
    for pid, g_ in ADM1.groupby("project_id"):
        provs = DF[DF.unit.isin(g_.unit)].prov.unique()
        for pv in provs:
            dd = DF[DF.prov == pv]; u = dd.loc[dd.tt_town_dry.idxmax() if pick == "max" else dd.tt_town_dry.idxmin(), "unit"]
            add.append({"project_id": pid, "unit": u, "w": 1 / len(provs)})
    estimate(lab, pd.concat([MAIN, pd.DataFrame(add)]), ["reach"], draws=999)
estimate("R3 Phnom Penh excluded", MAIN, ["reach"], draws=999, extra=counts(MAIN)[lambda d: ~d.capital])
estimate("R4 Weiss et al. (2018) travel time to cities", MAIN, ["reach_weiss"], draws=999)
estimate("R5 ten-component physical-access index (per SD)", MAIN, ["index"], draws=999)
say("== DONOR GROUPS (reach alone)")
for dg in ["EU/US", "World Bank", "China"]:
    sub = MAIN[MAIN.project_id.isin(A0[A0.dgroup == dg].project_id)]
    estimate(f"donors: {dg}", sub, ["reach"], draws=999)
    rows[-1]["activities"] = sub.project_id.nunique()        # clustered SE ignores that one activity spans many districts; RI does not rely on it
    say(f"    {dg}: {sub.project_id.nunique()} activities over {sub.unit.nunique()} districts")
pd.DataFrame(rows).to_csv(T / "kh_main_simple.csv", index=False)

# ---------------- figure: projects per 100,000 people by travel-time quintile, within province ----------------
d = counts(MAIN); d["rel"] = d.reach - d.groupby("prov").reach.transform("mean"); d["q"] = pd.qcut(d.rel, 5, labels=False)
Q = d.groupby("q").agg(N=("N", "sum"), pop=("pop_2019", "sum"), lo=("tt_town_dry", "min"), hi=("tt_town_dry", "max")); Q["rate"] = 1e5 * Q.N / Q["pop"]
say("== FIGURE: projects per 100,000 people by within-province travel-time quintile (1 = closest): " + ", ".join(f"Q{i + 1} {v:.2f}" for i, v in enumerate(Q.rate)))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
ax.bar(range(1, 6), Q.rate, color="#0b6b6b", width=.65)
for i, v in enumerate(Q.rate):
    ax.text(i + 1, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(1, 6)); ax.set_xticklabels(["1\nclosest", "2", "3", "4", "5\nfarthest"])
ax.set_xlabel("Travel time to the nearest town, quintile within province"); ax.set_ylabel("Projects per 100,000 people")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "M1_reach_quintiles.png"); plt.close(fig)
(T / "kh_main_simple.txt").write_text("\n".join(log), encoding="utf-8")
