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
import os
zs = lambda s: (s - s.mean()) / s.std()
CEN = Path(os.environ.get("KH_CENSUS_IN", PROC / "kh_census.csv"))   # built by kh_census_build.py from the IPUMS extract
DF["speed"] = zs(np.log(DF.mob_mbps_shrunk))                          # Ookla mobile speed, 2019-2020 (check measure)
if CEN.exists():
    c = pd.read_csv(CEN); c19, c08 = c[c.year == 2019].set_index("unit"), c[c.year == 2008].set_index("unit")
    DF["phone"] = zs(DF.unit.map(c19.phone_share)); DF["deprivation"] = zs(DF.unit.map(c19.deprivation))
    DF["phone_2008"] = zs(DF.unit.map(c08.phone_share))
    for col, src_ in [("dep_nowater", c19.deprivation_nowater), ("dep_latok", c19.deprivation_latok), ("dep_pc1", c19.deprivation_pc1), ("dep_2008", c08.deprivation)]:
        DF[col] = zs(DF.unit.map(src_))
    DIGITAL, POVERTY = "phone", "deprivation"
    say(f"census 2019: {int(c19.n_indicators.iloc[0])} deprivation indicators | phone ownership mean {c19.phone_share.reindex(DF.unit).mean():.2f}, "
        f"SD {c19.phone_share.reindex(DF.unit).std():.3f}")
else:
    DIGITAL, POVERTY = "speed", "poverty"
    say(f"census file {CEN.name} not found: digital access = Ookla speed, poverty = Meta RWI")
C3 = {"reach": "physical access", DIGITAL: "digital access", POVERTY: "poverty"}
say("correlations: " + ", ".join(f"{C3[a]}-{C3[b]} {DF[a].corr(DF[b]):+.2f}" for a, b in [("reach", DIGITAL), ("reach", POVERTY), (DIGITAL, POVERTY)]))

say("== MAIN: one construct per model (physical access per doubling of travel time; digital access and poverty per SD)")
for v, lab in C3.items():
    estimate(f"{lab} alone", MAIN, [v])
R = pd.DataFrame(rows)
p3 = R[R.model.isin([f"{l} alone" for l in C3.values()])].sort_values("p_ri"); run = 0; holm = {}
for i, (ix, r) in enumerate(p3.iterrows()):
    run = max(run, min(1, (3 - i) * r.p_ri)); holm[r.model] = run
say("  Holm-adjusted RI p (three tests): " + ", ".join(f"{k} {v:.3f}" for k, v in holm.items()))
say("== SUPPLEMENTARY: all three together (not interpreted where |r| >= 0.7, spec 9-29)")
estimate("all three together", MAIN, list(C3))
for x, y in [("reach", DIGITAL), ("reach", POVERTY), (DIGITAL, POVERTY)]:      # pairs, interpreted only if |r| < 0.7
    r_ = DF[x].corr(DF[y])
    if abs(r_) < .7:
        estimate(f"pair: {C3[x]} + {C3[y]} (r = {r_:+.2f})", MAIN, [x, y])
    else:
        say(f"  pair {C3[x]} + {C3[y]}: r = {r_:+.2f}, not estimated (|r| >= 0.7)")

say("== CHECK MEASURES (alone)")
estimate("digital check: phone ownership 2008", MAIN, ["phone_2008"], draws=999) if "phone_2008" in DF else None
estimate("digital check: Ookla mobile speed", MAIN, ["speed"], draws=999)
estimate("poverty check: Meta RWI", MAIN, ["poverty"], draws=999)
for col, lab in [("dep_nowater", "without water (piped only)"), ("dep_latok", "latrine counted as improved"), ("dep_pc1", "first principal component"),
                 ("dep_2008", "2008 index")]:
    if col in DF:
        estimate(f"poverty check: {lab}", MAIN, [col], draws=999)


# ---------------- poverty chapter: night lights (economic activity, 2015, before the projects) ----------------
DF["lights"] = zs(np.log1p(DF.ntl_pc_2015 * 1000))
estimate("need check: night lights per person 2015", MAIN, ["lights"], draws=999)

# ---------------- conclusion: which matters more (spec 9-30) ----------------
say("== WHICH MATTERS MORE (spec 9-30): travel time, phones, deprivation, lights")
DF["reach_sd"] = zs(DF.reach)                                          # per SD of log travel time, for a common unit
FOUR = {"reach_sd": "physical access", DIGITAL: "digital access", POVERTY: "poverty", "lights": "night lights"}
CM = DF[list(FOUR)].corr()
say("  correlations:\n" + CM.round(2).rename(index=FOUR, columns=FOUR).to_string())
for v, lab in FOUR.items():                                             # (a) same unit: per SD
    estimate(f"per SD: {lab}", MAIN, [v], draws=999)
dd = counts(MAIN).dropna(subset=list(FOUR)).reset_index(drop=True)
dev = lambda xs: smf.glm("N ~ " + " + ".join(list(xs) + ["lnpop", "C(prov)"]), data=dd, family=sm.families.Poisson()).fit(maxiter=300).deviance
D0, Dnull = dev([]), smf.glm("N ~ 1", data=dd, family=sm.families.Poisson()).fit().deviance
import itertools, math
V = {S: (D0 - dev(S)) / Dnull for k in range(len(FOUR) + 1) for S in itertools.combinations(sorted(FOUR), k)}
key = lambda S: tuple(sorted(S))
shap = {j: sum(math.factorial(len(S)) * math.factorial(len(FOUR) - len(S) - 1) / math.factorial(len(FOUR)) * (V[key(S + (j,))] - V[key(S)])
               for k in range(len(FOUR)) for S in itertools.combinations(sorted(set(FOUR) - {j}), k)) for j in FOUR}
tot = sum(shap.values())
say(f"  (b) Shapley split of the deviance explained by the four (pseudo-R2 {1 - D0 / Dnull:.3f} -> {1 - dev(list(FOUR)) / Dnull:.3f}): " +
    ", ".join(f"{FOUR[j]} {100 * shap[j] / tot:.0f}%" for j in sorted(shap, key=shap.get, reverse=True)))
CVR = np.random.default_rng(11)


def cv_gain(xs, reps=20, k=5):
    """Out-of-sample Poisson deviance gain over province FE + ln population, folds stratified by province."""
    gains = []
    for _ in range(reps):
        fold = np.full(len(dd), -1)
        for pv, ix in dd.groupby("prov").indices.items():
            if len(ix) >= 2:
                fold[ix] = CVR.permutation(np.arange(len(ix)) % k)
        db = dm = 0.0
        for j in range(k):
            tr, te = dd[fold != j], dd[(fold == j)]
            te = te[te.prov.isin(tr.prov)]
            if te.empty:
                continue
            for xs_, acc_ in ((list(), "b"), (list(xs), "m")):
                m = smf.glm("N ~ " + " + ".join(xs_ + ["lnpop", "C(prov)"]), data=tr, family=sm.families.Poisson()).fit(maxiter=200)
                mu = np.clip(m.predict(te), 1e-9, None); y = te.N.values
                dv = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
                if acc_ == "b":
                    db += dv
                else:
                    dm += dv
        gains.append(100 * (db - dm) / db)
    return float(np.mean(gains)), float(np.std(gains))


CV = {j: cv_gain([j]) for j in FOUR}
say("  (c) cross-validated deviance gain over province FE + ln population (5-fold x 20): " +
    ", ".join(f"{FOUR[j]} {CV[j][0]:+.1f}% (sd {CV[j][1]:.1f})" for j in sorted(CV, key=lambda j: CV[j][0], reverse=True)))
for a, b in itertools.combinations(list(FOUR), 2):                     # (d) pairs, only where |r| < 0.7
    r_ = CM.loc[a, b]
    if abs(r_) < .7:
        estimate(f"pair per SD: {FOUR[a]} + {FOUR[b]} (r = {r_:+.2f})", MAIN, [a, b], draws=999)   # distinct label from the 9-29 pair models
    else:
        say(f"  pair {FOUR[a]} + {FOUR[b]}: r = {r_:+.2f}, not interpreted (|r| >= 0.7)")
pd.DataFrame([{"measure": FOUR[j], "var": j, "shapley_share": 100 * shap[j] / tot, "shapley_pts": 100 * shap[j], "cv_gain": CV[j][0], "cv_sd": CV[j][1]}
              for j in FOUR]).to_csv(T / "kh_main_importance.csv", index=False)

say("== WHERE THE GRADIENT COMES FROM (exploratory, spec 9-28): capital khans, provincial towns (Krong), rural districts")
DF["type"] = np.where(DF.capital, "capital", np.where(DF.ADMIN_NAME.str.contains("Krong"), "town", "rural"))
DF["capital_d"], DF["town_d"] = (DF.type == "capital").astype(int), (DF.type == "town").astype(int)
dt = counts(MAIN); tot_n, tot_p = dt.N.sum(), dt.pop_2019.sum()
for t, g in dt.groupby("type"):
    say(f"  {t:<8s} districts {len(g):>3d} | projects {g.N.sum():6.1f} ({100 * g.N.sum() / tot_n:.0f}%) | population {100 * g.pop_2019.sum() / tot_p:.0f}% | "
        f"projects per 100,000 people {1e5 * g.N.sum() / g.pop_2019.sum():.2f} | median travel time {g.tt_town_dry.median():.0f} min")
estimate("district type (vs rural)", MAIN, ["capital_d", "town_d"])
RUR = counts(MAIN)[lambda x: x.type == "rural"]
for v, lab in C3.items():
    estimate(f"rural districts: {lab} alone", MAIN, [v], extra=RUR)
estimate("rural districts: night lights alone", MAIN, ["lights"], extra=RUR)
estimate("rural districts: all three together", MAIN, list(C3), extra=RUR)

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
