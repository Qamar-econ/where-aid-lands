"""Regressions: H1 cross-section, heterogeneity, need-vs-access decomposition, Moran's I,
H2 panel (PPML with unit & year FE) and event study around corridor completion, validation vs BB5.
Outputs: outputs/tables/*.csv, outputs/results_summary.txt
"""
import itertools, re, warnings
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC, INT = ROOT / "data" / "processed", ROOT / "data" / "interim"
TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
LOG = open(ROOT / "outputs" / "results_summary.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n")


U = gpd.read_file(PROC / "units.gpkg")
U["dili"] = (U.unit == "TL06U").astype(int)
z = lambda s: (s - s.mean()) / s.std()
U["rwi_z"] = z(U.rwi)
U["poor_z"] = -U["rwi_z"]          # higher = poorer (need)
U["log_tt_wet_z"] = z(np.log(U.tt_wet_min))
U["paved_z"] = z(U.paved_share)
U["roadden_z"] = z(np.log(U.road_density))
U["net_z"] = z(U.access_net)
U["phys_z"] = z(U.access_phys)
U["steep_z"] = z(U.steep_share)
U["density_z"] = z(np.log(U.density))


def nb(formula, data, name):
    """Negative binomial (NB2) with HC1 robust SE; returns tidy rows with IRR."""
    y = formula.split("~")[0].strip()
    d = data.copy()
    d[y] = d[y].round().astype(int)
    m = smf.negativebinomial(formula, d).fit(disp=0, maxiter=500, cov_type="HC1")
    rows = []
    for v in m.params.index:
        if v in ("Intercept", "alpha"):
            continue
        rows.append({"model": name, "term": v, "coef": m.params[v], "se": m.bse[v], "p": m.pvalues[v],
                     "irr": np.exp(m.params[v]), "n": int(m.nobs), "pseudo_r2": m.prsquared})
    return m, rows


def fmt(rows):
    return "\n".join(f"   {r['term']:<16} b={r['coef']:+.3f} (se {r['se']:.3f}) p={r['p']:.3f}  IRR={r['irr']:.2f}" for r in rows)


allrows = []
say("=" * 90, "\nH1. Cross-section, 62 units (Dili urban = 4 merged postos). Outcome counts of non-infrastructure\n"
    "development projects started 2000-2026. NB2, robust SE. access_pc1 & need standardised (1 SD).")
SPECS = [
    ("M1 all units, main outcome", "n_dev ~ access_pc1 + log_pop + poor_z", U),
    ("M2 + Dili dummy", "n_dev ~ access_pc1 + log_pop + poor_z + dili", U),
    ("M3 excl. Dili urban", "n_dev ~ access_pc1 + log_pop + poor_z", U[U.dili == 0]),
    ("M4 precise-only outcome", "np_dev ~ access_pc1 + log_pop + poor_z", U),
    ("M5 precise-only, excl. Dili", "np_dev ~ access_pc1 + log_pop + poor_z", U[U.dili == 0]),
    ("M6 precise, + density", "np_dev ~ access_pc1 + log_pop + poor_z + density_z", U[U.dili == 0]),
]
for name, f, d in SPECS:
    m, r = nb(f, d, name); allrows += r
    say(f"\n{name}: {f}  (N={int(m.nobs)}, pseudo-R2={m.prsquared:.3f})\n" + fmt(r))

# OLS on per-capita dollars
U["log_usd_pc"] = np.log1p(U.usd_dev_pc)
m = smf.ols("log_usd_pc ~ access_pc1 + log_pop + poor_z", U[U.dili == 0]).fit(cov_type="HC1")
say(f"\nM7 OLS log(1+USD per capita), excl. Dili (N={int(m.nobs)}, R2={m.rsquared:.3f})")
for v in ["access_pc1", "log_pop", "poor_z"]:
    say(f"   {v:<16} b={m.params[v]:+.3f} (se {m.bse[v]:.3f}) p={m.pvalues[v]:.3f}")
    allrows.append({"model": "M7 OLS log USD pc", "term": v, "coef": m.params[v], "se": m.bse[v], "p": m.pvalues[v], "n": int(m.nobs)})

# municipality fixed effects: within-municipality selection
m = smf.ols("np.log1p(np_dev) ~ access_pc1 + log_pop + poor_z + C(mun_pcode)", U[U.dili == 0]).fit(cov_type="HC1")
say(f"\nM8 OLS log(1+precise count) with municipality FE, excl. Dili (N={int(m.nobs)}, R2={m.rsquared:.3f})")
for v in ["access_pc1", "log_pop", "poor_z"]:
    say(f"   {v:<16} b={m.params[v]:+.3f} (se {m.bse[v]:.3f}) p={m.pvalues[v]:.3f}")
    allrows.append({"model": "M8 OLS muni FE", "term": v, "coef": m.params[v], "se": m.bse[v], "p": m.pvalues[v], "n": int(m.nobs)})

say("\n" + "=" * 90 + "\nComponents (each entered separately; precise outcome; controls log_pop + poor_z; excl. Dili)")
comp_rows = []
for v, lab in [("log_tt_wet_z", "log wet-season travel time (+ = farther)"), ("paved_z", "paved share"),
               ("roadden_z", "log road density"), ("steep_z", "steep/landslide terrain share"),
               ("phys_z", "physical sub-index"), ("net_z", "network sub-index (Ookla)")]:
    m, r = nb(f"np_dev ~ {v} + log_pop + poor_z", U[U.dili == 0], f"C {lab}")
    rr = [x for x in r if x["term"] == v][0]; rr["label"] = lab; comp_rows.append(rr); allrows += r
    say(f"   {lab:<42} b={rr['coef']:+.3f} p={rr['p']:.3f} IRR={rr['irr']:.2f}")
m, r = nb("np_dev ~ phys_z + net_z + log_pop + poor_z", U[U.dili == 0], "C phys+net joint"); allrows += r
say("   joint physical + network:\n" + fmt(r))
pd.DataFrame(comp_rows).to_csv(TAB / "h1_components.csv", index=False)

say("\n" + "=" * 90 + "\nHeterogeneity (NB, access_pc1 + log_pop + poor_z, excl. Dili; outcome = all counts incl. muni split)")
het = []
for y, lab in [("n_dev_e1", "Era 1: start 2000-2015"), ("n_dev_e2", "Era 2: start 2016-2026"),
               ("n_dev_bilat", "Bilateral donors"), ("n_dev_multi", "Multilaterals & UN"), ("n_dev_kj", "Korea + Japan"),
               ("n_SOC_education", "Education"), ("n_SOC_health", "Health"), ("n_SOC_water", "Water & sanitation"),
               ("n_GOV", "Governance"), ("n_PROD", "Productive sectors"), ("n_access", "ACCESS infrastructure projects"),
               ("n_human", "Humanitarian")]:
    try:
        m, r = nb(f"{y} ~ access_pc1 + log_pop + poor_z", U[U.dili == 0], f"H {lab}")
    except Exception as e:
        say("   fail", lab, e); continue
    a = [x for x in r if x["term"] == "access_pc1"][0]; p = [x for x in r if x["term"] == "poor_z"][0]
    het.append({"group": lab, "b_access": a["coef"], "se_access": a["se"], "p_access": a["p"],
                "b_need": p["coef"], "se_need": p["se"], "p_need": p["p"], "total": U.loc[U.dili == 0, y].sum()})
    allrows += r
    say(f"   {lab:<32} access b={a['coef']:+.3f} (p={a['p']:.3f})   poverty b={p['coef']:+.3f} (p={p['p']:.3f})   N proj={U.loc[U.dili==0, y].sum():.0f}")
pd.DataFrame(het).to_csv(TAB / "h1_heterogeneity.csv", index=False)

# ---------------- relative importance: Shapley decomposition of R2 ----------------
say("\n" + "=" * 90 + "\nRelative importance (Shapley R2 decomposition, OLS log(1+precise count per 10k pop), excl. Dili)")
D = U[U.dili == 0].copy()
D["y"] = np.log1p(D.np_dev / D.pop2022 * 1e4)
preds = {"Accessibility (PC1)": "access_pc1", "Poverty (-RWI)": "poor_z", "Population size": "log_pop", "Density": "density_z"}


def r2(vs):
    if not vs:
        return 0.0
    return smf.ols("y ~ " + " + ".join(vs), D).fit().rsquared


keys = list(preds)
sh = {k: 0.0 for k in keys}
from math import factorial
n = len(keys)
for k in keys:
    others = [o for o in keys if o != k]
    for s in range(n):
        for S in itertools.combinations(others, s):
            w = factorial(s) * factorial(n - s - 1) / factorial(n)
            sh[k] += w * (r2([preds[x] for x in S] + [preds[k]]) - r2([preds[x] for x in S]))
tot = r2(list(preds.values()))
shp = pd.DataFrame({"predictor": keys, "shapley_r2": [sh[k] for k in keys]})
shp["share"] = shp.shapley_r2 / tot
shp.to_csv(TAB / "shapley_r2.csv", index=False)
say(shp.round(3).to_string(index=False), f"\n   total R2 = {tot:.3f}")
m = smf.ols("y ~ access_pc1 + poor_z + log_pop + density_z", D).fit(cov_type="HC1")
say("   OLS coefficients:\n" + "\n".join(f"   {v:<12} b={m.params[v]:+.3f} p={m.pvalues[v]:.3f}" for v in m.params.index[1:]))

# ---------------- Moran's I on M5 residuals ----------------
m5 = smf.negativebinomial("np_dev ~ access_pc1 + log_pop + poor_z", D.assign(np_dev=D.np_dev.round().astype(int))).fit(disp=0)
res = (D.np_dev - m5.predict()) / np.sqrt(m5.predict())
geo = D.to_crs(32751).reset_index(drop=True)
W = np.zeros((len(geo), len(geo)))
for i, gi in enumerate(geo.geometry):
    for j, gj in enumerate(geo.geometry):
        if i != j and gi.buffer(50).intersects(gj):
            W[i, j] = 1
W = W / np.maximum(W.sum(1, keepdims=True), 1)


def moran(x):
    x = np.asarray(x) - np.mean(x)
    return (len(x) / W.sum()) * (x @ W @ x) / (x @ x)


I = moran(res.values)
rng = np.random.default_rng(0)
perm = np.array([moran(rng.permutation(res.values)) for _ in range(999)])
say(f"\nMoran's I of M5 Pearson residuals = {I:.3f} (permutation p = {(np.sum(perm >= I) + 1) / 1000:.3f})")
I_y = moran(D.np_dev / D.pop2022)
say(f"Moran's I of precise projects per capita = {I_y:.3f}")

pd.DataFrame(allrows).to_csv(TAB / "h1_all_models.csv", index=False)

# ---------------- H2 panel ----------------
say("\n" + "=" * 90 + "\nH2. Unit-year panel 2006-2025 (excl. Dili urban). New development projects started per unit-year.\n"
    "Time-varying wet-season travel time from corridor completion dates. PPML with unit & year FE, SE clustered by unit.")
P = pd.read_csv(PROC / "unit_year_panel.csv")
P = P[P.unit != "TL06U"].copy()
P["log_tt"] = np.log(P.tt_wet_min)
for y in ["n_dev_new", "np_dev_new", "n_access_new"]:
    m = smf.glm(f"{y} ~ log_tt + C(unit) + C(year)", P, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.unit)[0]})
    say(f"   {y:<14} elasticity wrt travel time = {m.params['log_tt']:+.3f} (se {m.bse['log_tt']:.3f}, p={m.pvalues['log_tt']:.3f}); "
        f"a 30% faster trip -> {(-m.params['log_tt'] * np.log(1/0.7)) * 100:+.1f}% projects")

# event study: treated = >=60 min wet-season gain; rel. years binned [-6+, +5+]; ref = -1
P["rel"] = P.year - P.treat_year
P["relb"] = P.rel.clip(-6, 5)
ev = []
for y in ["n_dev_new", "np_dev_new"]:
    d = P.copy()
    terms = []
    for k in range(-6, 6):
        if k == -1:
            continue
        c = f"e_{'m' if k < 0 else 'p'}{abs(k)}"
        d[c] = ((d.relb == k) & d.treat_year.notna()).astype(int)
        terms.append((k, c))
    f = f"{y} ~ " + " + ".join(c for _, c in terms) + " + C(unit) + C(year)"
    m = smf.glm(f, d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]})
    for k, c in terms:
        ev.append({"outcome": y, "rel_year": k, "coef": m.params[c], "se": m.bse[c], "p": m.pvalues[c]})
    ev.append({"outcome": y, "rel_year": -1, "coef": 0, "se": 0, "p": np.nan})
    post = [c for k, c in terms if k >= 0]; pre = [c for k, c in terms if k < -1]
    say(f"   event study {y}: mean post coef = {np.mean([m.params[c] for c in post]):+.3f}; mean pre coef = {np.mean([m.params[c] for c in pre]):+.3f}")
    wt = m.wald_test(" , ".join(f"{c} = 0" for c in pre), scalar=True)
    say(f"     joint pre-trend test p = {float(wt.pvalue):.3f}")
EV = pd.DataFrame(ev).sort_values(["outcome", "rel_year"])
EV.to_csv(TAB / "h2_event_study.csv", index=False)

# simple DiD by era: treated vs not, 2006-2015 vs 2021-2025
P["post"] = (P.year >= P.treat_year.fillna(9999)).astype(int)
m = smf.glm("n_dev_new ~ post + C(unit) + C(year)", P, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.unit)[0]})
say(f"   staggered DiD (PPML TWFE) post-upgrade effect on new dev projects: {m.params['post']:+.3f} (se {m.bse['post']:.3f}, p={m.pvalues['post']:.3f}) -> IRR {np.exp(m.params['post']):.2f}")

# ---------------- validation vs official Budget Book municipal counts ----------------
say("\n" + "=" * 90 + "\nValidation: our DB vs Ministry of Finance Budget Book municipal project counts (2016, 2017)")
X = pd.read_csv(PROC / "unit_project_links.csv")
PR = pd.read_csv(PROC / "oda_projects.csv")
cov = pd.read_csv(PROC / "posto_covariates.csv")
u2m = U.set_index("unit").municipio.to_dict()
vals = []
for yr in [2016, 2017]:
    t = (INT / "bb5_text" / f"{yr}.txt").read_text(encoding="utf-8")
    off = {}
    for mm in re.finditer(r"([A-Za-zçé]+)\s+Municipality\s+\((\d+)\s+projects\)", t):
        off.setdefault(mm.group(1).replace("Liquica", "Liquiçá").replace("Lautem", "Lautém").replace("Oecusse", "Oecussi"), int(mm.group(2)))
    # our DB: projects active in that year (start <= yr) with sub-national assignment, by municipality
    act = PR[(PR.start_year <= yr) & (PR.start_year >= yr - 4)].uid
    xx = X[X.uid.isin(act)].copy(); xx["mun"] = xx.unit.map(u2m)
    ours = xx.groupby("mun").uid.nunique()
    for k, v in off.items():
        vals.append({"year": yr, "municipio": k, "official": v, "ours": ours.get(k, 0)})
V = pd.DataFrame(vals)
V.to_csv(TAB / "validation_bb5.csv", index=False)
from scipy.stats import spearmanr, pearsonr
say(f"   N municipality-years = {len(V)}; Spearman rho = {spearmanr(V.official, V.ours)[0]:.2f}; Pearson r = {pearsonr(V.official, V.ours)[0]:.2f}")
LOG.close()
