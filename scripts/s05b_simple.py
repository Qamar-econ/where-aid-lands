"""Simple, transparent statistics complementing s05_models.py.

  1. Access quintiles: mean projects, per-capita, share of projects vs share of population
  2. Correlation matrix (Pearson & Spearman) with p-values
  3. High vs low access (median split): Welch t-test & Mann-Whitney U
  4. Bivariate OLS lines
  5. Concentration: Gini of projects per capita; concentration index ranked by access and by poverty
  6. 2x2 means: upgraded-corridor units vs others, before (2006-2015) vs after (2016-2025)
Outputs: outputs/tables/simple_*.csv, appended to outputs/results_simple.txt
"""
import numpy as np, pandas as pd, geopandas as gpd
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data" / "processed", ROOT / "outputs" / "tables"
LOG = open(ROOT / "outputs" / "results_simple.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")


U = gpd.read_file(PROC / "units.gpkg").drop(columns="geometry")
U["np_per10k"] = U.np_dev / U.pop2022 * 1e4
U["n_per10k"] = U.n_dev / U.pop2022 * 1e4
U["poverty"] = -U.rwi
D = U[U.unit != "TL06U"].copy()          # main simple stats exclude Dili (urban); shown with Dili where noted

# 1. quintiles
say("1. Access quintiles (PC1), 61 units excl. Dili urban; Dili shown separately")
D["q"] = pd.qcut(D.access_pc1, 5, labels=["Q1 least", "Q2", "Q3", "Q4", "Q5 most"])
Q = D.groupby("q", observed=True).agg(units=("unit", "size"), pop=("pop2022", "sum"), tt_wet_h=("tt_wet_min", lambda s: s.mean() / 60),
                                      rwi=("rwi", "mean"), precise=("np_dev", "sum"), all_=("n_dev", "sum"))
Q.loc["Dili (urban)"] = [1, U.loc[U.unit == "TL06U", "pop2022"].iloc[0], U.loc[U.unit == "TL06U", "tt_wet_min"].iloc[0] / 60,
                         U.loc[U.unit == "TL06U", "rwi"].iloc[0], U.loc[U.unit == "TL06U", "np_dev"].iloc[0], U.loc[U.unit == "TL06U", "n_dev"].iloc[0]]
Q["pop_share_%"] = Q["pop"] / Q["pop"].sum() * 100
Q["precise_share_%"] = Q.precise / Q.precise.sum() * 100
Q["precise_per10k"] = Q.precise / Q["pop"] * 1e4
Q["ratio_share"] = Q["precise_share_%"] / Q["pop_share_%"]
Q.to_csv(TAB / "simple_quintiles.csv")
say(Q.round(2).to_string())

# 2. correlations
say("\n2. Correlations (61 units excl. Dili). r = Pearson, rho = Spearman")
vars_ = {"np_per10k": "precise projects /10k", "np_dev": "precise projects (count)", "access_pc1": "access index",
         "tt_wet_min": "wet travel time", "access_net": "network sub-index", "access_phys": "physical sub-index",
         "poverty": "poverty (-RWI)", "pop2022": "population", "density": "density"}
rows = []
for a in ["np_dev", "np_per10k"]:
    for b in [v for v in vars_ if v not in ("np_dev", "np_per10k")]:
        r, pr = stats.pearsonr(D[a], D[b]); rho, ps = stats.spearmanr(D[a], D[b])
        rows.append({"y": vars_[a], "x": vars_[b], "pearson_r": r, "p_r": pr, "spearman_rho": rho, "p_rho": ps})
C = pd.DataFrame(rows); C.to_csv(TAB / "simple_correlations.csv", index=False)
say(C.round(3).to_string(index=False))
r, p = stats.spearmanr(D.access_pc1, D.poverty)
say(f"   access vs poverty: rho = {r:.2f} (p={p:.3f})  -> accessible postos are also richer (collinearity to keep in mind)")

# 3. median split tests
say("\n3. High vs low accessibility (median split of PC1), 61 units")
hi, lo = D[D.access_pc1 >= D.access_pc1.median()], D[D.access_pc1 < D.access_pc1.median()]
T = []
for v, lab in [("np_dev", "precise projects (count)"), ("n_dev", "all projects incl. muni split"), ("np_per10k", "precise per 10k"),
               ("n_per10k", "all per 10k"), ("pop2022", "population"), ("rwi", "RWI (wealth)")]:
    t, pt = stats.ttest_ind(hi[v], lo[v], equal_var=False); u, pu = stats.mannwhitneyu(hi[v], lo[v])
    T.append({"variable": lab, "mean_high": hi[v].mean(), "mean_low": lo[v].mean(), "welch_t": t, "p_t": pt, "mannwhitney_p": pu})
T = pd.DataFrame(T); T.to_csv(TAB / "simple_median_split.csv", index=False)
say(T.round(3).to_string(index=False))

# 4. bivariate OLS
say("\n4. Bivariate OLS (61 units)")
B = []
for f in ["np_dev ~ access_pc1", "np_dev ~ np.log(tt_wet_min)", "np_dev ~ access_net", "np_dev ~ poverty", "np_per10k ~ access_pc1",
          "np_per10k ~ poverty", "np.log(np_dev+1) ~ np.log(pop2022)"]:
    m = smf.ols(f, D).fit(cov_type="HC1")
    v = m.params.index[1]
    B.append({"model": f, "slope": m.params[v], "se": m.bse[v], "p": m.pvalues[v], "r2": m.rsquared})
B = pd.DataFrame(B); B.to_csv(TAB / "simple_bivariate.csv", index=False)
say(B.round(3).to_string(index=False))

# 5. concentration
say("\n5. Concentration of precise development projects relative to population (62 units incl. Dili)")


def conc_index(df, rank_by, y="np_dev"):
    d = df.sort_values(rank_by)
    cp = np.r_[0, d.pop2022.cumsum() / d.pop2022.sum()]
    cy = np.r_[0, d[y].cumsum() / d[y].sum()]
    return 1 - 2 * np.trapezoid(cy, cp), cp, cy


for rk, lab in [("access_pc1", "ranked by accessibility (low->high)"), ("rwi", "ranked by wealth (poor->rich)"),
                ("np_per10k", "Gini (ranked by projects per capita)")]:
    for df, s in [(U, "incl. Dili"), (D, "excl. Dili")]:
        ci, _, _ = conc_index(df, rk)
        say(f"   {lab:<40} {s:<11} index = {ci:+.3f}")
say("   (concentration index > 0: projects concentrated in more accessible / richer units relative to population)")

# 6. 2x2 corridor table
say("\n6. Upgraded-corridor units vs others: new development projects per unit-year (excl. Dili urban)")
P = pd.read_csv(PROC / "unit_year_panel.csv")
P = P[P.unit != "TL06U"]
P["group"] = np.where(P.treat_year.notna(), "Upgraded corridor (>=60 min gain)", "Not upgraded")
P["period"] = np.where(P.year <= 2015, "2006-2015 (before)", "2016-2025 (after)")
for y in ["n_dev_new", "np_dev_new"]:
    tab = P.pivot_table(index="group", columns="period", values=y, aggfunc="mean")
    tab["change_%"] = (tab.iloc[:, 1] / tab.iloc[:, 0] - 1) * 100
    say(f"   outcome: {y}\n" + tab.round(3).to_string())
    did = (tab.iloc[1, 1] - tab.iloc[1, 0]) - (tab.iloc[0, 1] - tab.iloc[0, 0])
    say(f"   simple difference-in-differences (upgraded - not) = {did:+.3f} projects per unit-year")
    tab.to_csv(TAB / f"simple_2x2_{y}.csv")
LOG.close()
