"""Cambodia models, as specified in 모형명세서_캄보디아.md (written before estimation).

  E[N_d | X_d, Pop_d, p] = exp( alpha_p(d) + beta_X X_d + theta ln Pop_d )        PPML, one construct per model

  N_d   GODAD projects starting 2017-2020 with district-reliable locations, split equally over a project's districts
  X     combined access (PC1 of physical + connectivity components) | physical access (PC1, OSM network) |
        connectivity (log shrunk mobile speed) | night lights per person (log)        -- all standardised
  alpha province fixed effects (25); theta from the population-only model (H0: theta = 1)
Inference for each beta: (i) province-clustered sandwich SE, (ii) Conley spatial-HAC SE (Bartlett, 50 km and 100 km),
(iii) Freedman-Lane randomization inference (residualised X permuted within province, 2,000 draws); decision uses the most conservative
p. Holm correction over the three decomposed confirmatory tests. MDE = exp(2.8 SE_max) - 1.
Robustness R1-R9 and heterogeneity as in the specification.
Outputs: outputs/tables/kh_main.csv, kh_robustness.csv, kh_heterogeneity.csv, outputs/results_kh_models.txt,
         outputs/figures/cambodia/S1_scatter.png
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from scipy.stats import norm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC, T = ROOT / "data/processed/cambodia", ROOT / "outputs/tables"
RNG = np.random.default_rng(20261101)
z = lambda s: (s - s.mean()) / s.std()

# ---------------- data ----------------
DF = pd.read_csv(PROC / "kh_covariates.csv").merge(pd.read_csv(PROC / "kh_access.csv").rename(columns={"k": "unit"}), on="unit", how="left")
U = gpd.read_file(PROC / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"}).to_crs(32648)
U["unit"] = U.unit.astype("int64")
cent = U.set_index("unit").geometry.centroid
DF["x_km"], DF["y_km"] = DF.unit.map(cent.x) / 1000, DF.unit.map(cent.y) / 1000
DF = DF[~DF.exclude].reset_index(drop=True)
DF["lnpop"] = np.log(DF.pop_2019)
DF["phys"] = z(DF.access_pc1)
DF["conn"] = z(np.log(DF.mob_mbps_shrunk))
DF["conn_tests"] = z(np.log1p(DF.mob_tests_pc))
DF["conn_fixed"] = z(np.log(DF.fix_mbps_shrunk))
DF["ntl"] = z(np.log1p(DF.ntl_pc_2019 * 1000))
comb = DF[["phys", "conn", "conn_tests", "conn_fixed"]]
w_, V = np.linalg.eigh(np.cov(comb.T)); pc = V[:, -1] * np.sign(V[:, -1].sum())
DF["combined"] = z(comb.values @ pc)
DF["prov"] = DF.PARENT.astype(str)
PP_CORE = {116008004, 116008005, 116008007, 116008011}  # pure Phnom Penh khans in GEO2_KH (parent 8 = Phnom Penh + Kandal)
PP_MIXED = {116008001, 116008002}                        # units mixing Phnom Penh khans with Kandal districts
DF["capital"] = DF.unit.isin(PP_CORE)
DF["capital_wide"] = DF.unit.isin(PP_CORE | PP_MIXED)
A = pd.read_csv(PROC / "kh_assign.csv").merge(pd.read_csv(PROC / "kh_projects.csv")[["project_id", "dgroup", "sector", "sector_code", "comm", "startyear", "n_units"]],
                                               on="project_id")
# ICT projects (first DAC sector code 220) are excluded, as in kh_s06-s11, to avoid reverse causality with connectivity
A = A[pd.to_numeric(A.sector_code.astype(str).str.extract(r"(\d{3})")[0], errors="coerce").ne(220)]
D = np.hypot(DF.x_km.values[:, None] - DF.x_km.values[None, :], DF.y_km.values[:, None] - DF.y_km.values[None, :])


def outcome(a, weight="w"):
    return DF.assign(N=DF.unit.map(a.groupby("unit")[weight].sum()).fillna(0))


def ppml(f, d, cov="cluster"):
    m = smf.glm(f, data=d, family=sm.families.Poisson())
    return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300) if cov == "cluster" else m.fit(maxiter=300)


def conley_se(res, d, var, cutoff):
    """Spatial HAC (Bartlett kernel) sandwich for a Poisson GLM: bread from the Hessian, meat from kernel-weighted scores."""
    X = res.model.exog; mu = res.fittedvalues.values; y = res.model.endog
    S = X * (y - mu)[:, None]
    idx = d.index.values
    K = np.clip(1 - D[np.ix_(idx, idx)] / cutoff, 0, None)
    bread = np.linalg.pinv(X.T @ (X * mu[:, None]))
    meat = S.T @ K @ S
    Vc = bread @ meat @ bread
    j = res.model.exog_names.index(var)
    return float(np.sqrt(Vc[j, j]))


def ri_p(d, var, b_obs, f, draws=2000):
    """Freedman-Lane randomization inference: X = fitted(province FE, ln pop) + residual; residuals are permuted within
    province and added back, so the X-population covariance is preserved under the null."""
    aux = smf.ols(f"{var} ~ lnpop + C(prov)", data=d).fit()
    fitted, resid = aux.fittedvalues.values, aux.resid.values; groups = d.prov.values; bs = []
    for _ in range(draws):
        e = resid.copy()
        for g in np.unique(groups):
            ix = np.where(groups == g)[0]; e[ix] = RNG.permutation(resid[ix])
        perm = fitted + e
        try:
            bs.append(smf.glm(f, data=d.assign(**{var: perm}), family=sm.families.Poisson()).fit(maxiter=100).params[var])
        except Exception:
            pass
    return float(np.mean(np.abs(bs) >= abs(b_obs)))


def full(d, var, ri=True):
    f = f"N ~ {var} + lnpop + C(prov)"
    r = ppml(f, d); b = r.params[var]
    se_cl, se_50, se_100 = r.bse[var], conley_se(r, d, var, 50), conley_se(r, d, var, 100)
    p = {"p_cluster": r.pvalues[var], "p_conley50": 2 * norm.sf(abs(b / se_50)), "p_conley100": 2 * norm.sf(abs(b / se_100))}
    if ri:
        p["p_ri"] = ri_p(d, var, b, f)
    se_max = max(se_cl, se_50, se_100)
    sd_within = (d[var] - d.groupby("prov")[var].transform("mean")).std()
    return {"var": var, "b": b, "pct": 100 * (np.exp(b) - 1), "sd_within": sd_within, "pct_per_within_sd": 100 * (np.exp(b * sd_within) - 1),
            "se_cluster": se_cl, "se_conley50": se_50, "se_conley100": se_100, **p,
            "p_conservative": max(p.values()), "mde_pct": 100 * (np.exp(2.8 * se_max) - 1),
            "ci_lo_pct": 100 * (np.exp(b - 1.96 * se_max) - 1), "ci_hi_pct": 100 * (np.exp(b + 1.96 * se_max) - 1)}


def theta(d):
    r = ppml("N ~ lnpop + C(prov)", d)
    t, se = r.params["lnpop"], max(r.bse["lnpop"], conley_se(r, d, "lnpop", 100))
    return {"theta": t, "theta_se": se, "p_theta_eq1": 2 * norm.sf(abs(t - 1) / se)}


def quick(d, vars_=("combined", "phys", "conn", "ntl")):
    o = {}
    for v in vars_:
        r = ppml(f"N ~ {v} + lnpop + C(prov)", d)
        o[v + "_pct"], o[v + "_p"] = 100 * (np.exp(r.params[v]) - 1), r.pvalues[v]
    o.update(theta(d)); o["n_projects"] = round(d.N.sum(), 1)
    return o


base = A[A.level == "adm2"]
d0 = outcome(base)
txt = [f"districts {len(DF)} | projects (district-reliable) {base.project_id.nunique()} | mean N {d0.N.mean():.2f} | zero districts {(d0.N == 0).sum()}",
       f"combined index: loadings {dict(zip(comb.columns, pc.round(2)))}, PC1 share {w_[-1] / w_.sum():.2f}",
       "correlations:\n" + DF[["phys", "conn", "conn_tests", "conn_fixed", "ntl", "lnpop"]].corr().round(2).to_string()]

# ---------------- main table ----------------
M = pd.DataFrame([full(d0, v) for v in ["combined", "phys", "conn", "ntl"]])
dec = M[M["var"].isin(["phys", "conn", "ntl"])].sort_values("p_conservative")
k, run = len(dec), 0
for i, ix in enumerate(dec.index):
    run = max(run, min(1, (k - i) * dec.loc[ix, "p_conservative"])); M.loc[ix, "p_holm"] = run
th = theta(d0)
M.to_csv(T / "kh_main.csv", index=False)
txt.append("MAIN (one construct per model; % per SD; SE: cluster / Conley 50 / Conley 100; RI = within-province permutation):\n" +
           M.round(3).to_string(index=False))
txt.append(f"theta (population only) = {th['theta']:.3f} (SE {th['theta_se']:.3f}; p[theta=1] = {th['p_theta_eq1']:.4f})")

# ---------------- robustness ----------------
RB = [{"spec": "R0 main", **quick(d0)},
      {"spec": "R1a excluding Phnom Penh core khans (4)", **quick(d0[~d0.capital])},
      {"spec": "R1b excluding core + mixed Phnom Penh units (6)", **quick(d0[~d0.capital_wide])},
      {"spec": "R2 binary counting", **quick(outcome(base.assign(one=1.0), "one"))},
      {"spec": "R3 + province-only locations spread", **quick(outcome(A))},
      {"spec": "R4 USD-weighted ($m)", **quick(outcome(base.assign(um=base.comm.fillna(0) * base.w / 1e6), "um"))},
      {"spec": "R5 starts 2017-2019", **quick(outcome(base[base.startyear <= 2019]))},
      {"spec": "R6 projects in <= 5 districts", **quick(outcome(base[base.n_units <= 5]))},
      {"spec": "R9 connectivity = tests per capita", **quick(d0, ("conn_tests",))},
      {"spec": "R9 connectivity = fixed speed", **quick(d0, ("conn_fixed",))}]
nb = {}
for v in ["combined", "phys", "conn", "ntl"]:
    try:
        r = smf.negativebinomial(f"N ~ {v} + lnpop + C(prov)", data=d0.assign(N=np.round(d0.N * 10) / 10)).fit(disp=0, maxiter=300)
        nb[v + "_pct"], nb[v + "_p"] = 100 * (np.exp(r.params[v]) - 1), r.pvalues[v]
    except Exception as e:
        nb[v + "_pct"] = np.nan
RB.append({"spec": "R7 negative binomial", **nb})
hur = {}
for v in ["combined", "phys", "conn", "ntl"]:
    lg = smf.logit(f"any ~ {v} + lnpop", data=d0.assign(any=(d0.N > 0).astype(int))).fit(disp=0)
    pos = d0[d0.N > 0]
    pp = smf.glm(f"N ~ {v} + lnpop", data=pos, family=sm.families.Poisson()).fit(cov_type="HC1")
    hur[v + "_extensive_OR"], hur[v + "_extensive_p"] = np.exp(lg.params[v]), lg.pvalues[v]
    hur[v + "_intensive_pct"], hur[v + "_intensive_p"] = 100 * (np.exp(pp.params[v]) - 1), pp.pvalues[v]
RB.append({"spec": "R8 hurdle (no province FE): any project (logit OR) / count if any (PPML)", **hur})
RB = pd.DataFrame(RB); RB.to_csv(T / "kh_robustness.csv", index=False)
txt.append("ROBUSTNESS:\n" + RB.round(3).to_string(index=False))

# ---------------- heterogeneity ----------------
HET = []
for tag, a in [("Donor: EU/US", base[base.dgroup == "EU/US"]), ("Donor: China", base[base.dgroup == "China"]), ("Donor: World Bank", base[base.dgroup == "World Bank"]),
               ("Sector: Social", base[base.sector.str.contains("Social", na=False)]), ("Sector: Economic infrastructure", base[base.sector.str.contains("Economic", na=False)]),
               ("Sector: Production/other", base[~base.sector.str.contains("Social|Economic", na=False)])]:
    if a.project_id.nunique() >= 25:
        try:
            HET.append({"group": tag, **quick(outcome(a))})
        except Exception as e:
            txt.append(f"{tag} failed: {e}")
HET = pd.DataFrame(HET)
for v in ["combined", "phys", "conn", "ntl"]:
    p = HET[v + "_p"].values; o = np.argsort(p); q = np.empty(len(p)); prev = 1
    for i in range(len(p) - 1, -1, -1):
        prev = min(prev, p[o[i]] * len(p) / (i + 1)); q[o[i]] = prev
    HET[v + "_q_bh"] = q
HET.to_csv(T / "kh_heterogeneity.csv", index=False)
txt.append("HETEROGENEITY (exploratory; BH q within construct):\n" + HET.round(3).to_string(index=False))
txt.append("Phnom Penh check: districts flagged capital = " + ", ".join(DF[DF.capital].ADMIN_NAME.head(15)))
(ROOT / "outputs/results_kh_models.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))

# ---------------- scatter with trend lines ----------------
(ROOT / "outputs/figures/cambodia").mkdir(parents=True, exist_ok=True)
fig, axs = plt.subplots(1, 4, figsize=(8.4, 2.4), dpi=220, sharey=True)
y = np.log1p(d0.N / d0.pop_2019 * 1e5)
lab = {"combined": "Combined access", "phys": "Physical access", "conn": "Connectivity (mobile speed)", "ntl": "Night lights per person"}
for ax, (_, r) in zip(axs, M.iterrows()):
    v = r["var"]
    ax.scatter(d0[v], y, s=9, color="#1f6feb", alpha=0.7, edgecolor="white", linewidth=0.3)
    b = np.polyfit(d0[v], y, 1); xs = np.linspace(d0[v].min(), d0[v].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="#1f2328", lw=1.1)
    ax.set_title(f"{lab[v]}\n{r.pct:+.0f}% per SD (conservative p = {r.p_conservative:.2f})", fontsize=6.4, loc="left", color="#1f2328")
    ax.set_xlabel("SD (within Cambodia)", fontsize=6); ax.tick_params(labelsize=5.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("ln(1 + projects per 100k people)", fontsize=6)
fig.text(0.01, 0.005, "Dots: districts. Line: OLS fit for display; estimates from PPML with province FE and ln population.", fontsize=5.5, color="#6e7781")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(ROOT / "outputs/figures/cambodia/S1_scatter.png"); plt.close(fig)
