"""Cambodia model ladder: one model per question from the literature, then a 'race' model; H-infra2 is the focal hypothesis.

Outcome (two tracks): non-ICT development projects starting 2017-2020 with district-reliable GODAD locations
  N_d  = number of projects (split equally over a project's districts)            -> where donors place projects
  M_d  = commitments in US$ million (2014 constant, split equally)                 -> where money goes
All models: PPML with province fixed effects and ln population; one row per model:

  M0 Scale          ln Pop only                       -> theta (small-country bias, Alesina & Dollar 2000, subnational analogue)
  M1 Wealth / need  night lights per person           -> Briggs (2017, 2018): does aid go to richer (lit) places?
  M2 Access         physical-access index (OSM)       -> Chambers (1983) tarmac bias; Briggs (2021) implementability
  M3 Connectivity   mobile speed (Ookla, shrunk)      -> new: digital convenience
  M4 Prior infra    any transport/energy project 2010-2016 in the district  -> H-infra2: does infrastructure attract later aid?
  M5 Race           M1-M4 together                    -> which survives when all compete (VIF reported)

H-infra2 checks: (a) controlling for access and lights; (b) prior infrastructure vs prior SOCIAL projects (is it infrastructure,
or just donors returning to familiar districts?); (c) Phnom Penh excluded; (d) prior infra as a count; (e) randomization inference.
Fit: deviance explained over M0 (pseudo-R2 increment) and 5-fold cross-validated Poisson deviance (10 repeats).
Simple table: OLS of ln(1 + projects per 100k) on each regressor + province dummies (robust SE) for intuition.
Outputs: outputs/tables/kh_ladder.csv, kh_infra2.csv, kh_ols_simple.csv, outputs/results_kh_ladder.txt, figures/cambodia/S2_ladder.png
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from scipy.stats import norm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, T = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia", ROOT / "outputs/tables"
RNG = np.random.default_rng(7)
z = lambda s: (s - s.mean()) / s.std()

# ---------------- districts ----------------
DF = pd.read_csv(PROC / "kh_covariates.csv").merge(pd.read_csv(PROC / "kh_access.csv").rename(columns={"k": "unit"}), on="unit", how="left")
U = gpd.read_file(PROC / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
U["unit"] = U.unit.astype("int64")  # gpkg stores the code as text; covariate tables use integers
DF = DF[~DF.exclude].reset_index(drop=True)
DF["lnpop"] = np.log(DF.pop_2019); DF["prov"] = DF.PARENT.astype(str)
DF["lights"] = z(np.log1p(DF.ntl_pc_2015 * 1000)); DF["access"] = z(DF.access_pc1); DF["conn"] = z(np.log(DF.mob_mbps_shrunk))  # lights: pre-period 2015 (spec 9-8)
DF["capital"] = DF.unit.isin({116008004, 116008005, 116008007, 116008011})  # pure Phnom Penh khans (GEO2_KH parent 8 = PP + Kandal)

# ---------------- projects (all GODAD years, district-reliable) ----------------
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else ROOT / "scripts"))
from kh_common import activityize, assign_units
K = activityize(pd.read_csv(RAW / "godad_khm.csv", low_memory=False))   # activities, dated by first commitment (spec 9-5, 9-7)
K["adm2"] = np.where(K.dgroup == "EU/US", K.precision_crs == 3, K.precision_code <= 3)
K["sec3"] = pd.to_numeric(K.sector_codes.astype(str).str.extract(r"(\d{3})")[0], errors="coerce")
K["family"] = np.select([K.sec3.isin([210, 230]), K.sec3.eq(220), K.sec3.between(110, 169), K.sec3.between(310, 339)],
                        ["infra", "ict", "social", "production"], "other")
g = assign_units(K[K.adm2], U[["unit", "geometry"]]).dropna(subset=["unit"])
g["unit"] = g.unit.astype("int64")
L = g.groupby(["project_id", "unit"]).agg(start=("startyear", "first"), family=("family", "first"), comm=("comm", "first"), dgroup=("dgroup", "first")).reset_index()
L["w"] = 1 / L.groupby("project_id").unit.transform("nunique")
prior = L[L.start.between(2010, 2016)]
DF["prior_infra"] = DF.unit.isin(prior[prior.family == "infra"].unit).astype(int)
DF["prior_infra_n"] = np.log1p(DF.unit.map(prior[prior.family == "infra"].groupby("unit").w.sum()).fillna(0))
DF["prior_social"] = DF.unit.isin(prior[prior.family == "social"].unit).astype(int)
from kh_common import reclassify_geocodes
cur = reclassify_geocodes(L[L.start.between(2017, 2020) & (L.family != "ict")], DF, ROOT / "data/processed/cambodia")  # spec 9-26


def outcome(a):
    return DF.assign(N=DF.unit.map(a.groupby("unit").w.sum()).fillna(0),
                    M=DF.unit.map((a.w * a.comm.fillna(0) / 1e6).groupby(a.unit).sum()).fillna(0))


def fit(f, d):
    return smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)


def dev(f, d):
    return smf.glm(f, data=d, family=sm.families.Poisson()).fit(maxiter=300).deviance


def cv_dev(f, d, y, reps=10, k=5):
    out = []
    for r in range(reps):
        fold = np.empty(len(d), int)  # stratified by province so every province is in each training set
        for pv, ix in d.groupby("prov").indices.items():
            fold[ix] = RNG.permutation(np.arange(len(ix)) % k) if len(ix) >= 2 else -1  # singleton provinces never tested
        s = 0
        for j in range(k):
            tr, te = d[fold != j], d[fold == j]
            te = te[te.prov.isin(tr.prov)]
            try:
                m = smf.glm(f, data=tr, family=sm.families.Poisson()).fit(maxiter=200)
                mu = np.clip(m.predict(te), 1e-9, None); yy = te[y].values
                s += 2 * np.sum(np.where(yy > 0, yy * np.log(np.where(yy > 0, yy, 1) / mu), 0) - (yy - mu))
            except Exception:
                s += np.nan
        out.append(s)
    return float(np.nanmean(out))


LADDER = [("M0 Scale (theta)", []), ("M1 Wealth/need: night lights", ["lights"]), ("M2 Access: physical access", ["access"]),
          ("M3 Connectivity: mobile speed", ["conn"]), ("M4 Prior infrastructure (H-infra2)", ["prior_infra"]),
          ("M5 Race: all together", ["lights", "access", "conn", "prior_infra"])]
d = outcome(cur)
rows = []
for y in ["N", "M"]:
    base_f = f"{y} ~ lnpop + C(prov)"
    D0, Dnull = dev(base_f, d), dev(f"{y} ~ 1", d)
    cv0 = cv_dev(base_f, d, y)
    for name, xs in LADDER:
        f = f"{y} ~ " + " + ".join(xs + ["lnpop", "C(prov)"])
        m = fit(f, d); Dm = dev(f, d)
        r = {"track": "count" if y == "N" else "amount ($m)", "model": name, "theta": m.params["lnpop"],
             "p_theta_eq1": 2 * norm.sf(abs(m.params["lnpop"] - 1) / m.bse["lnpop"]),
             "pseudoR2": 1 - Dm / Dnull, "dR2_over_M0": (D0 - Dm) / Dnull, "cv_dev_gain_pct": 100 * (cv0 - cv_dev(f, d, y)) / cv0}
        for x in xs:
            r[x + "_pct"], r[x + "_p"] = 100 * (np.exp(m.params[x]) - 1), m.pvalues[x]
        rows.append(r)
LAD = pd.DataFrame(rows); LAD.to_csv(T / "kh_ladder.csv", index=False)
vif = np.diag(np.linalg.inv(np.corrcoef(d[["lights", "access", "conn", "prior_infra", "lnpop"]].values.T)))

# ---------------- H-infra2 deep dive ----------------
non_infra = cur[cur.family != "infra"]
dn = outcome(non_infra)
H = []
def h(tag, f, dd, var="prior_infra"):
    m = fit(f, dd)
    H.append({"spec": tag, "pct": 100 * (np.exp(m.params[var]) - 1), "p": m.pvalues[var], "n_projects": round(dd.N.sum(), 1)})
h("a0 non-infra projects ~ prior infra + ln pop + province", "N ~ prior_infra + lnpop + C(prov)", dn)
h("a1 + physical access", "N ~ prior_infra + access + lnpop + C(prov)", dn)
h("a2 + access + night lights", "N ~ prior_infra + access + lights + lnpop + C(prov)", dn)
h("b1 prior SOCIAL projects instead (placebo-type comparison)", "N ~ prior_social + lnpop + C(prov)", dn, "prior_social")
m = fit("N ~ prior_infra + prior_social + lnpop + C(prov)", dn)
H.append({"spec": "b2 both together: prior infra", "pct": 100 * (np.exp(m.params["prior_infra"]) - 1), "p": m.pvalues["prior_infra"], "n_projects": round(dn.N.sum(), 1)})
H.append({"spec": "b2 both together: prior social", "pct": 100 * (np.exp(m.params["prior_social"]) - 1), "p": m.pvalues["prior_social"], "n_projects": round(dn.N.sum(), 1)})
h("c  excluding Phnom Penh", "N ~ prior_infra + lnpop + C(prov)", dn[~dn.capital])
h("d  prior infra as log count", "N ~ prior_infra_n + lnpop + C(prov)", dn, "prior_infra_n")
h("e  amount track ($m)", "M ~ prior_infra + lnpop + C(prov)", dn.assign(N=dn.M).assign(M=dn.M))
b_obs = fit("N ~ prior_infra + lnpop + C(prov)", dn).params["prior_infra"]; bs = []
for _ in range(2000):
    perm = dn.prior_infra.values.copy()
    for pv in dn.prov.unique():
        ix = np.where(dn.prov.values == pv)[0]; perm[ix] = RNG.permutation(perm[ix])
    bs.append(smf.glm("N ~ prior_infra + lnpop + C(prov)", data=dn.assign(prior_infra=perm), family=sm.families.Poisson()).fit(maxiter=100).params["prior_infra"])
H.append({"spec": "f  randomization inference p (a0)", "pct": np.nan, "p": float((1 + np.sum(np.abs(np.array(bs)) >= abs(b_obs))) / (1 + len(bs))), "n_projects": np.nan})
HI = pd.DataFrame(H); HI.to_csv(T / "kh_infra2.csv", index=False)

# ---------------- H-infra1: fully interacted stacked model (infrastructure vs social projects) ----------------
H1 = []
for x in ["access", "conn", "lights"]:
    st = []
    for s, fam in [(0, "social"), (1, "infra")]:
        q = outcome(cur[cur.family == fam]); q["infra"] = s; q["pfe"] = q.prov + f"_{s}"; st.append(q)
    st = pd.concat(st, ignore_index=True)
    m = smf.glm(f"N ~ C(pfe) + {x} + {x}:infra + lnpop + lnpop:infra", data=st, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(st.prov)[0]}, maxiter=300)
    H1.append({"construct": x, "social_pct": 100 * (np.exp(m.params[x]) - 1), "social_p": m.pvalues[x],
               "infra_pct": 100 * (np.exp(m.params[x] + m.params[f"{x}:infra"]) - 1),
               "difference_b": m.params[f"{x}:infra"], "difference_p": m.pvalues[f"{x}:infra"],
               "n_social": int(cur[cur.family == "social"].project_id.nunique()), "n_infra": int(cur[cur.family == "infra"].project_id.nunique())})
H1 = pd.DataFrame(H1); H1.to_csv(T / "kh_infra1.csv", index=False)

# ---------------- simple OLS (intuition table) ----------------
d["y_pc"] = np.log1p(d.N / d.pop_2019 * 1e5)
O = []
for x in ["lights", "access", "conn", "prior_infra"]:
    m = smf.ols(f"y_pc ~ {x} + C(prov)", data=d).fit(cov_type="HC1")
    O.append({"regressor": x, "b": m.params[x], "p": m.pvalues[x], "R2": m.rsquared})
O = pd.DataFrame(O); O.to_csv(T / "kh_ols_simple.csv", index=False)

out = [f"districts {len(DF)} | current non-ICT projects 2017-2020: {cur.project_id.nunique()} (non-infra {non_infra.project_id.nunique()}) | "
       f"districts with prior infra (2010-16): {DF.prior_infra.sum()} | with prior social: {DF.prior_social.sum()}",
       f"VIF (race model): {dict(zip(['lights', 'access', 'conn', 'prior_infra', 'lnpop'], vif.round(2)))}",
       "LADDER (% per SD; prior_infra = % more for districts with prior infrastructure):\n" + LAD.round(3).to_string(index=False),
       "H-INFRA2:\n" + HI.round(3).to_string(index=False),
       "H-INFRA1 (fully interacted stacked model; infra vs social):\n" + H1.round(3).to_string(index=False), "SIMPLE OLS (ln(1+projects per 100k), province dummies):\n" + O.round(3).to_string(index=False)]
(ROOT / "outputs/figures/cambodia").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/results_kh_ladder.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))

# ---------------- figure: ladder coefficients + explanatory power ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), dpi=220, gridspec_kw={"width_ratios": [1.6, 1]})
lab = {"lights": "Night lights (wealth)", "access": "Physical access", "conn": "Connectivity", "prior_infra": "Prior infrastructure"}
cnt = LAD[LAD.track == "count"]
for i, x in enumerate(lab):
    single = cnt[cnt.model.str.startswith(("M1", "M2", "M3", "M4")) & cnt[x + "_pct"].notna() & ~cnt.model.str.startswith("M5")].iloc[0]
    race = cnt[cnt.model.str.startswith("M5")].iloc[0]
    for off, r, alpha in [(-0.15, single, 1.0), (0.15, race, 0.45)]:
        col = "#1f6feb" if r[x + "_p"] < 0.05 else "#8c959f"
        axs[0].barh(i + off, r[x + "_pct"], height=0.28, color=col, alpha=alpha)
axs[0].set_yticks(range(len(lab))); axs[0].set_yticklabels(list(lab.values()), fontsize=6.5); axs[0].invert_yaxis()
axs[0].axvline(0, color="#1f2328", lw=0.7); axs[0].set_xlabel("% more projects (per SD / with prior infra)", fontsize=6.5)
axs[0].set_title("Coefficients: alone (solid) vs race model (faded); blue p<0.05", fontsize=7, loc="left")
axs[1].barh(range(5), cnt[cnt.model.str.startswith(("M1", "M2", "M3", "M4", "M5"))].dR2_over_M0.values * 100, color="#1f2328")
axs[1].set_yticks(range(5)); axs[1].set_yticklabels(["M1 lights", "M2 access", "M3 conn.", "M4 prior infra", "M5 race"], fontsize=6.5); axs[1].invert_yaxis()
axs[1].set_xlabel("extra deviance explained over M0 (pp)", fontsize=6.5); axs[1].set_title("Explanatory power", fontsize=7, loc="left")
for ax in axs:
    ax.tick_params(labelsize=6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/cambodia/S2_ladder.png"); plt.close(fig)
