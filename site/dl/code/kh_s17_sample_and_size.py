"""(1) Who gets located? Activities starting 2017-2020 located at district level vs only at province or country level:
        composition (donor group, sector, flow), size, description length, and the road access of the province they are in.
    (2) Counts or bigger projects? Total commitments M_d = N_d x S_d (number x average size), on the main sample.
        District level: count model (beta_N), amount model (beta_M), average size = amount model with offset ln N_d among
        districts with projects (beta_S). Activity level: commitment of each activity on the mean road access of its districts,
        with donor-group and sector fixed effects.
Outputs: outputs/tables/kh_sample_located.csv, kh_size_decomposition.csv
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])          # DF (main sample districts), A, outcome(), ppml()
sys.path.insert(0, str(ROOT / "scripts"))
from kh_common import activityize, assign_units
import geopandas as gpd

PR = pd.read_csv(PROC / "kh_projects.csv")
K = activityize(pd.read_csv(ROOT / "data/raw/cambodia/godad_khm.csv", low_memory=False))
txt = K.drop_duplicates("record_id").assign(n=lambda d: d.description.fillna("").str.len()).groupby("project_id").n.max()
sec3 = pd.to_numeric(PR.sector_code.astype(str).str.extract(r"(\d{3})")[0], errors="coerce")
PR["family"] = np.select([sec3.isin([210, 230]), sec3.between(110, 169), sec3.between(310, 339)], ["infrastructure", "social", "production"], "other")
PR["located"] = PR.best.eq("adm2")
PR["text_len"] = PR.project_id.map(txt)
PR["usd_m"] = PR.comm / 1e6

# province of every activity (from any located point) and that province's population-weighted road access
UG = gpd.read_file(PROC / "kh_units.gpkg")[["GEOLEVEL2", "PARENT", "geometry"]].rename(columns={"GEOLEVEL2": "unit"})
UG["unit"] = UG.unit.astype("int64")
W = assign_units(K[K.startyear.between(2017, 2020)], UG)
prov_of = W.dropna(subset=["PARENT"]).assign(PARENT=lambda d: d.PARENT.astype(float).astype(int)).groupby("project_id").PARENT.agg(lambda s: s.mode().iloc[0])
pacc = DF.assign(w=DF.pop_2019, PARENT=DF.PARENT.astype(float).astype(int)).groupby("PARENT").apply(lambda d: np.average(d.phys, weights=d.w), include_groups=False)
PR["prov_access"] = PR.project_id.map(prov_of).map(pacc)

rows = []
for col in ["dgroup", "family", "flow"]:
    ct = pd.crosstab(PR[col].fillna("NA"), PR.located)
    p = chi2_contingency(ct)[1]
    for k in ct.index:
        rows.append({"check": col, "level": k, "located_n": int(ct.loc[k, True]), "not_located_n": int(ct.loc[k, False]),
                     "located_share_of_level": ct.loc[k, True] / ct.loc[k].sum(), "test_p": p})
for col, lab in [("usd_m", "commitment, US$ million (median)"), ("text_len", "description length, characters (median)"),
                 ("prov_access", "road access of the activity's province (mean, SD units)")]:
    a, b = PR.loc[PR.located, col].dropna(), PR.loc[~PR.located, col].dropna()
    stat = (lambda s: s.mean()) if col == "prov_access" else (lambda s: s.median())
    rows.append({"check": lab, "level": "located vs not", "located_n": len(a), "not_located_n": len(b),
                 "located_value": stat(a), "not_located_value": stat(b), "test_p": mannwhitneyu(a, b).pvalue if len(b) > 2 else np.nan})
S = pd.DataFrame(rows)
S.to_csv(T / "kh_sample_located.csv", index=False)
print("activities 2017-2020:", len(PR), "| located at district level:", int(PR.located.sum()))
print(S.round(3).to_string(index=False))

# ---------- (2) size decomposition ----------
base = A[A.level == "adm2"].merge(PR[["project_id", "usd_m"]], on="project_id", how="left")
d = outcome(base).assign(M=lambda x: x.unit.map((base.w * base.usd_m.fillna(0)).groupby(base.unit).sum()).fillna(0))
out = []
for v in ["phys", "conn", "ntl"]:
    rN = ppml(f"N ~ {v} + lnpop + C(prov)", d)
    rM = ppml(f"M ~ {v} + lnpop + C(prov)", d)
    dp = d[d.N > 0].copy()
    rS = smf.glm(f"M ~ {v} + lnpop + C(prov)", data=dp, family=sm.families.Poisson(), offset=np.log(dp.N)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dp.prov)[0]}, maxiter=300)
    for tag, r in [("count N", rN), ("amount M", rM), ("average size M/N (districts with projects)", rS)]:
        out.append({"construct": v, "outcome": tag, "pct": 100 * (np.exp(r.params[v]) - 1), "p": r.pvalues[v], "n": int(r.nobs)})
# activity level: does a bigger project sit in better-connected districts?
acc = base.merge(DF[["unit", "phys"]], on="unit").groupby("project_id").apply(lambda g: np.average(g.phys, weights=g.w), include_groups=False).rename("act_access")
AP = PR.set_index("project_id").join(acc, how="inner").dropna(subset=["usd_m"])
AP = AP[AP.usd_m > 0]
rA = smf.glm("usd_m ~ act_access + C(dgroup) + C(family)", data=AP.reset_index(), family=sm.families.Poisson()).fit(cov_type="HC1")
rL = smf.ols("np.log(usd_m) ~ act_access + C(dgroup) + C(family)", data=AP.reset_index()).fit(cov_type="HC1")
out.append({"construct": "phys", "outcome": "activity commitment (PPML, donor + sector FE)", "pct": 100 * (np.exp(rA.params["act_access"]) - 1), "p": rA.pvalues["act_access"], "n": int(rA.nobs)})
out.append({"construct": "phys", "outcome": "activity log commitment (OLS, donor + sector FE)", "pct": 100 * (np.exp(rL.params["act_access"]) - 1), "p": rL.pvalues["act_access"], "n": int(rL.nobs)})
O = pd.DataFrame(out)
O.to_csv(T / "kh_size_decomposition.csv", index=False)
print("\n" + O.round(3).to_string(index=False))
top = AP.usd_m.sort_values(ascending=False)
print(f"\nlargest 5 activities hold {top.head(5).sum() / top.sum():.0%} of located commitments; median {top.median():.2f} m, mean {top.mean():.1f} m")
