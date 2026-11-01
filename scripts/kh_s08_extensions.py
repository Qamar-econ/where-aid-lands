"""Cambodia extensions (모형명세서 §9-3; reported as supplementary, added after the first results):
(A) geocoding-bias bounds: province-only located projects assigned to the least / most accessible district of their province;
    precision check (share of district-precise locations vs province access)
(B) stacked event study: first transport/energy project in a district -> later non-infrastructure projects (Cengiz et al. 2019)
(C) connectivity: connectivity-only composite; control-function correction of mobile speed using fixed speed (province block
    bootstrap); connectivity x physical access; social vs other (fully interacted)
Outputs: outputs/tables/kh_ext_*.csv, outputs/results_kh_extensions.txt, outputs/figures/cambodia/S4_event_study.png
"""
import math, warnings
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from scipy.stats import norm, chi2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, K, U, outcome, ROOT, RAW = ns["DF"], ns["L"], ns["K"], ns["U"], ns["outcome"], ns["ROOT"], ns["RAW"]
T = ROOT / "outputs/tables"; RNG = np.random.default_rng(11)
z = lambda s: (s - s.mean()) / s.std()
fitc = lambda f, d, g="prov": smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[g])[0]}, maxiter=300)
out = []

# ======================= (A) geocoding-bias bounds =======================
cur = L[L.start.between(2017, 2020) & (L.family != "ict")]
base = outcome(cur)
K["adm1"] = np.where(K.dgroup == "EU/US", K.precision_crs == 4, K.precision_code == 4)
prov_of = gpd.read_file(ROOT / "data/processed/cambodia/kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
prov_of["unit"] = prov_of.unit.astype("int64")
provs = prov_of.dissolve("PARENT").reset_index()[["PARENT", "geometry"]]
a1 = K[K.startyear.between(2017, 2020) & K.adm1 & (K.family != "ict")]
a1 = gpd.sjoin(gpd.GeoDataFrame(a1, geometry=gpd.points_from_xy(a1.longitude, a1.latitude), crs=4326), provs, predicate="within")
a1 = a1.drop_duplicates(["project_id", "PARENT"])
a1["w"] = 1 / a1.groupby("project_id").PARENT.transform("nunique")
lo = DF.loc[DF.groupby("PARENT").access.idxmin(), ["PARENT", "unit"]].set_index("PARENT").unit
hi = DF.loc[DF.groupby("PARENT").access.idxmax(), ["PARENT", "unit"]].set_index("PARENT").unit
a1["PARENT"] = a1.PARENT.astype(DF.PARENT.dtype)
A_rows = []
for tag, target in [("observed (district-precise only)", None), ("worst case: province-only -> least accessible district", lo),
                    ("best case: province-only -> most accessible district", hi)]:
    d = base.copy()
    if target is not None:
        add = a1.assign(unit=a1.PARENT.map(target)).groupby("unit").w.sum()
        d["N"] = d.N + d.unit.map(add).fillna(0)
    m = fitc("N ~ access + lnpop + C(prov)", d)
    A_rows.append({"scenario": tag, "access_pct": 100 * (np.exp(m.params["access"]) - 1), "p": m.pvalues["access"], "projects": round(d.N.sum(), 1)})
prec = K[K.startyear.between(2017, 2020) & (K.adm1 | K.adm2)]
prec = gpd.sjoin(gpd.GeoDataFrame(prec, geometry=gpd.points_from_xy(prec.longitude, prec.latitude), crs=4326), provs, predicate="within")
ps = prec.groupby("PARENT").adm2.mean().rename("share_precise")
pa = DF.groupby("PARENT").access.mean().rename("access")
pc = pd.concat([ps, pa.rename(index=lambda x: x)], axis=1).dropna()
pc.index = pc.index.astype(str)
pp = pd.concat([ps.rename(index=lambda x: str(int(float(x)))), pa.rename(index=lambda x: str(int(float(x))))], axis=1).dropna()
rho = pp.share_precise.corr(pp.access, method="spearman")
A_rows.append({"scenario": f"precision check: Spearman(share district-precise, province access) over {len(pp)} provinces", "access_pct": np.nan, "p": rho,
               "projects": np.nan})
EA = pd.DataFrame(A_rows); EA.to_csv(T / "kh_ext_A_bounds.csv", index=False)
out.append("(A) GEOCODING-BIAS BOUNDS (physical access, % per SD):\n" + EA.round(3).to_string(index=False) +
           f"\n   province-only projects added: {a1.project_id.nunique()}")

# ======================= (B) stacked event study =======================
L["yr"] = L.start.astype("Int64")
infra_first = L[L.family == "infra"].groupby("unit").yr.min()
panel_units = DF.unit.values
non_infra = L[(L.family != "infra") & (L.family != "ict")]
Y = non_infra.groupby(["unit", "yr"]).w.sum()
stacks = []
for g in range(2008, 2017):
    treated = infra_first[infra_first == g].index
    if len(treated) == 0:
        continue
    clean = [u for u in panel_units if (u not in infra_first.index) or (infra_first[u] > g + 4)]
    for u in list(treated) + clean:
        if u not in panel_units:
            continue
        for t in range(g - 5, g + 5):
            stacks.append({"unit": u, "yr": t, "cohort": g, "treat": int(u in treated), "k": t - g, "N": Y.get((u, t), 0.0)})
S = pd.DataFrame(stacks); S["N"] = pd.to_numeric(S.N, errors="coerce").fillna(0).astype(float)
S["uc"] = S.unit.astype(str) + "_" + S.cohort.astype(str); S["yc"] = S.yr.astype(str) + "_" + S.cohort.astype(str)
for k in range(-5, 5):
    if k != -1:
        S[f"e{k + 5}"] = ((S.k == k) & (S.treat == 1)).astype(int)
ev = [f"e{k + 5}" for k in range(-5, 5) if k != -1]
while True:  # drop fixed-effect groups whose outcome is always zero (they do not identify anything and break PPML)
    n0 = len(S)
    S = S[(S.groupby("uc").N.transform("sum") > 0) & (S.groupby("yc").N.transform("sum") > 0)]
    if len(S) == n0:
        break
ev = [e for e in ev if S.loc[S[e] == 1, "N"].sum() > 0]  # event-time cells with no projects at all would be separated
m = smf.glm("N ~ " + " + ".join(ev) + " + C(uc) + C(yc)", data=S, family=sm.families.Poisson()).fit(
    method="lbfgs", maxiter=5000, cov_type="cluster", cov_kwds={"groups": pd.factorize(S.unit)[0]})  # IRLS is numerically unstable with ~300 FE here
E = pd.DataFrame([{"k": int(e[1:]) - 5, "b": m.params[e], "se": m.bse[e], "p": m.pvalues[e]} for e in ev])
pre = [e for e in ev if int(e[1:]) - 5 < -1]
V = m.cov_params().loc[pre, pre].values; b = m.params[pre].values
wald_p = chi2.sf(float(b @ np.linalg.pinv(V) @ b), len(pre))
post = [e for e in ev if int(e[1:]) - 5 >= 0]
avg_post = m.params[post].mean()
E.to_csv(T / "kh_ext_B_event.csv", index=False)
out.append(f"(B) STACKED EVENT STUDY: cohorts 2008-2016, treated districts {S[S.treat == 1].unit.nunique()}, clean controls "
           f"{S[S.treat == 0].unit.nunique()}, stacked obs {len(S)}\n" + E.round(3).to_string(index=False) +
           f"\n   pre-trend joint Wald p = {wald_p:.3f} | average post effect = {100 * (np.exp(avg_post) - 1):+.0f}%")
fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=220)
EE = pd.concat([E, pd.DataFrame([{"k": -1, "b": 0, "se": 0, "p": 1}])]).sort_values("k")
ax.axhline(0, color="#1f2328", lw=0.7); ax.axvline(-0.5, color="#8c959f", lw=0.7, ls="--")
ax.errorbar(EE.k, 100 * (np.exp(EE.b) - 1), yerr=[100 * (np.exp(EE.b) - np.exp(EE.b - 1.96 * EE.se)), 100 * (np.exp(EE.b + 1.96 * EE.se) - np.exp(EE.b))],
            fmt="o", color="#1f6feb", ms=4, capsize=2, lw=1)
ax.set_xlabel("Years since first transport/energy project in the district", fontsize=7)
ax.set_ylabel("Change in later non-infrastructure\nprojects (%), 95% CI", fontsize=7)
ax.set_title(f"Stacked event study (PPML, district×cohort and year×cohort FE). Pre-trend p = {wald_p:.2f}", fontsize=7, loc="left")
ax.tick_params(labelsize=6)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/cambodia/S4_event_study.png"); plt.close(fig)

# ======================= (C) connectivity =======================
o = pd.concat([pd.read_csv(f) for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv"))]).drop_duplicates(["tx", "ty"])
n16 = 2 ** 16
o["lon"] = (o.tx + .5) / n16 * 360 - 180
o["lat"] = [math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n16)))) for y in o.ty]
og = gpd.sjoin(gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o.lon, o.lat), crs=4326), U[["unit", "geometry"]], predicate="within")
d = base.copy()
d["tiles_km2"] = d.unit.map(og.groupby("unit").size()).fillna(0) / d.filter(like="area_km2").iloc[:, 0]
d["c_speed"], d["c_tests"] = z(np.log(d.mob_mbps_shrunk)), z(np.log1p(d.mob_tests_pc))
d["c_foot"], d["c_fixed"] = z(np.log1p(d.tiles_km2)), z(np.log(d.fix_mbps_shrunk))
cc = d[["c_speed", "c_tests", "c_foot", "c_fixed"]]
w_, V_ = np.linalg.eigh(np.cov(cc.T)); pcv = V_[:, -1] * np.sign(V_[:, -1].sum())
d["conn_index"] = z(cc.values @ pcv)
C_rows = []
for v in ["conn_index", "c_speed", "c_tests", "c_foot", "c_fixed"]:
    mm = fitc(f"N ~ {v} + lnpop + C(prov)", d)
    C_rows.append({"model": f"alone: {v}", "pct": 100 * (np.exp(mm.params[v]) - 1), "p": mm.pvalues[v]})
mm = fitc("N ~ conn_index + access + lnpop + C(prov)", d)
C_rows.append({"model": "conn_index controlling physical access", "pct": 100 * (np.exp(mm.params["conn_index"]) - 1), "p": mm.pvalues["conn_index"]})
# control function: mobile speed instrumented by fixed speed
fs = smf.ols("c_speed ~ c_fixed + lnpop + C(prov)", data=d).fit()
d["cf_resid"] = fs.resid
mcf = smf.glm("N ~ c_speed + cf_resid + lnpop + C(prov)", data=d, family=sm.families.Poisson()).fit(maxiter=300)
bs = []
provs_list = d.prov.unique()
for _ in range(999):
    draw = RNG.choice(provs_list, len(provs_list), replace=True)
    bd = pd.concat([d[d.prov == p].assign(prov=f"{p}_{i}") for i, p in enumerate(draw)], ignore_index=True)
    try:
        r1 = smf.ols("c_speed ~ c_fixed + lnpop + C(prov)", data=bd).fit(); bd["cf_resid"] = r1.resid
        bs.append(smf.glm("N ~ c_speed + cf_resid + lnpop + C(prov)", data=bd, family=sm.families.Poisson()).fit(maxiter=100).params["c_speed"])
    except Exception:
        pass
se_cf = np.nanstd(bs)
C_rows.append({"model": f"control function (mobile speed | fixed speed IV; first-stage F = {fs.tvalues['c_fixed'] ** 2:.1f})",
               "pct": 100 * (np.exp(mcf.params["c_speed"]) - 1), "p": 2 * norm.sf(abs(mcf.params["c_speed"] / se_cf))})
mi = fitc("N ~ conn_index * access + lnpop + C(prov)", d)
C_rows.append({"model": "interaction conn_index x access (negative = substitutes)", "pct": 100 * (np.exp(mi.params["conn_index:access"]) - 1),
               "p": mi.pvalues["conn_index:access"]})
st = []
for s, fam in [(0, "other"), (1, "social")]:
    q = outcome(cur[(cur.family == "social") if fam == "social" else (cur.family != "social")])
    q = q.merge(d[["unit", "conn_index"]], on="unit"); q["soc"] = s; q["pfe"] = q.prov + f"_{s}"; st.append(q)
st = pd.concat(st, ignore_index=True)
ms = fitc("N ~ C(pfe) + conn_index + conn_index:soc + lnpop + lnpop:soc", st)
C_rows.append({"model": "social vs other: other-sector conn effect", "pct": 100 * (np.exp(ms.params["conn_index"]) - 1), "p": ms.pvalues["conn_index"]})
C_rows.append({"model": "social vs other: social-sector conn effect", "pct": 100 * (np.exp(ms.params["conn_index"] + ms.params["conn_index:soc"]) - 1),
               "p": ms.pvalues["conn_index:soc"]})
EC = pd.DataFrame(C_rows); EC.to_csv(T / "kh_ext_C_connectivity.csv", index=False)
out.append(f"(C) CONNECTIVITY (connectivity-only PC1 loadings {dict(zip(cc.columns, pcv.round(2)))}, share {w_[-1] / w_.sum():.2f}):\n" + EC.round(3).to_string(index=False) +
           "\n   (for the 'social-sector' row, p refers to the difference from other sectors)")
(ROOT / "outputs/results_kh_extensions.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
