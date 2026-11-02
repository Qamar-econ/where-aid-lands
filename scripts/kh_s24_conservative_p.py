"""Most-conservative p-values for the secondary models shown on the site (they were reported with clustered errors only).

Same rule as the main estimates: the largest of province-clustered, Conley 50 km, Conley 100 km and Freedman-Lane
randomisation inference (here 999 draws, with the other regressors in the auxiliary regression).
Models: the four- and five-factor joint models; the Ookla digital-access index alone and holding physical access fixed;
physical access and the digital index on social-sector and on transport/energy projects separately.
Output: outputs/tables/kh_conservative_p.csv
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, cur, outcome, ROOT = ns["DF"], ns["cur"], ns["outcome"], ns["ROOT"]
PROCD, RAW, T = ROOT / "data/processed/cambodia", ROOT / "data/raw/cambodia", ROOT / "outputs/tables"
RNG = np.random.default_rng(20261102)
z = lambda s: (s - s.mean()) / s.std()

U = gpd.read_file(PROCD / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
cen = U.to_crs(32648).geometry.centroid
XY = pd.DataFrame({"x": cen.x.values / 1000, "y": cen.y.values / 1000}, index=U.unit.astype(str))
rw = pd.read_csv(PROCD / "kh_rwi.csv"); rw["unit"] = rw.unit.astype(str); rw = rw.set_index("unit").rwi

# digital-access index exactly as in kh_s08 (four Ookla measures, PC1)
o = pd.concat([pd.read_csv(f) for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv"))]).drop_duplicates(["tx", "ty"])
n16 = 2 ** 16; o["lon"] = (o.tx + .5) / n16 * 360 - 180
o["lat"] = [math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n16)))) for y in o.ty]
og = gpd.sjoin(gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o.lon, o.lat), crs=4326), U[["unit", "geometry"]], predicate="within")
tiles = og.groupby(og.unit.astype(str)).size()


key = lambda s: pd.Series(s).astype(float).astype(int).astype(str).values      # unit codes arrive as int, float or str
XY.index = key(XY.index); rw.index = key(rw.index); tiles.index = key(tiles.index)


def prep(dd):
    dd = dd.copy(); u = pd.Series(key(dd.unit), index=dd.index)
    dd["x_km"], dd["y_km"] = u.map(XY.x).values, u.map(XY.y).values
    dd["need"] = z(-u.map(rw)).values
    area = dd.filter(like="area_km2").iloc[:, 0]
    dd["c_speed"], dd["c_tests"] = z(np.log(dd.mob_mbps_shrunk)), z(np.log1p(dd.mob_tests_pc))
    dd["c_foot"], dd["c_fixed"] = z(np.log1p(u.map(tiles).fillna(0).values / area)), z(np.log(dd.fix_mbps_shrunk))
    cc = dd[["c_speed", "c_tests", "c_foot", "c_fixed"]]
    w_, V_ = np.linalg.eigh(np.cov(cc.T)); pcv = V_[:, -1] * np.sign(V_[:, -1].sum())
    dd["conn_index"] = z(cc.values @ pcv)
    return dd.reset_index(drop=True)


def conley_p(res, dd, var, cutoff):
    X = res.model.exog; mu = res.fittedvalues.values; y = res.model.endog; S = X * (y - mu)[:, None]
    Dm = np.hypot(dd.x_km.values[:, None] - dd.x_km.values[None, :], dd.y_km.values[:, None] - dd.y_km.values[None, :])
    K = np.clip(1 - Dm / cutoff, 0, None); bread = np.linalg.pinv(X.T @ (X * mu[:, None]))
    j = res.model.exog_names.index(var); se = math.sqrt((bread @ (S.T @ K @ S) @ bread)[j, j])
    return 2 * norm.sf(abs(res.params[var] / se))


def ri_p(dd, var, others, f, b_obs, draws=999):
    aux = smf.ols(f"{var} ~ " + " + ".join(others + ["lnpop", "C(prov)"]), data=dd).fit()
    fit_, res_ = aux.fittedvalues.values, aux.resid.values; g = dd.prov.values; bs = []
    for _ in range(draws):
        e = res_.copy()
        for k in np.unique(g):
            ix = np.where(g == k)[0]; e[ix] = RNG.permutation(res_[ix])
        try:
            bs.append(smf.glm(f, data=dd.assign(**{var: fit_ + e}), family=sm.families.Poisson()).fit(maxiter=100).params[var])
        except Exception:
            pass
    return float((1 + np.sum(np.abs(bs) >= abs(b_obs))) / (1 + len(bs)))   # observed draw included (Phipson & Smyth 2010)


rows = []


def run(label, dd, xs, ri=True, key=None):
    f = "N ~ " + " + ".join(xs + ["lnpop", "C(prov)"])
    r = smf.glm(f, data=dd, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]}, maxiter=300)
    for v in xs:
        p = {"p_cluster": r.pvalues[v], "p_conley50": conley_p(r, dd, v, 50), "p_conley100": conley_p(r, dd, v, 100)}
        if ri:
            p["p_ri"] = ri_p(dd, v, [x for x in xs if x != v], f, r.params[v])
        assert not any(np.isnan(x) for x in p.values()), p
        rows.append({"model": label, "var": v, "site_key": (key or {}).get(v, ""), "pct": 100 * (np.exp(r.params[v]) - 1), **p,
                     "p_conservative": max(p.values())})
        print(rows[-1])


# site_key ties a row to the site table it corrects: conn:<kh_ext_C_connectivity model>, infra1:<construct>, ladder:<var>
d = prep(outcome(cur))
run("four factors", d, ["lights", "access", "conn", "prior_infra"], key={v: f"ladder:{v}" for v in ["lights", "access", "conn", "prior_infra"]})
run("five factors", d.dropna(subset=["need"]).reset_index(drop=True), ["lights", "access", "conn", "prior_infra", "need"])
for v, m in [("conn_index", "alone: conn_index"), ("c_tests", "alone: c_tests"), ("c_foot", "alone: c_foot"), ("c_fixed", "alone: c_fixed")]:
    run(f"digital measure alone: {v}", d, [v], key={v: f"conn:{m}"})
run("digital index + physical access", d, ["conn_index", "access"], key={"conn_index": "conn:conn_index controlling physical access"})
for fam, lab in [("social", "social projects"), ("other", "all other projects"), ("infra", "transport/energy projects")]:
    ds = prep(outcome(cur[(cur.family == "social") if fam == "social" else (cur.family != "social") if fam == "other" else (cur.family == "infra")]))
    if fam == "social":
        run(f"{lab}: physical access", ds, ["access"], key={"access": "infra1:access"})
        run(f"{lab}: mobile speed", ds, ["conn"], key={"conn": "infra1:conn"})
        run(f"{lab}: night lights", ds, ["lights"], key={"lights": "infra1:lights"})
        run(f"{lab}: digital index", ds, ["conn_index"], key={"conn_index": "conn:social vs other: social-sector conn effect"})
    elif fam == "other":
        run(f"{lab}: digital index", ds, ["conn_index"], key={"conn_index": "conn:social vs other: other-sector conn effect"})
    else:
        run(f"{lab}: physical access", ds, ["access"])
        run(f"{lab}: digital index", ds, ["conn_index"])
R = pd.DataFrame(rows); R.to_csv(T / "kh_conservative_p.csv", index=False)
print(R[["model", "var", "pct", "p_cluster", "p_conley100", "p_ri", "p_conservative"]].round(3).to_string(index=False))
