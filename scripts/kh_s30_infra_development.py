"""Does infrastructure aid come before economic development? (exploratory, added after the first results)

The 'develop the reachable places first, then expand' account needs a middle link: districts that receive a transport or
energy project should afterwards grow faster. Two outcomes from annual rasters (kh_fetch_panel.py):
  lights  ln(1 + sum of harmonised night-light DN in the district)   (DMSP to 2013, simulated VIIRS from 2014)
  pop     ln(WorldPop population)
(A) Stacked event study with the same treatment, cohorts and clean controls as the aid event study in kh_s08: first transport
    or energy project starting in year g (2008-2016) vs districts with none up to g+4; OLS with district-by-cohort and
    year-by-cohort fixed effects, errors clustered by district. The 2013/14 sensor change is common to all districts in a
    year and so absorbed by the year effects.
(B) Long difference 2008 -> 2019 on 'any transport/energy project starting 2010-2016', with province fixed effects and the
    2008 level, clustered by province.
Output: outputs/tables/kh_infra_development_event.csv, kh_infra_development.txt
"""
import numpy as np, pandas as pd, geopandas as gpd, rasterio, statsmodels.formula.api as smf
from rasterio.mask import mask
from scipy.stats import chi2
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, ROOT = ns["DF"], ns["L"], ns["ROOT"]
RAW, T = ROOT / "data/raw/cambodia", ROOT / "outputs/tables"
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

U = gpd.read_file(ROOT / "data/processed/cambodia/kh_units.gpkg"); U["unit"] = U.GEOLEVEL2.astype("int64")
U = U[U.unit.isin(DF.unit)]


def zsum(path):
    with rasterio.open(path) as s:
        g = U.to_crs(s.crs); out = []
        for geom in g.geometry:
            a = mask(s, [geom], crop=True, filled=True, nodata=0)[0][0].astype(float); a[a < 0] = 0; out.append(a.sum())
    return pd.Series(out, index=U.unit.values)


years = range(2003, 2021)
P = []
for y in years:
    ln, pp = RAW / f"ntl/ntl_{y}_KHM.tif", RAW / f"worldpop/khm_ppp_{y}_1km.tif"
    if ln.exists() and pp.exists():
        P.append(pd.DataFrame({"unit": U.unit.values, "yr": y, "lights": np.log1p(zsum(ln).values), "pop": np.log(zsum(pp).values)}))
P = pd.concat(P, ignore_index=True)
say(f"panel: {P.unit.nunique()} districts x years {P.yr.min()}-{P.yr.max()} ({P.yr.nunique()} years)")
V = P.set_index(["unit", "yr"])

# ---------- (A) stacked event study ----------
L["yr"] = L.start.astype("Int64")
infra_first = L[L.family == "infra"].groupby("unit").yr.min()
rows = []
for g in range(2008, 2017):
    treated = infra_first[infra_first == g].index
    clean = [u for u in DF.unit if (u not in infra_first.index) or (infra_first[u] > g + 4)]
    for u in [u for u in treated if u in set(DF.unit)] + clean:
        for t in range(g - 5, g + 5):
            if (u, t) in V.index:
                rows.append({"unit": u, "yr": t, "cohort": g, "treat": int(u in treated), "k": t - g, **V.loc[(u, t)].to_dict()})
S = pd.DataFrame(rows)
S["uc"] = S.unit.astype(str) + "_" + S.cohort.astype(str); S["yc"] = S.yr.astype(str) + "_" + S.cohort.astype(str)
ev = []
for k in range(-5, 5):
    if k != -1:
        S[f"e{k + 5}"] = ((S.k == k) & (S.treat == 1)).astype(int); ev.append(f"e{k + 5}")
say(f"stacked event study: treated {S[S.treat == 1].unit.nunique()}, clean controls {S[S.treat == 0].unit.nunique()}, obs {len(S)}")
def absorb(frame, cols, fes=("uc", "yc"), tol=1e-10):
    """Remove the two sets of fixed effects by alternating projections (with ~800 dummy columns and 131 clusters the
    dummy-variable sandwich is rank deficient, so the fixed effects are partialled out instead)."""
    X = frame[cols].astype(float).copy()
    for _ in range(1000):
        old = X.values.copy()
        for f in fes:
            X = X - X.groupby(frame[f]).transform("mean")
        if np.max(np.abs(X.values - old)) < tol:
            break
    return X


E = []
for y in ["lights", "pop"]:
    Xd = absorb(S, [y] + ev)
    m = smf.ols(f"{y} ~ 0 + " + " + ".join(ev), data=Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.unit)[0]})
    pre = [e for e in ev if int(e[1:]) - 5 < -1]; post = [e for e in ev if int(e[1:]) - 5 >= 0]
    Vp = m.cov_params().loc[pre, pre].values; bp = m.params[pre].values
    wp = chi2.sf(float(bp @ np.linalg.pinv(Vp) @ bp), len(pre))
    w = np.ones(len(post)) / len(post); avg = float(m.params[post] @ w); se = float(np.sqrt(w @ m.cov_params().loc[post, post].values @ w))
    for e in ev:
        E.append({"outcome": y, "k": int(e[1:]) - 5, "b": m.params[e], "se": m.bse[e], "p": m.pvalues[e]})
    say(f"  {y:<6s} pre-trend joint p = {wp:.3f} | average post effect = {100 * (np.exp(avg) - 1):+.1f}% "
        f"(95% CI {100 * (np.exp(avg - 1.96 * se) - 1):+.1f}% to {100 * (np.exp(avg + 1.96 * se) - 1):+.1f}%)")
pd.DataFrame(E).to_csv(T / "kh_infra_development_event.csv", index=False)

# ---------- (B) long difference 2008 -> 2019 ----------
W = P.pivot(index="unit", columns="yr")
d = DF.set_index("unit")[["prov", "prior_infra", "access"]].join(pd.DataFrame({
    "dl": W["lights"][2019] - W["lights"][2008], "dp": W["pop"][2019] - W["pop"][2008],
    "l08": W["lights"][2008], "p08": W["pop"][2008]})).dropna().reset_index()
say("long difference 2008 -> 2019, any transport/energy project starting 2010-2016 (province FE, clustered by province):")
for y, lev, lab in [("dl", "l08", "night lights"), ("dp", "p08", "population")]:
    for f, tag in [(f"{y} ~ prior_infra + {lev} + C(prov)", "with 2008 level"), (f"{y} ~ prior_infra + {lev} + access + C(prov)", "+ physical access")]:
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]})
        b, se = m.params["prior_infra"], m.bse["prior_infra"]
        say(f"  {lab:<12s} {tag:<17s} {100 * (np.exp(b) - 1):+6.1f}% more growth (95% CI {100 * (np.exp(b - 1.96 * se) - 1):+.1f}% to "
            f"{100 * (np.exp(b + 1.96 * se) - 1):+.1f}%), p = {m.pvalues['prior_infra']:.3f}, n = {len(d)}")
# energy projects (grid, generation) can add light by themselves, so transport and energy are separated
K = ns["K"]; sec = K.groupby("project_id").sec3.first()
L["sec3"] = L.project_id.map(sec)
pr = L[L.start.between(2010, 2016)]
d["prior_transport"] = d.unit.isin(pr[pr.sec3 == 210].unit).astype(int)
d["prior_energy"] = d.unit.isin(pr[pr.sec3 == 230].unit).astype(int)
say(f"  districts with a 2010-16 project: transport {d.prior_transport.sum()}, energy {d.prior_energy.sum()}, both {(d.prior_transport & d.prior_energy).sum()}")
for y, lev, lab in [("dl", "l08", "night lights"), ("dp", "p08", "population")]:
    m = smf.ols(f"{y} ~ prior_transport + prior_energy + {lev} + access + C(prov)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]})
    say("  " + lab + ", transport and energy together: " + " | ".join(
        f"{v.replace('prior_', '')} {100 * (np.exp(m.params[v]) - 1):+.1f}% (95% CI {100 * (np.exp(m.params[v] - 1.96 * m.bse[v]) - 1):+.1f}% to "
        f"{100 * (np.exp(m.params[v] + 1.96 * m.bse[v]) - 1):+.1f}%, p {m.pvalues[v]:.3f})" for v in ["prior_transport", "prior_energy"]))
(T / "kh_infra_development.txt").write_text("\n".join(log), encoding="utf-8")
