"""Boost analyses for Timor-Leste and Nepal: night lights (DMSP 2010 and VIIRS-era 2015) vs network horse race and
Gelbach decomposition of the night-light coefficient (OLS log(1+y), upper-unit FE; bootstrap by upper unit).
Outputs: outputs/tables/boost_lights.csv, outputs/results_boost.txt
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
LOG = open(ROOT / "outputs" / "results_boost.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()


def lights(geoms, path):
    with rasterio.open(path) as s:
        return np.array([mask(s, [g], crop=True, filled=True, nodata=0)[0][0].astype(float).sum() for g in geoms])


# ---- Timor-Leste (61 units excl. Dili urban), outcome as in s10 primary ----
U = gpd.read_file(ROOT / "data/processed/units.gpkg")
X = pd.read_csv(ROOT / "data/processed/unit_project_links.csv"); AS = pd.read_csv(ROOT / "data/processed/oda_assignments.csv")
tier = AS[AS.level == "posto"].groupby("uid").tier.min(); X["tier"] = X.uid.map(tier)
U["y"] = U.unit.map(X[(X.grp == "dev") & (X.precise == 1)].groupby("unit").cnt.sum()).fillna(0)
for yy in [2010, 2015]:
    U[f"ntl{yy}v"] = lights(U.geometry, ROOT / f"data/raw/ntl/ntl_{yy}_TLS.tif")
T = U[U.unit != "TL06U"].copy()
T = T.assign(net=z(T.access_net), phys=z(T.access_phys), dep=z(T.deprivation), lpop=np.log(T.cpop2010), fe=T.mun_pcode)
# ---- Nepal (72 districts excl. valley) ----
N = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")[["k", "geometry"]].merge(pd.read_csv(ROOT / "data/processed/nepal/district_panel.csv"), on="k")
for yy in [2010, 2015]:
    N[f"ntl{yy}v"] = lights(N.geometry, ROOT / f"data/raw/nepal/ntl/ntl_{yy}_NPL.tif")
N = N[N.capital == 0].copy()
N = N.assign(y=N.y_AB, net=z(N.net_census), phys=z(N.phys_idx), dep=z(N.deprivation), lpop=np.log(N.pop2011), fe=N.province)

rows = []
for name, d in [("Timor-Leste", T), ("Nepal", N)]:
    d = d.copy()
    for yy in [2010, 2015]:
        area = d.to_crs(6933).area / 1e6 if hasattr(d, "to_crs") else 1
        d[f"ntl{yy}"] = z(np.log1p(d[f"ntl{yy}v"] / area))
    say("=" * 90 + f"\n{name}: N={len(d)}  (zero-light units: 2010 {int((d.ntl2010v == 0).sum())}, 2015 {int((d.ntl2015v == 0).sum())})")
    d["ly"] = np.log1p(d.y)
    for yy in [2010, 2015]:
        L = f"ntl{yy}"
        for lab, rhs in [("lights only", f"{L}"), ("lights + network", f"{L} + net"), ("lights + network + road", f"{L} + net + phys")]:
            m = smf.glm(f"y ~ {rhs} + dep + lpop + C(fe)", d, family=sm.families.Poisson()).fit(cov_type="HC1")
            say(f"  {yy} {lab:<26} lights={m.params[L]:+.3f}(p={m.pvalues[L]:.3f})" + (f"  network={m.params['net']:+.3f}(p={m.pvalues['net']:.3f})" if "net" in rhs else ""))
            rows.append({"country": name, "lights_year": yy, "model": lab, "b_lights": m.params[L], "p_lights": m.pvalues[L],
                         "b_net": m.params.get("net", np.nan), "p_net": m.pvalues.get("net", np.nan)})
        # Gelbach: how much of the lights coefficient is explained by network (and road)
        base = "dep + lpop + C(fe)"
        b0 = smf.ols(f"ly ~ {L} + {base}", d).fit().params[L]
        full = smf.ols(f"ly ~ {L} + net + phys + {base}", d).fit()
        cn = full.params["net"] * smf.ols(f"net ~ {L} + {base}", d).fit().params[L]
        cp = full.params["phys"] * smf.ols(f"phys ~ {L} + {base}", d).fit().params[L]
        rng = np.random.default_rng(0); fes = d.fe.unique(); bs = []
        for _ in range(400):
            s = pd.concat([d[d.fe == f].assign(fe=f"{f}_{i}") for i, f in enumerate(rng.choice(fes, len(fes)))])
            try:
                fb = smf.ols(f"ly ~ {L} + net + phys + {base}", s).fit()
                bs.append(fb.params["net"] * smf.ols(f"net ~ {L} + {base}", s).fit().params[L])
            except Exception:
                pass
        say(f"  {yy} Gelbach (OLS): lights {b0:+.3f} -> {full.params[L]:+.3f}; network explains {cn:+.3f} ({cn / b0:.0%}, boot SE {np.std(bs):.3f}), road {cp:+.3f} ({cp / b0:.0%})")
        rows.append({"country": name, "lights_year": yy, "model": "Gelbach", "b_lights": b0, "b_net": full.params[L], "share_net": cn / b0,
                     "se_net_contrib": np.std(bs), "share_road": cp / b0})
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/boost_lights.csv", index=False)
LOG.close()
