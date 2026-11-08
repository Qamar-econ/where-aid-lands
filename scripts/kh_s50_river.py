"""Spec 9-39 (exploratory): river barriers as a natural source of road detour.

Barrier  river water polygons in the 2017-01-01 OSM extract (waterway=riverbank; natural=water + water=river/oxbow/canal),
         dissolved, connected components >= 5 km2; variant adds the Tonle Sap lake
Treatment cross = share of a district's population (1 km cells) whose straight line to the nearest town of 20,000+ crosses
         the barrier (towns and distances exactly as kh_s46 B)
First stage  detour (log2 min/km) ~ cross + straight-line distance + ln pop + province FE (OLS, province-clustered F)
Reduced form PPML projects ~ cross + distance + ln pop + province FE; Freedman-Lane randomisation inference within province
Samples  all districts, rural districts (no hub), riverine band (population-weighted centre within 20 km of the barrier)
Output: outputs/tables/kh_river.csv, .txt; data/interim/kh_river_barrier.gpkg; data/processed/cambodia/kh_river_cells.csv
"""
import sys, numpy as np, pandas as pd, geopandas as gpd, shapely, osmium, shapely.wkb as wkblib
import statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
src = (HERE / "kh_s46_designs.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_s46_designs.py"), "__name__": "kh_s46_ns"}
sys.argv = [sys.argv[0], "0"]          # rebuild kh_s46's data (towns, cells, distance, detour); its draws are skipped
exec(src[:src.index("Mc = M[M.type.isin")].replace("for _ in range(DRAWS):", "for _ in range(0):"), ns)
M0 = ns["ns"]; M0["RNG"] = np.random.default_rng(20261115)
DF, MAIN, counts, T, CE, tw, cl_ = ns["DF"], ns["MAIN"], ns["counts"], ns["T"], ns["C"], ns["tw"], ns["cl_"]
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
pc = lambda b: 100 * (np.exp(b) - 1)

# ---------------- barrier ----------------
CACHE = ROOT / "data/interim/kh_river_barrier.gpkg"
if not CACHE.exists():
    fab = osmium.geom.WKBFactory(); geoms, kinds = [], []

    class H(osmium.SimpleHandler):
        def area(self, a):
            t = a.tags
            river = t.get("waterway") == "riverbank" or (t.get("natural") == "water" and t.get("water") in ("river", "oxbow", "canal"))
            lake = (t.get("name") or "").endswith("Tonle Sap") or t.get("name:en") == "Tonle Sap Lake" or "Boeung Tonle Sap" in (t.get("name") or "") + (t.get("name:en") or "")
            if river or (lake and t.get("natural") == "water"):
                try:
                    geoms.append(wkblib.loads(fab.create_multipolygon(a), hex=True)); kinds.append("river" if river else "lake")
                except Exception:
                    pass
    H().apply_file(str(ROOT / "data/raw/cambodia/osm/cambodia-170101.osm.pbf"), locations=True, idx="flex_mem")
    W = gpd.GeoDataFrame({"kind": kinds}, geometry=geoms, crs=4326).to_crs(32648)
    W["geometry"] = W.buffer(0)
    R = W[W.kind == "river"].dissolve().explode(index_parts=False).reset_index(drop=True); R["km2"] = R.area / 1e6
    R = R[R.km2 >= 5].assign(kind="river")
    L = W[W.kind == "lake"].dissolve().assign(kind="lake"); L["km2"] = L.area / 1e6
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([R, L])[["kind", "km2", "geometry"]].to_file(CACHE, driver="GPKG")
B = gpd.read_file(CACHE)
say(f"barrier: {int((B.kind == 'river').sum())} river components, {B[B.kind == 'river'].km2.sum():.0f} km2; Tonle Sap lake {B[B.kind == 'lake'].km2.sum():.0f} km2")
riv = shapely.union_all(B[B.kind == "river"].geometry.values); lak = shapely.union_all(B.geometry.values)
shapely.prepare(riv); shapely.prepare(lak)

# ---------------- crossing per cell ----------------
from scipy.spatial import cKDTree
_, j = cKDTree(np.c_[tw.x, tw.y]).query(np.c_[cl_.x, cl_.y])
seg = shapely.linestrings(np.stack([np.c_[cl_.x, cl_.y], np.c_[tw.x.values[j], tw.y.values[j]]], axis=1))
CE["cross"] = shapely.intersects(riv, seg).astype(float); CE["cross_lake"] = shapely.intersects(lak, seg).astype(float)
CE["d_riv"] = shapely.distance(riv, shapely.points(np.c_[cl_.x, cl_.y])) / 1000
CE[["k", "lat", "lon", "pop", "km", "tt_town_dry", "cross", "cross_lake", "d_riv"]].to_csv(ROOT / "data/processed/cambodia/kh_river_cells.csv", index=False)
wavg = lambda col: CE.groupby("k").apply(lambda g: np.average(g[col], weights=g["pop"]), include_groups=False)
for col in ["cross", "cross_lake", "d_riv"]:
    DF[col] = DF.unit.map(wavg(col))
DF["hub"] = DF.capital | DF.ADMIN_NAME.str.contains("Krong")
cc = CE[CE["pop"] > 0]
say(f"cells {len(cc)}: crossing share of population {np.average(cc.cross, weights=cc['pop']):.3f} (with lake {np.average(cc.cross_lake, weights=cc['pop']):.3f})")
say(f"  cell check, minutes per km to town, crossing vs not, within 30 km of a town: "
    f"{(cc.tt_town_dry / cc.km.clip(lower=.5))[(cc.cross == 1) & (cc.km < 30)].median():.2f} vs {(cc.tt_town_dry / cc.km.clip(lower=.5))[(cc.cross == 0) & (cc.km < 30)].median():.2f}")

d0 = counts(MAIN).reset_index(drop=True)
samples = {"all districts": d0, "rural districts only": d0[~d0.hub].reset_index(drop=True),
           "riverine band (centre within 20 km)": d0[d0.d_riv <= 20].reset_index(drop=True)}
for lab, d in samples.items():
    say(f"  {lab}: {len(d)} districts, {int((d.cross > 0.05).sum())} with >5% crossing, mean cross {d.cross.mean():.3f}, provinces with variation "
        f"{int(d.groupby('prov').cross.std().gt(0.05).sum())}")


def add(stage, sample, var, b, s, p_ri, p_cl, n, F=np.nan, extra=""):
    r = {"stage": stage, "sample": sample, "var": var, "coef": b, "se": s, "pct": pc(b) if stage != "first stage" else np.nan,
         "ci_lo": pc(b - 1.96 * s) if stage != "first stage" else b - 1.96 * s, "ci_hi": pc(b + 1.96 * s) if stage != "first stage" else b + 1.96 * s,
         "mde_pct": pc(-2.8 * s) if stage != "first stage" else np.nan, "p_ri": p_ri, "p_cluster": p_cl, "F": F, "n": n}
    rows.append(r)
    if stage == "first stage":
        say(f"  {stage:<12s} {sample:<38s} {var:<10s} detour log2 +{b:.3f} (SE {s:.3f}) | F {F:.1f} | n {n} {extra}")
    else:
        say(f"  {stage:<12s} {sample:<38s} {var:<10s} {r['pct']:+6.0f}% (95% CI {r['ci_lo']:+.0f}% to {r['ci_hi']:+.0f}%) | MDE {r['mde_pct']:+.0f}% | "
            f"p RI {p_ri:.3f} | p cluster {p_cl:.3f} | n {n} {extra}")
    return r


say("== first stage and reduced form (cross = share of population whose line to the town crosses the river)")
for var in ["cross", "cross_lake"]:
    for lab, d in samples.items():
        if var == "cross_lake" and lab != "all districts":
            continue
        fs = smf.ols(f"detour ~ {var} + dist + lnpop + C(prov)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]})
        F = (fs.params[var] / fs.bse[var]) ** 2
        add("first stage", lab, var, fs.params[var], fs.bse[var], np.nan, fs.pvalues[var], len(d), F)
        f, r = M0["fit"](d, [var, "dist"])
        p = M0["ri_p"](d, var, ["dist"], f, r.params[var], DRAWS)
        exp_ = pc(fs.params[var] * np.log(0.58))   # detour effect per doubling (log 0.58) x first-stage shift in log2 detour
        add("reduced form", lab, var, r.params[var], r.bse[var], p, r.pvalues[var], len(d), F,
            f"| expected if detour acts like -42% per doubling: {exp_:+.0f}%")
pd.DataFrame(rows).to_csv(T / "kh_river.csv", index=False)
(T / "kh_river.txt").write_text("\n".join(log), encoding="utf-8")
