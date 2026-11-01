"""District x year connectivity for Nepal, 2019-2025 (Ookla Q1 open data), plus validation.

Measures (per district d, year t):
  mob_tests_pc   mobile speed tests per 1,000 people (population log-linearly interpolated 2011 -> 2023 census/WorldPop)
  mob_mbps       test-weighted mean mobile download speed (Mbps)
  mob_tiles_km2  ~600 m tiles with >= 1 mobile test per 100 km2 (spatial footprint of use)
  fix_tests_pc   fixed-broadband tests per 1,000 people
  net_idx        first principal component of the standardised logs (pooled over years; higher = more connected)
  hs_year        first year in which test-weighted mobile download exceeds 10 Mbps (proxy for 4G-class service)
Validation: correlation of each 2019 measure with census-2011 household internet / mobile shares (district level)
Output: data/processed/nepal/np2_connectivity.csv, outputs/tables/np2_connectivity_validation.csv
"""
import math
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nepal/ookla"
D = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")


def xy2lonlat(x, y, z=16):
    n = 2 ** z
    lon = (x + 0.5) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return lon, lat


rows = []
for typ in ["mobile", "fixed"]:
    for y in range(2019, 2026):
        f = RAW / f"ookla_{typ}_{y}Q1_NP.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f)
        ll = [xy2lonlat(a, b) for a, b in zip(o.tx, o.ty)]
        g = gpd.GeoDataFrame(o, geometry=gpd.points_from_xy([a for a, b in ll], [b for a, b in ll]), crs=4326)
        g = gpd.sjoin(g, D[["k", "geometry"]], predicate="within")
        a = g.groupby("k").apply(lambda d: pd.Series({"tests": d.tests.sum(), "tiles": len(d),
                                                       "mbps": np.average(d.avg_d_kbps, weights=d.tests) / 1000}), include_groups=False)
        a["typ"], a["year"] = typ, y
        rows.append(a.reset_index())
X = pd.concat(rows)
P = X.pivot_table(index=["k", "year"], columns="typ", values=["tests", "tiles", "mbps"]).fillna(0)
P.columns = [f"{b[:3]}_{a}" for a, b in P.columns]
P = P.reset_index()
full = pd.MultiIndex.from_product([D.k, range(2019, 2026)], names=["k", "year"]).to_frame(index=False)
P = full.merge(P, on=["k", "year"], how="left").fillna(0)
P = P.merge(D[["k", "pop2011", "pop2023", "area_km2", "internet_pct", "mobile_pct", "computer_pct"]], on="k")
P["pop"] = np.exp(np.log(P.pop2011) + (np.log(P.pop2023) - np.log(P.pop2011)) * (P.year - 2011) / 12)
P["mob_tests_pc"] = P.mob_tests / P["pop"] * 1000
P["fix_tests_pc"] = P.fix_tests / P["pop"] * 1000
P["mob_tiles_km2"] = P.mob_tiles / P.area_km2 * 100
P["mob_mbps"] = P.mob_mbps
comp = pd.DataFrame({c: np.log1p(P[c]) for c in ["mob_tests_pc", "mob_mbps", "mob_tiles_km2", "fix_tests_pc"]})
Z = (comp - comp.mean()) / comp.std()
w, v = np.linalg.eigh(np.cov(Z.T))
pc = v[:, -1] * np.sign(v[:, -1].sum())
P["net_idx"] = Z.values @ pc
P["net_idx"] = (P.net_idx - P.net_idx.mean()) / P.net_idx.std()
hs = P[P.mob_mbps > 10].groupby("k").year.min()
P["hs_year"] = P.k.map(hs)
P.to_csv(ROOT / "data/processed/nepal/np2_connectivity.csv", index=False, encoding="utf-8-sig")

V = []
p19 = P[P.year == 2019]
for c in ["mob_tests_pc", "mob_mbps", "mob_tiles_km2", "fix_tests_pc", "net_idx"]:
    for ref in ["internet_pct", "mobile_pct", "computer_pct"]:
        V.append({"measure_2019": c, "census2011": ref, "spearman": round(p19[c].corr(p19[ref], method="spearman"), 3)})
# within-district stability: rank correlation across years
for y in range(2020, 2026):
    a = P[P.year == 2019].set_index("k").net_idx; b = P[P.year == y].set_index("k").net_idx
    V.append({"measure_2019": "net_idx", "census2011": f"net_idx_{y}", "spearman": round(a.corr(b, method="spearman"), 3)})
V = pd.DataFrame(V)
(ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
V.to_csv(ROOT / "outputs/tables/np2_connectivity_validation.csv", index=False)
print("PC1 loadings:", dict(zip(comp.columns, pc.round(2))), "| var share:", round(w[-1] / w.sum(), 2))
print(V.to_string(index=False))
print("national mean by year:", P.groupby("year")[["mob_tests_pc", "mob_mbps", "fix_tests_pc"]].mean().round(2).to_dict())
print("districts reaching >10 Mbps by year:", hs.value_counts().sort_index().to_dict(), "| never:", int(P.k.nunique() - hs.size))
print("within-district SD of net_idx / total SD:", round(P.groupby("k").net_idx.transform(lambda s: s - s.mean()).std() / P.net_idx.std(), 2))
