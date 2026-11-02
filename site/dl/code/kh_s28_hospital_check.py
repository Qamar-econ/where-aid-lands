"""Hospitals: OpenStreetMap (2017 snapshot) against the Ministry of Health list of referral hospitals (2010, via HDX).

Location agreement (share of MoH hospitals with an OSM hospital within 2 and 5 km, and the reverse), districts covered by
each source, and the physical-access index rebuilt with MoH hospitals as the destination (kh_s03 KH_HOSP=moh, tag mohhosp):
its correlation with the main index and the PPML effect (clustered and Conley p).
Output: outputs/tables/kh_hospital_check.txt
"""
import numpy as np, pandas as pd, geopandas as gpd, osmium
from scipy.spatial import cKDTree
from scipy.stats import norm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCD = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.h = []

    def node(self, n):
        if n.tags.get("amenity") == "hospital" or n.tags.get("healthcare") == "hospital":
            self.h.append((n.location.lon, n.location.lat))

    def way(self, w):
        if w.tags.get("amenity") == "hospital":
            try:
                self.h.append((w.nodes[0].lon, w.nodes[0].lat))
            except Exception:
                pass


h = H(); h.apply_file(str(RAW / "osm/cambodia-170101.osm.pbf"), locations=True)
osm = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*np.array(h.h).T), crs=4326).to_crs(32648)
moh = gpd.read_file(RAW / "moh_health/Health/khm_hltfacp_referral_gov.shp").to_crs(32648)
xy = lambda g: np.c_[g.geometry.x, g.geometry.y] / 1000
d_m2o = cKDTree(xy(osm)).query(xy(moh))[0]; d_o2m = cKDTree(xy(moh)).query(xy(osm))[0]
say(f"MoH referral hospitals: {len(moh)} | OSM 2017 hospital features: {len(osm)}")
for k in (2, 5):
    say(f"  MoH hospitals with an OSM hospital within {k} km: {np.mean(d_m2o <= k):.0%} | OSM hospitals within {k} km of a MoH one: {np.mean(d_o2m <= k):.0%}")
U = gpd.read_file(PROCD / "kh_units.gpkg").to_crs(32648)
cnt = lambda g: gpd.sjoin(g, U[["GEOLEVEL2", "geometry"]], predicate="within").GEOLEVEL2.nunique()
say(f"  districts with at least one hospital: MoH {cnt(moh)} | OSM {cnt(osm)} | of {len(U)}")

f = PROCD / "kh_access_mohhosp.csv"
if f.exists():
    src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
    g = {"__file__": str(ROOT / "scripts/kh_s04_models.py")}; exec(src[:src.index("\nbase = A[A.level")], g)
    A, outcome, ppml, conley_se = g["A"], g["outcome"], g["ppml"], g["conley_se"]
    main = pd.read_csv(PROCD / "kh_access.csv").set_index("k"); alt = pd.read_csv(f).set_index("k")
    d = outcome(A[A.level == "adm2"]); s_ = d.unit.map(alt.access_pc1); d["xv"] = (s_ - s_.mean()) / s_.std()
    r = ppml("N ~ xv + lnpop + C(prov)", d); b = r.params["xv"]
    pmax = max([r.pvalues["xv"]] + [2 * norm.sf(abs(b / conley_se(r, d, "xv", c))) for c in (50, 100)])
    hosp_corr = np.corrcoef(np.log(main.tt_hosp_dry), np.log(alt.tt_hosp_dry))[0, 1]
    say(f"Index with MoH hospitals: corr with main index {alt.access_pc1.corr(main.access_pc1):.3f}; corr of log travel time to hospital {hosp_corr:.2f}; "
        f"effect {100 * (np.exp(b) - 1):+.0f}% (largest of clustered and Conley p = {pmax:.3f})")
(ROOT / "outputs/tables/kh_hospital_check.txt").write_text("\n".join(log), encoding="utf-8")
