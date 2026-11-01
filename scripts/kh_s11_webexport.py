"""Export data for the Cambodia research archive web page (research/site/).
site/data/districts.geojson  simplified GEO2_KH districts with all district measures
site/data/projects.json      GODAD project locations (district-precise, starts 2017-2020) for the hover maps
site/data/results.json       compact result tables used by the page's charts
site/dl/*                    downloadable bundles (data zip, tables zip, code zip, documents)
"""
import json, math, zipfile
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, K, U, ROOT, RAW = ns["DF"], ns["L"], ns["K"], ns["U"], ns["ROOT"], ns["RAW"]
SITE = ROOT / "site"; (SITE / "data").mkdir(parents=True, exist_ok=True); (SITE / "dl").mkdir(parents=True, exist_ok=True)
T = ROOT / "outputs/tables"
z = lambda s: (s - s.mean()) / s.std()

# connectivity index and need (as in kh_s08 / kh_s10)
o = pd.concat([pd.read_csv(f) for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv"))]).drop_duplicates(["tx", "ty"])
n16 = 2 ** 16
o["lon"] = (o.tx + .5) / n16 * 360 - 180
o["lat"] = [math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n16)))) for y in o.ty]
og = gpd.sjoin(gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o.lon, o.lat), crs=4326), U[["unit", "geometry"]], predicate="within")
area = DF.filter(like="area_km2").iloc[:, 0]
cc = pd.DataFrame({"s": z(np.log(DF.mob_mbps_shrunk)), "t": z(np.log1p(DF.mob_tests_pc)),
                   "f": z(np.log1p(DF.unit.map(og.groupby("unit").size()).fillna(0) / area)), "x": z(np.log(DF.fix_mbps_shrunk))})
w_, V_ = np.linalg.eigh(np.cov(cc.T)); pcv = V_[:, -1] * np.sign(V_[:, -1].sum())
DF["conn_index"] = z(cc.values @ pcv)
rwi = pd.read_csv(ROOT / "data/processed/cambodia/kh_rwi.csv").set_index("unit").rwi
DF["rwi"] = DF.unit.map(rwi)

cur = L[L.start.between(2017, 2020) & (L.family != "ict")]
N = cur.groupby("unit").w.sum()
DF["N"] = DF.unit.map(N).fillna(0)
DF["pc100k"] = DF.N / DF.pop_2019 * 1e5
geo = U[["unit", "ADMIN_NAME", "geometry"]].merge(
    DF[["unit", "PARENT", "pop_2019", "N", "pc100k", "ntl_pc_2019", "access", "access_pc1", "conn_index", "mob_mbps_shrunk", "rwi", "tt_town_dry", "prior_infra"]],
    on="unit", how="left")
geo["geometry"] = geo.geometry.simplify(0.004, preserve_topology=True)
for c in geo.columns:
    if c not in ("geometry", "ADMIN_NAME"):
        geo[c] = pd.to_numeric(geo[c], errors="coerce").round(4)
geo = geo.rename(columns={"ADMIN_NAME": "name", "PARENT": "prov"})
geo.to_file(SITE / "data/districts.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)

# project locations for the hover layer
K["adm2"] = np.where(K.dgroup == "EU/US", K.precision_crs == 3, K.precision_code <= 3)
P = K[K.startyear.between(2017, 2020) & K.adm2 & (K.family != "ict")].copy()
FAM = {"social": "사회", "infra": "인프라(교통·에너지)", "production": "생산", "other": "기타"}
pts = [{"id": str(r.project_id), "t": str(r.title)[:160], "d": r.donor, "g": r.dgroup, "y": int(r.startyear),
        "s": FAM.get(r.family, "기타"), "c": None if pd.isna(r.comm) else round(float(r.comm) / 1e6, 2),
        "x": round(float(r.longitude), 4), "yy": round(float(r.latitude), 4)} for r in P.itertuples() if pd.notna(r.longitude)]
(SITE / "data/projects.json").write_text(json.dumps(pts, ensure_ascii=False), encoding="utf-8")

# compact results
rd = lambda f: pd.read_csv(T / f)
res = {"main": rd("kh_main.csv")[["var", "pct", "p_conservative", "p_cluster", "p_conley100", "p_ri", "mde_pct", "ci_lo_pct", "ci_hi_pct", "pct_per_within_sd"]].round(3).to_dict("records"),
       "shapley5": rd("kh_shapley5.csv").round(2).to_dict("records"),
       "event": rd("kh_ext_B_event.csv").round(3).to_dict("records"),
       "bounds": rd("kh_ext_A_bounds.csv").round(3).to_dict("records"),
       "geobias": rd("kh_geobias_checks.csv").round(3).to_dict("records"),
       "infra1": rd("kh_infra1.csv").round(3).to_dict("records"),
       "infra2": rd("kh_infra2.csv").round(3).to_dict("records"),
       "need": rd("kh_need.csv").round(3).to_dict("records"),
       "conn": rd("kh_ext_C_connectivity.csv").round(3).to_dict("records"),
       "ladder": rd("kh_ladder.csv").round(3).to_dict("records")}
(SITE / "data/results.json").write_text(json.dumps(res, ensure_ascii=False, default=lambda x: None).replace("NaN", "null"), encoding="utf-8")

# downloadable bundles
def zipit(name, files, base):
    with zipfile.ZipFile(SITE / "dl" / name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists():
                zf.write(f, f.relative_to(base))
proc = ROOT / "data/processed/cambodia"
zipit("cambodia_data.zip", [proc / x for x in ["kh_projects.csv", "kh_assign.csv", "kh_covariates.csv", "kh_access.csv", "kh_rwi.csv", "kh_units.gpkg"]]
      + [RAW / "godad_khm.csv", ROOT / "outputs/tables/kh_access_pca.csv"], ROOT)
zipit("cambodia_results_tables.zip", sorted(T.glob("kh_*.csv")), ROOT)
zipit("cambodia_code.zip", sorted((ROOT / "scripts").glob("kh_*.py")), ROOT)
for f in ["kh_projects.csv", "kh_covariates.csv", "kh_access.csv"]:
    (SITE / "dl" / f).write_bytes((proc / f).read_bytes())
docs = {"model_spec.md": ROOT / "모형명세서_캄보디아.md", "results_report.md": ROOT / "보고서_캄보디아_1차결과.md",
        "literature_review.md": ROOT / "paper/선행연구_체계적검토.md", "geocoding_bias_review.md": ROOT / "paper/지오코딩편향_문헌조사.md"}
for k, v in docs.items():
    (SITE / "dl" / k).write_bytes(v.read_bytes())
print("districts", len(geo), "| project points", len(pts), "| site files:", sorted(p.name for p in (SITE / "dl").iterdir()))
print("sizes KB:", {p.name: round(p.stat().st_size / 1024) for p in list((SITE / "data").iterdir()) + list((SITE / "dl").iterdir())})

# province outlines for the map overlay (dissolved GEO2_KH parents, lake excluded)
pv = gpd.read_file(proc / "kh_units.gpkg")
pv = pv[~pv.ADMIN_NAME.str.contains("Lake", case=False, na=False)].dissolve(by="PARENT", as_index=False)[["PARENT", "geometry"]]
pv["geometry"] = pv.geometry.simplify(0.003, preserve_topology=True)
pv.to_file(SITE / "data/provinces.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)