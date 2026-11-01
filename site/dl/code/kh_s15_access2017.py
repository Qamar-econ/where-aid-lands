"""Road access measured before the projects: the index rebuilt on Geofabrik's 1 January 2017 OSM extract
(kh_s03_access.py with KH_OSM_SNAPSHOT=170101) against the 2026 extract used so far.
Outcome: activities (kh_s14_dedup.py, D1) and, for comparison, CRS records (D0). Same PPML, FE, inference as kh_s04.
Outputs: outputs/tables/kh_access2017_main.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s14 = (ROOT / "scripts/kh_s14_dedup.py").read_text(encoding="utf-8")
exec(s14[:s14.index("\n# since spec 9-7 kh_assign.csv")])   # kh_s04 data + functions, and kh_s14's activity table AA

# since spec 9-7 kh_access.csv is the 2017 network; the current network is kh_access_latest.csv
a26 = pd.read_csv(OUT / "kh_access_latest.csv").rename(columns={"k": "unit"})[["unit", "access_pc1", "road_km", "tt_town_dry"]]
a26.columns = ["unit"] + [c + "_26" for c in a26.columns[1:]]
DF = DF.merge(a26, on="unit", how="left")
DF["phys17"] = DF.phys
DF["phys26"] = z(DF.access_pc1_26)
DF["road_km_17"], DF["tt_town_dry_17"] = DF.road_km, DF.tt_town_dry
c17 = DF[["phys17", "conn", "conn_tests", "conn_fixed"]]
w17, V17 = np.linalg.eigh(np.cov(c17.T)); pc17 = V17[:, -1] * np.sign(V17[:, -1].sum())
DF["combined17"] = DF.combined
c26 = DF[["phys26", "conn", "conn_tests", "conn_fixed"]]
w26, V26 = np.linalg.eigh(np.cov(c26.T)); pc26 = V26[:, -1] * np.sign(V26[:, -1].sum())
DF["combined26"] = z(c26.values @ pc26)
print("districts", len(DF), "| corr(access 2017, 2026) =", round(DF.phys17.corr(DF.phys26), 3),
      "| spearman", round(DF.phys17.corr(DF.phys26, method="spearman"), 3))
print("road km 2017 vs 2026 (sum):", round(DF.road_km_17.sum()), "vs", round(DF.road_km_26.sum()),
      "| median town travel time 2017 vs 2026:", round(DF.tt_town_dry_17.median(), 1), "vs", round(DF.tt_town_dry_26.median(), 1))
DF["gain"] = DF.phys26 - DF.phys17
print("where the map grew most (largest 2026-2017 index gain):")
print(DF.nlargest(6, "gain")[["unit", "phys17", "phys26", "pop_2019"]].round(2).to_string(index=False))

Ar = pd.read_csv(OUT / "kh_assign_records.csv")
defs = {"Main activities": A[A.level == "adm2"], "D0 records": Ar[Ar.level == "adm2"]}
rows = []
for dn, a in defs.items():
    d = outcome(a)
    for v in ["phys17", "phys26", "combined17", "combined26"]:
        rows.append({"outcome": dn, **full(d, v, ri=(dn == "Main activities"))})
    # does the pre-period index survive the later map? both in one model (they are the same construct measured twice)
    r = ppml("N ~ phys17 + gain + lnpop + C(prov)", d.assign(gain=d.phys26 - d.phys17))
    rows.append({"outcome": dn, "var": "phys17 | +gain", "pct": 100 * (np.exp(r.params["phys17"]) - 1), "p_cluster": r.pvalues["phys17"],
                 "gain_pct": 100 * (np.exp(r.params["gain"]) - 1), "gain_p": r.pvalues["gain"]})
R = pd.DataFrame(rows)
R.to_csv(T / "kh_access2017_main.csv", index=False)
print(R[[c for c in ["outcome", "var", "pct", "p_cluster", "p_conley100", "p_ri", "p_conservative", "ci_lo_pct", "ci_hi_pct", "gain_pct", "gain_p"] if c in R]]
      .round(3).to_string(index=False))
