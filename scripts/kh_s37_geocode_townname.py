"""Geocoding check: EU/US activities placed at a province-capital town whose name is also the province (spec 9-25, exploratory).

GODAD EU/US records with CRS precision 3 are kept as district-precise, but a place name such as "Krong Siem Reap" may stand
for the whole province. Re-estimates the physical-access model with these activities dropped, spread over the province
(population, equal, worst case), plus checks without province FE and for multi-district activities. Prints only.
"""
import os; os.environ["KH_GEOCODE"] = "original"   # this script needs the sample before the spec 9-26 reclassification
from pathlib import Path
SRC = Path("kh_s06_ladder.py"); ns = {"__file__": str(SRC.resolve())}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
import pandas as pd, numpy as np
DF, g, cur, outcome, fit = ns["DF"], ns["g"], ns["cur"], ns["outcome"], ns["fit"]
nm = DF.set_index("unit").ADMIN_NAME
gg = g[g.project_id.isin(cur.project_id)].copy()
gg["krong"] = gg.unit.map(nm).fillna("").str.contains("Krong")
gg["pn"] = gg.place_name.astype(str).str.split(",").str[0].str.strip()
eu_k = gg[(gg.dgroup == "EU/US") & gg.krong]
print("EU/US records in Krong: precision_crs", eu_k.precision_crs.value_counts().to_dict())
print(eu_k[["title","pn"]].drop_duplicates("title").head(12).to_string())
TOWNPROV = r"Krong (Siem Reap|Battambang|Pursat|Kampong Chhnang|Kracheh|Kampot|Kampong Cham|Chbar Mon|Stueng Saen|Ta Khmau|Takeo|Doun Kaev|Svay Rieng|Prey Veng|Banlung|Stung Treng|Sen Monorom|Preah Vihear|Samraong|Pailin|Kaeb|Serei Saophoan|Kampong Thum|Kampong Spueu)"
def est(c, lab):
    d = outcome(c); m = fit("N ~ access + lnpop + C(prov)", d)
    b, s = m.params["access"], m.bse["access"]
    print(f"{lab:<60s} projects {d.N.sum():6.1f} | access {100*(np.exp(b)-1):+.0f}% (95% CI {100*(np.exp(b-1.96*s)-1):+.0f}% to {100*(np.exp(b+1.96*s)-1):+.0f}%), cluster p {m.pvalues['access']:.3f}")
est(cur, "baseline")
ids_a = gg[(gg.dgroup=="EU/US") & gg.krong & gg.pn.str.match(TOWNPROV)].project_id.unique()
est(cur[~cur.project_id.isin(ids_a)], f"drop EU/US activities geocoded to a province-capital town name ({len(ids_a)})")
ids_b = eu_k.project_id.unique()
est(cur[~cur.project_id.isin(ids_b)], f"drop all EU/US activities in Krong districts ({len(ids_b)})")
krong_units = DF[DF.ADMIN_NAME.str.contains("Krong")].unit
est(cur[~cur.unit.isin(krong_units)], "drop all activities in the 25 Krong districts")
# treat town-name EU/US geocodes as province-level: spread each activity's weight over its province's districts by population
def spread(c, ids, how="pop"):
    keep = c[~c.project_id.isin(ids)]
    moved = c[c.project_id.isin(ids)].copy(); moved["prov"] = moved.unit.map(DF.set_index("unit").prov)
    tot = moved.groupby(["project_id", "prov"]).w.sum().reset_index()
    pr = DF[["unit", "prov", "pop_2019", "access"]]
    out = []
    for r in tot.itertuples():
        d = pr[pr.prov == r.prov]
        if how == "pop": sh = d.pop_2019 / d.pop_2019.sum()
        elif how == "worst": sh = (d.access == d.access.min()).astype(float)
        else: sh = pd.Series(1 / len(d), index=d.index)
        out.append(pd.DataFrame({"project_id": r.project_id, "unit": d.unit.values, "w": r.w * sh.values, "comm": 0.0}))
    return pd.concat([keep] + out, ignore_index=True)
est(spread(cur, ids_a, "pop"), "town-name geocodes spread over province by population")
est(spread(cur, ids_a, "equal"), "town-name geocodes spread equally over province")
est(spread(cur, ids_a, "worst"), "town-name geocodes moved to least accessible district (worst case)")
wb_k = gg[(gg.dgroup=="World Bank") & gg.krong]
print("World Bank records in Krong:", len(wb_k), "precision", wb_k.precision_code.value_counts().to_dict(), "| distinct activities", wb_k.project_id.nunique())
d = outcome(cur); m = fit("N ~ access + lnpop", d); b = m.params["access"]
print(f"no province FE: access {100*(np.exp(b)-1):+.0f}%, cluster p {m.pvalues['access']:.3f}")
K_ = cur.groupby("project_id").unit.transform("nunique")
for k in (1, 3, 5):
    est(cur[K_ <= k], f"only activities listed in <= {k} districts")
print("activities total", cur.project_id.nunique(), "| town-name EU/US", len(ids_a), "| their N share", round(cur[cur.project_id.isin(ids_a)].w.sum() / cur.w.sum(), 3))
