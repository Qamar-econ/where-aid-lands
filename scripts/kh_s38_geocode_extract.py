"""Lists EU/US activities in the 2017-2020 sample whose geocoded place name is a province-capital town (Krong ...) sharing the
province's name, or a province name itself, with title and description, for hand classification (spec 9-26).
Output: data/processed/cambodia/kh_geocode_flagged.csv (classification column filled by hand in kh_geocode_classified.csv)
"""
import re, pandas as pd
from pathlib import Path
SRC = Path(__file__).with_name("kh_s06_ladder.py"); ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, g, cur, ROOT = ns["DF"], ns["g"], ns["cur"], ns["ROOT"]
PROV = ["banteay meanchey", "battambang", "kampong cham", "kampong chhnang", "kampong speu", "kampong spueu", "kampong thom", "kampong thum",
        "kampot", "kandal", "kep", "kaeb", "koh kong", "kaoh kong", "kratie", "kracheh", "mondulkiri", "mondol kiri", "oddar meanchey",
        "otdar mean chey", "pailin", "phnom penh", "phnum penh", "preah sihanouk", "sihanoukville", "preah vihear", "prey veng", "pursat",
        "pouthisat", "ratanakiri", "rotanak kiri", "ratanak kiri", "siem reap", "siemreab", "stung treng", "steung treng", "stoeng treng",
        "svay rieng", "takeo", "takéo", "tbong khmum", "tboung khmum"]
TOWN = r"krong (siem reap|battambang|pursat|kampong chhnang|kracheh|kampot|kampong cham|chbar mon|stueng saen|ta khmau|takeo|doun kaev|svay rieng|prey veng|banlung|stung treng|sen monorom|preah vihear|samraong|pailin|kaeb|serei saophoan|kampong thum|kampong spueu|suong|preah sihanouk)"
gg = g[g.project_id.isin(cur.project_id) & (g.dgroup == "EU/US")].copy()
gg["pn"] = gg.place_name.astype(str).str.split(",").str[0].str.strip()
pl = gg.pn.str.lower().str.replace(r"\s+province$", "", regex=True).str.replace("é", "e")
gg["flag"] = None
gg.loc[pl.isin([p.replace("é", "e") for p in PROV]), "flag"] = "province name"
gg.loc[gg.pn.str.lower().str.fullmatch(TOWN), "flag"] = "town name"
F = gg[gg.flag.notna()]
nm = DF.set_index("unit")
out = (F.groupby("project_id").agg(flag=("flag", lambda s: "; ".join(sorted(set(s)))), place=("pn", lambda s: "; ".join(sorted(set(s)))),
                                  unit=("unit", "first"), donor=("donor", "first"), title=("title", "first"), description=("description", "first"))
       .reset_index())
out["district"] = out.unit.map(nm.ADMIN_NAME); out["prov"] = out.unit.map(nm.prov)
out["w_in_sample"] = out.project_id.map(cur.groupby("project_id").w.sum())
out["n_units"] = out.project_id.map(cur.groupby("project_id").unit.nunique())
out["description"] = out.description.astype(str).str.replace(r"\s+", " ", regex=True).str[:600]
out.to_csv(ROOT / "data/processed/cambodia/kh_geocode_flagged.csv", index=False)
F[["project_id", "unit", "pn", "flag"]].drop_duplicates(["project_id", "unit"]).to_csv(ROOT / "data/processed/cambodia/kh_geocode_flagged_units.csv", index=False)
print(len(out), "activities |", out.flag.value_counts().to_dict(), "| sample weight", round(out.w_in_sample.sum(), 1), "of", round(cur.w.sum(), 1))
