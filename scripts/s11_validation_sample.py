"""Validation samples.

(1) AI pre-check sample: 60 random text-geocoded projects with the evidence (matched term + context) -> printed for review.
(2) Human validation sheet: stratified random sample of 200 projects (by source x target type) with all automated codes,
    full text and empty columns for two human coders -> data/manual/validation_sheet.xlsx (+ instructions sheet).
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gazetteer import geoparse_detail, POSTO, MUN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, INT, MAN = ROOT / "data" / "processed", ROOT / "data" / "interim", ROOT / "data" / "manual"
PR = pd.read_csv(PROC / "oda_projects.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
iati = pd.read_csv(INT / "iati_fulltext.csv").drop_duplicates("aid").set_index("aid").fillna("")
import geopandas as gpd
a2 = gpd.read_file(ROOT / "data/raw/cod_ab/tls_admin_boundaries.shp/tls_admin2.shp")
name = dict(zip(a2.adm2_pcode, a2.adm2_name)) | dict(zip(a2.adm1_pcode, a2.adm1_name))


def full_text(r):
    if r.source == "IATI":
        a = r.uid[5:]
        return f"{r.title} | {r.desc} | {iati.location_names.get(a, '')} | {iati.text.get(a, '')}"
    return f"{r.title} | {r.desc}"


def units_str(uid):
    a = AS[AS.uid == uid]
    return "; ".join(f"{name.get(u, u)}[{t}]" for u, t in zip(a.unit, a.tier.fillna("-"))) if (a.level != "national").any() else ""


PR = PR[PR.start_year.between(2000, 2026)].copy()
meth = AS.groupby("uid").method.first()
PR["method"] = PR.uid.map(meth)

if len(sys.argv) > 1 and sys.argv[1] == "ai":
    s = PR[PR.method == "text"].sample(60, random_state=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    for i, r in enumerate(s.itertuples()):
        u, m, kept, hq = geoparse_detail(full_text(r))
        ev = " || ".join(f"[{k.term}] …{k.context[:110]}…" for k in kept[:3])
        print(f"#{i} {r.uid} | {str(r.title)[:90]}\n   -> {units_str(r.uid)}\n   EV: {ev}")
    s[["uid"]].to_csv(INT / "ai_check_sample.csv", index=False)
    sys.exit()

# human validation sheet (stratified)
PR["stratum"] = PR.source + " / " + PR.target_type
n_total = 200
w = PR.stratum.value_counts(normalize=True)
alloc = (w * n_total).round().clip(lower=3).astype(int)
parts = [PR[PR.stratum == k].sample(min(v, (PR.stratum == k).sum()), random_state=11) for k, v in alloc.items()]
S = pd.concat(parts).sample(frac=1, random_state=3)
S["auto_units"] = S.uid.map(units_str)
S["text"] = [full_text(r)[:3000] for r in S.itertuples()]
out = S[["uid", "source", "donor", "start_year", "title", "text", "cat", "ict", "target_type", "auto_units"]].copy()
for c in ["coder", "h_target_type", "h_municipalities", "h_postos", "h_sector", "h_ict", "h_confidence", "h_notes"]:
    out[c] = ""
instr = pd.DataFrame({"field": ["h_target_type", "h_municipalities", "h_postos", "h_sector", "h_ict", "h_confidence", "h_notes", "general"],
                      "instruction": [
                          "located / central_institution / nationwide / regional_multicountry / unresolved (codebook §3)",
                          "Target municipalities, ';'-separated official names (Aileu, Ainaro, Baucau, Bobonaro, Covalima, Dili, Ermera, Liquiçá, Lautém, Manufahi, Manatuto, Oecussi, Viqueque). Leave empty if not located.",
                          "Target postos if identifiable, ';'-separated official COD names. Mentions of offices/HQ ('based in Dili') are NOT targets.",
                          "One of: SOC_education, SOC_health, SOC_water, SOC_other, GOV, PROD, ENV_MULTI, ACCESS_transport, ACCESS_energy, ACCESS_comms, HUMAN, OTHER",
                          "1 if the project delivers or builds digital/ICT components (information systems, e-services, connectivity, digital learning), else 0",
                          "high / moderate / low", "free text", "Code from the 'text' column only (you may consult the donor's project page; note it in h_notes). Two coders independently; do not look at auto_* columns first if possible."]})
MAN.mkdir(exist_ok=True)
with pd.ExcelWriter(MAN / "validation_sheet.xlsx") as xw:
    instr.to_excel(xw, sheet_name="instructions", index=False)
    out.to_excel(xw, sheet_name="coder1", index=False)
    out.to_excel(xw, sheet_name="coder2", index=False)
print("validation sheet:", len(out), "projects;", out.source.value_counts().to_dict())
