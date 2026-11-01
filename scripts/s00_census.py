"""Posto census panel: 2010 Census Wall Chart (population + need indicators), 2015 (Wikidata), 2022 (de.wikipedia).

Output: data/processed/posto_census.csv keyed by COD pcode.
"""
import re, unicodedata
import pandas as pd, pdfplumber, geopandas as gpd
from rapidfuzz import process, fuzz
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
a2 = gpd.read_file(RAW / "cod_ab/tls_admin_boundaries.shp/tls_admin2.shp")[["adm2_pcode", "adm2_name", "adm1_name"]]
DIST = {"AINARO", "AILEU", "BAUCAU", "BOBONARO", "COVALIMA", "DILI", "ERMERA", "LAUTEM", "LIQUICA", "MANATUTO", "MANUFAHI", "OECUSSI", "VIQUEQUE"}


def key(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"ADMINISTRATIVE POST|SUB-?DISTRICT|POSTO", "", s)
    return re.sub(r"[^A-Z]", "", s)


ALIAS = {"HATOUDO": "HATUUDO", "HATUBUILICO": "HATUBUILICO", "FOHOREM": "FOROHEM", "MAUCATAR": "MAUKATAR", "HATULIA": "HATOLIA",
         "FATUBERLIO": "FATUBERLIU", "LEQUIDOE": "LIQUIDOE", "BARIQUENATARBORA": "BARIQUENATARBORA", "BARIQUE": "BARIQUENATARBORA",
         "NATARBORA": "BARIQUENATARBORA", "PANTEMAKASSAR": "PANTEMACASSAR", "PANTEMAKASAR": "PANTEMACASSAR", "OESILO": "OESILO",
         "UATOCARBAU": "UATUCARBAU", "UATULARI": "UATOLARI", "WATULARI": "UATOLARI", "AILEU": "AILEUVILA", "LAUTEMMORO": "LAUTEM", "MORO": "LAUTEM", "NAINFETO": "NAINFETO",
         "VERACRUZ": "VERACRUZ", "DOMALEIXO": "DOMALEIXO", "CRISTOREI": "CRISTOREI", "ATAURO": "ATAURO", "LIQUICA": "LIQUICA"}
cod = {key(n): (pc, mun) for n, pc, mun in zip(a2.adm2_name, a2.adm2_pcode, a2.adm1_name)}


def match(name, mun=None):
    k = ALIAS.get(key(name), key(name))
    if k in cod:
        return cod[k][0]
    cands = {kk: v for kk, v in cod.items() if mun is None or key(v[1])[:5] == key(mun)[:5]}
    best = process.extractOne(k, list(cands), scorer=fuzz.ratio)
    return cands[best[0]][0] if best and best[1] >= 80 else None


# ---- 2010 wall chart ----
with pdfplumber.open(RAW / "census/census2010_wallchart.pdf") as pdf:
    lines = (pdf.pages[0].extract_text() or "").splitlines()
NUM = r"-?[\d,]+(?:\.\d+)?-?"
rows, cur, seen_dist = [], None, set()
for ln in lines:
    m = re.match(rf"^([A-ZÀ-Ý][A-ZÀ-Ý\-\.' /]+?)\s+((?:{NUM}\s+){{10,}})([A-ZÀ-Ý][A-ZÀ-Ý\-\.' /]+?)\s+((?:{NUM}\s*)+)$", ln.strip())
    if not m:
        continue
    name = m.group(1).strip()
    a = [x.replace(",", "").rstrip("-") for x in m.group(2).split()]
    b = [x.replace(",", "").rstrip("-") for x in m.group(4).split()]
    if name in ("TIMOR-LESTE", "RURAL", "URBAN"):
        continue
    # a district name is a header the first time it appears; later (incl. right after its header) it is the capital posto
    is_dist = key(name) in DIST and name not in seen_dist
    if is_dist:
        cur = name; seen_dist.add(name)
        continue
    f = lambda xs, i: float(xs[i]) if i < len(xs) and xs[i] not in ("", "-") else None
    rows.append({"district": cur, "name2010": name, "pop2010": f(a, 0), "hh2010": f(a, 11), "area2010": f(a, 9),
                 "lfpr": f(a, 15), "ger_primary": f(b, 0), "ger_secondary": f(b, 3), "literacy_15_24": f(b, 6),
                 "disability": f(b, 9), "cook_clean": f(b, 14), "cook_wood": f(b, 16), "water_improved": f(b, 17),
                 "sanit_improved": f(b, 18), "crop_hh": f(b, 19), "livestock_hh": f(b, 20)})
C10 = pd.DataFrame(rows)
C10["pcode"] = [match(n, d) for n, d in zip(C10.name2010, C10.district)]
print("2010 rows", len(C10), "matched", C10.pcode.notna().sum())
print(C10[C10.pcode.isna()][["district", "name2010"]])

# ---- 2015 Wikidata, 2022 de.wikipedia ----
wd = pd.read_csv(RAW / "census/wikidata_posto_population.csv")
w15 = wd[wd.year == 2015].drop_duplicates("item")
w15 = w15.assign(pcode=[match(n, m) for n, m in zip(w15.itemLabel, w15.muniLabel)])
de = pd.read_csv(RAW / "census/dewiki_posto_population.csv")
de = de.merge(wd.drop_duplicates("item")[["item", "muniLabel"]], on="item", how="left")
de["pcode"] = [match(n, m) for n, m in zip(de.label, de.muniLabel)]
p15 = w15.groupby("pcode").pop.sum().rename("pop2015")
p22 = de[de.year_de == 2022].groupby("pcode").pop_de.sum().rename("pop2022c")
OUT = a2.rename(columns={"adm2_pcode": "pcode"})[["pcode", "adm2_name"]].merge(C10.drop(columns=["district"]), on="pcode", how="left")
OUT = OUT.merge(p15, on="pcode", how="left").merge(p22, on="pcode", how="left")
OUT["g_10_15"] = (OUT.pop2015 / OUT.pop2010) ** (1 / 5) - 1
OUT["g_15_22"] = (OUT.pop2022c / OUT.pop2015) ** (1 / 7) - 1
OUT.to_csv(PROC / "posto_census.csv", index=False, encoding="utf-8-sig")
print(OUT[["adm2_name", "pop2010", "pop2015", "pop2022c", "g_10_15", "water_improved", "sanit_improved"]].round(3).to_string())
print("coverage:", OUT[["pop2010", "pop2015", "pop2022c"]].notna().sum().to_dict())
print("national 2010 sum", OUT.pop2010.sum(), "2015 sum", OUT.pop2015.sum())
