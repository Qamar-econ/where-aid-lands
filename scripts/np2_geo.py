"""Nepal geocoder v2 (village / palika / district / province) with AidData-UCDP precision codes.

Precision codes used:  1 exact point | 2 village/town (<25 km) | 3 palika (ADM3) | 4 district (ADM2) | 6 province/zone/region (ADM1-like)
                       8 national / central institution
Rules (codebook v2):
  V1  a village name counts only if its district is also named in the same text (co-mention), or if the village name is a
      unique, long (>= 7 chars) town name in GeoNames with population > 0
  V2  a palika name counts if unique to one district, otherwise only with its district co-mentioned
  V3  capital-valley mentions in HQ contexts are dropped (geocoder.HQ rules); Kathmandu found only in location fields -> HQ
  V4  the most precise level found for a district supersedes coarser mentions of the same district
  V5  a village/palika sharing a district's name (Kathmandu, Parsa, Bhojpur ...) is coded at district level only
Usage:  from np2_geo import geocode_text  ->  list of dicts (level, unit, district, precision, term, snippet)
"""
import re, sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gazetteer_nepal import GEO, UPPER_DISTRICTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Z = pd.read_csv(ROOT / "data/interim/np2_gazetteer.csv")
Z["key"] = Z.name.str.lower()
ENTRIES = {}
for r in Z.itertuples():
    ENTRIES.setdefault(r.key, []).append(r)
MAXN = int(Z.key.str.split().map(len).max())
TOK = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _ngrams(text):
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in TOK.finditer(text)]
    for i in range(len(toks)):
        for n in range(1, min(MAXN, 5) + 1):
            if i + n > len(toks):
                break
            yield " ".join(t[0] for t in toks[i:i + n]), toks[i][1], toks[i + n - 1][2]


# province / zone / river names and parts of multi-word district names: never village/palika evidence
REGION_WORDS = {"gandaki", "bagmati", "koshi", "kosi", "madhesh", "madhesh pradesh", "lumbini", "karnali", "sudurpaschim", "sudurpashchim",
                "narayani", "janakpur", "mechi", "sagarmatha", "bheri", "seti", "mahakali", "rapti", "dhaulagiri", "palanchok", "palanchowk",
                "terai", "tarai", "himalaya", "himalayan", "mid-west", "far-west"}
_DN = {}


def _is_district_name(g):
    if g not in _DN:
        _DN[g] = bool(GEO.parse(f"in {g.title()} district")[0])
    return _DN[g]


def geocode_text(text):
    if not isinstance(text, str) or not text.strip():
        return []
    units, uppers, kept, _ = GEO.parse(text)
    out = [{"level": "district", "unit": d, "district": d, "precision": 4, "term": m.term, "snippet": m.context}
           for d in units for m in kept if m.level == "unit" and m.code == d][:200]
    named = set(units)
    for c in uppers:
        for d in UPPER_DISTRICTS[c]:
            out.append({"level": "region", "unit": c, "district": d, "precision": 6, "term": c, "snippet": ""})
    seen = set()
    for g, a, b in _ngrams(text):
        ents = ENTRIES.get(g)
        if not ents or g in seen:
            continue
        if g in REGION_WORDS or _is_district_name(g):  # V5: a place sharing a district's name is coded at district level only
            seen.add(g)
            continue
        for e in ents:
            ok = False
            if e.level == "palika":
                ok = (e.n_districts == 1 and len(g) >= 6) or e.k in named
                prec = 3
            else:
                ok = e.k in named
                prec = 2
            if ok:
                out.append({"level": e.level, "unit": e.pcode, "district": e.k, "precision": prec, "term": e.name,
                            "snippet": text[max(0, a - 70):b + 70].replace("\n", " ")})
                seen.add(g)
    return out


def best_by_district(hits):
    """V4: keep the most precise hit per district (and all distinct palikas at that precision)."""
    if not hits:
        return []
    df = pd.DataFrame(hits)
    m = df.groupby("district").precision.transform("min")
    return df[df.precision == m].drop_duplicates(["district", "unit"]).to_dict("records")
