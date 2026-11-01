"""2022 census posto populations from German Wikipedia (sitelinks of Wikidata posto items) and the
2010 Census Wall Chart PDF (Internet Archive copy of the National Statistics Directorate file)."""
import re, time, requests
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "census"
H = {"User-Agent": "tl-oda-research/0.1 (academic study)"}

q = """SELECT ?item ?itemLabel ?de WHERE { ?item wdt:P31 wd:Q1512109 .
 OPTIONAL { ?de schema:about ?item ; schema:isPartOf <https://de.wikipedia.org/> . }
 SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }"""
r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, headers=H, timeout=180).json()
items = pd.DataFrame([{k: v["value"] for k, v in b.items()} for b in r["results"]["bindings"]])
rows = []
for it in items.itertuples():
    if not isinstance(it.de, str):
        continue
    title = requests.utils.unquote(it.de.rsplit("/", 1)[1]).replace("_", " ")
    w = ""
    for attempt in range(4):
        time.sleep(1.0 + attempt * 2)
        resp = requests.get("https://de.wikipedia.org/w/api.php", params={"action": "parse", "page": title, "prop": "wikitext",
                            "format": "json", "redirects": 1}, headers=H, timeout=60)
        try:
            w = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
            break
        except ValueError:
            continue
    m = re.search(r"leben\s+([\d\.]+)\s+Menschen\s*\((\d{4})\)", w)
    m10 = re.search(r"([\d\.]+)\s+Einwohner\s*\(2010\)|2010[^\n]{0,40}?([\d\.]{4,})\s+Einwohner", w)
    rows.append({"item": it.item, "label": it.itemLabel, "de_title": title,
                 "pop_de": int(m.group(1).replace(".", "")) if m else None, "year_de": m.group(2) if m else None})
D = pd.DataFrame(rows)
D.to_csv(OUT / "dewiki_posto_population.csv", index=False, encoding="utf-8-sig")
print(D.year_de.value_counts(dropna=False))

u = ("https://web.archive.org/web/20110812113436/http://dne.mof.gov.tl/published/2010%20and%202011%20Publications/"
     "Wall%20Chart%20English/Wall%20Chart%20English%209%20June%202011.pdf")
rr = requests.get(u, headers=H, timeout=300)
print("wall chart", rr.status_code, rr.headers.get("Content-Type"), len(rr.content))
if rr.status_code == 200 and rr.content[:4] == b"%PDF":
    (OUT / "census2010_wallchart.pdf").write_bytes(rr.content)
