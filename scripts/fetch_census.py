"""Census populations of administrative posts (2004, 2010, 2015, 2022) from Wikidata (P1082 with point-in-time)."""
import requests
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "census"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "tl-oda-research/0.1 (academic study)"}


def sparql(q):
    r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, headers=H, timeout=180)
    r.raise_for_status()
    return pd.DataFrame([{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]])


pp = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "titles": "Laga Administrative Post",
                  "prop": "pageprops", "format": "json", "redirects": 1}, headers=H, timeout=60).json()
qid = next(iter(pp["query"]["pages"].values()))["pageprops"]["wikibase_item"]
cls = sparql(f'SELECT ?cls ?clsLabel WHERE {{ wd:{qid} wdt:P31 ?cls . SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')
print(qid, cls)
c = cls.cls.iloc[0].rsplit("/", 1)[1]
df = sparql(f"""SELECT ?item ?itemLabel ?muniLabel ?pop ?date WHERE {{
  ?item wdt:P31 wd:{c} . OPTIONAL {{ ?item wdt:P131 ?muni . }}
  ?item p:P1082 ?st . ?st ps:P1082 ?pop . OPTIONAL {{ ?st pq:P585 ?date . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}""")
df["year"] = df["date"].str[:4] if "date" in df else None
df.to_csv(OUT / "wikidata_posto_population.csv", index=False, encoding="utf-8-sig")
print(df.shape, df.item.nunique(), "posts")
print(df.year.value_counts())
print(df.pivot_table(index="itemLabel", columns="year", values="pop", aggfunc="first").head(70).to_string())
