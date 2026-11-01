"""Download BB5 2025 PDF and peek at its text; list Ookla parquet sizes; inspect COD admin points."""
import requests, re
import pdfplumber, geopandas as gpd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
H = {"User-Agent": "research/0.1"}

d = RAW / "budget_bb5"
d.mkdir(parents=True, exist_ok=True)
f = d / "BB5_2025.pdf"
if not f.exists():
    f.write_bytes(requests.get("https://www.laohamutuk.org/econ/OGE25/Finalbooks/04.09.02.DevPartnersEn.pdf", headers=H, timeout=300).content)
with pdfplumber.open(f) as pdf:
    print("pages", len(pdf.pages))
    for i in [0, 1, 2, 3, 10, 30, 60, 100]:
        if i < len(pdf.pages):
            t = pdf.pages[i].extract_text() or ""
            print(f"--- page {i} ---\n", t[:1500])
    # find pages mentioning municipality breakdown
    hits = []
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ""
        if re.search(r"Municipal|Oecusse|Viqueque", t):
            hits.append(i)
    print("pages mentioning municipalities:", hits[:80], len(hits))

for typ in ["mobile", "fixed"]:
    for yq in [("2019", "1"), ("2022", "1"), ("2025", "1"), ("2026", "2")]:
        u = f"https://ookla-open-data.s3.amazonaws.com/?list-type=2&prefix=parquet/performance/type={typ}/year={yq[0]}/quarter={yq[1]}/"
        x = requests.get(u, timeout=60).text
        print(typ, yq, re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", x))

pts = gpd.read_file(RAW / "cod_ab/tls_admin_boundaries.shp/tls_adminpoints.shp")
print(pts.shape, list(pts.columns))
print(pts.drop(columns="geometry").head(5).T)
