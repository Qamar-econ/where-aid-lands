"""Fetch budget books, poverty reports, WorldPop, Copernicus DEM, OSM roads, WorldClim precipitation."""
import json, requests, zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
H = {"User-Agent": "Mozilla/5.0 (research)"}


def dl(sub, name, url, timeout=600):
    d = RAW / sub
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    if f.exists() and f.stat().st_size > 0:
        return f
    try:
        r = requests.get(url, headers=H, timeout=timeout)
        if r.status_code != 200:
            print("FAIL", r.status_code, sub, name)
            return None
        f.write_bytes(r.content)
        print("saved", sub, name, len(r.content), r.headers.get("Content-Type"))
        return f
    except Exception as e:
        print("ERR", sub, name, e)


BB5 = {
    2011: "https://www.laohamutuk.org/econ/OGE11/MF5DonorsEn.pdf",
    2013: "https://www.laohamutuk.org/econ/OGE13/BksApr2013/OGE13Bk5en.pdf",
    2014: "https://www.laohamutuk.org/econ/OGE14/Prop/OGE14Bk5Oct2013en.pdf",
    2015: "https://www.laohamutuk.org/econ/OGE15/BksJan15/BB5-31Jan15en.pdf",
    2016: "https://www.laohamutuk.org/econ/OGE16/BksMar2016/OGE16finalBB5en.pdf",
    2017: "https://www.laohamutuk.org/econ/OGE17/FinalBks/FinalOJE17Bk5Feb17en.pdf",
    2018: "https://www.laohamutuk.org/econ/OGE18/PropBks/BB5en.pdf",
    2019: "https://www.laohamutuk.org/econ/OGE19/final/BB5-Eng.pdf",
    2020: "https://www.laohamutuk.org/econ/OGE20/third/BB5_Eng.pdf",
    2021: "https://www.laohamutuk.org/econ/OGE21/Finalbooks/BB5_Eng.pdf",
    2022: "https://www.laohamutuk.org/econ/OGE22/books/BB5_EN_Aprovado.pdf",
}
for y, u in BB5.items():
    dl("budget_bb5", f"BB5_{y}.pdf", u)

dl("poverty", "wb_small_area_note_a.pdf", "https://openknowledge.worldbank.org/bitstreams/ba8b9e35-2944-52a7-8de6-c225df89d48e/download")
dl("poverty", "wb_small_area_note_b.pdf", "https://openknowledge.worldbank.org/bitstreams/5fd87c46-95b9-514b-9de5-4dde34ea907e/download")

for y in [2005, 2010, 2015, 2020]:
    dl("worldpop", f"tls_ppp_{y}_1km_UNadj.tif",
       f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{y}/TLS/tls_ppp_{y}_1km_Aggregated_UNadj.tif")
dl("worldpop", "tls_ppp_2020_100m.tif", "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/TLS/tls_ppp_2020.tif")

for lat in ["S09", "S10"]:
    for lon in ["E124", "E125", "E126", "E127"]:
        n = f"Copernicus_DSM_COG_30_{lat}_00_{lon}_00_DEM"
        dl("dem", n + ".tif", f"https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif")

q = '[out:json][timeout:600];area["ISO3166-1"="TL"][admin_level=2]->.a;way["highway"](area.a);out tags geom;'
d = RAW / "osm"; d.mkdir(parents=True, exist_ok=True)
if not (d / "tl_highways.json").exists():
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=H, timeout=900)
    print("overpass", r.status_code, len(r.content))
    if r.status_code == 200:
        (d / "tl_highways.json").write_bytes(r.content)
qp = '[out:json][timeout:300];area["ISO3166-1"="TL"][admin_level=2]->.a;(node["amenity"~"hospital|clinic|doctors|school|bank"](area.a);node["place"~"town|city|village"](area.a););out;'
if not (d / "tl_pois.json").exists():
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": qp}, headers=H, timeout=600)
    print("overpass pois", r.status_code, len(r.content))
    if r.status_code == 200:
        (d / "tl_pois.json").write_bytes(r.content)

f = dl("climate", "wc2.1_5m_prec.zip", "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_5m_prec.zip", timeout=900)
if f and not (RAW / "climate" / "wc2.1_5m_prec").exists():
    with zipfile.ZipFile(f) as z:
        z.extractall(RAW / "climate" / "wc2.1_5m_prec")
