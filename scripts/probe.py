"""Probe availability of candidate data sources (prints status + size only)."""
import requests, sys

URLS = {
    "geoboundaries_adm2": "https://www.geoboundaries.org/api/current/gbOpen/TLS/ADM2/",
    "geoboundaries_adm3": "https://www.geoboundaries.org/api/current/gbOpen/TLS/ADM3/",
    "hdx_cod_ab": "https://data.humdata.org/api/3/action/package_show?id=cod-ab-tls",
    "hdx_rwi": "https://data.humdata.org/api/3/action/package_search?q=relative%20wealth%20index%20timor",
    "hdx_pop": "https://data.humdata.org/api/3/action/package_search?q=timor-leste%20population%20census",
    "aiddata_tl_github": "https://api.github.com/repos/AidData-WM/public_datasets/contents/geocoded",
    "wb_projects": "https://search.worldbank.org/api/v2/projects?format=json&countrycode_exact=TP&rows=5",
    "dportal_act": "https://d-portal.org/q.json?from=act,country&country_code=TL&limit=3",
    "dportal_loc": "https://d-portal.org/q.json?from=act,location,country&country_code=TL&limit=3",
    "ookla_s3": "https://ookla-open-data.s3.amazonaws.com/?list-type=2&prefix=parquet/performance/type=mobile/year=2023/quarter=1/",
    "cop_dem": "https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_S09_00_E125_00_DEM/Copernicus_DSM_COG_30_S09_00_E125_00_DEM.tif",
    "laohamutuk_bb5_2025": "https://www.laohamutuk.org/econ/OGE25/Finalbooks/04.09.02.DevPartnersEn.pdf",
    "citypop": "https://www.citypopulation.de/en/easttimor/admin/",
}

for k, u in URLS.items():
    try:
        r = requests.get(u, timeout=40, stream=True, headers={"User-Agent": "research-probe/0.1"})
        size = r.headers.get("Content-Length")
        head = r.raw.read(400, decode_content=True)
        print(f"{k}: {r.status_code} {r.headers.get('Content-Type')} len={size}")
        print("   ", head[:300].decode("utf-8", "replace").replace("\n", " "))
        r.close()
    except Exception as e:
        print(f"{k}: ERROR {e}", file=sys.stdout)
