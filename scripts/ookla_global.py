"""Ookla mobile tests per capita by country (2019Q1 and 2025Q1): where do Timor-Leste and Nepal stand among LMICs?
Country assignment by tile centre (Natural Earth 1:50m admin-0); population from World Bank API.
Output: outputs/tables/ookla_country_density.csv
"""
import math, os, requests, zipfile, io
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(os.environ["TEMP"]) / "tloda" / "ookla_tmp"; TMP.mkdir(parents=True, exist_ok=True)
NE = ROOT / "data/raw/refs/ne_50m_admin_0_countries"
if not NE.exists():
    r = requests.get("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip", timeout=300, headers={"User-Agent": "research"})
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(NE)
W = gpd.read_file(next(NE.glob("*.shp")))[["ADM0_A3", "NAME", "INCOME_GRP", "geometry"]]
pop = requests.get("https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2023&format=json&per_page=400", timeout=120).json()[1]
POP = {p["countryiso3code"]: p["value"] for p in pop if p["value"]}
inc = requests.get("https://api.worldbank.org/v2/country?format=json&per_page=400", timeout=120).json()[1]
INC = {c["id"]: c["incomeLevel"]["value"] for c in inc}


def centre(qk):
    x = y = 0; z = len(qk)
    for i in range(z):
        m = 1 << (z - i - 1); c = int(qk[i])
        if c & 1: x |= m
        if c & 2: y |= m
    n = 2 ** z
    return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))


rows = []
for yy in ["2019", "2025"]:
    key = f"parquet/performance/type=mobile/year={yy}/quarter=1/{yy}-01-01_performance_mobile_tiles.parquet"
    f = TMP / Path(key).name
    if not f.exists():
        with requests.get("https://ookla-open-data.s3.amazonaws.com/" + key, stream=True, timeout=1800) as r:
            with open(f, "wb") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
    df = pq.read_table(f, columns=["quadkey", "tests", "avg_d_kbps"]).to_pandas()
    # aggregate to zoom-10 parents first (fast), then assign country by parent centre
    df["qk10"] = df.quadkey.str[:10]
    g = df.groupby("qk10").agg(tests=("tests", "sum"), tiles=("tests", "size")).reset_index()
    c = g.qk10.map(centre)
    gg = gpd.sjoin(gpd.GeoDataFrame(g, geometry=gpd.points_from_xy([p[0] for p in c], [p[1] for p in c]), crs=4326), W, predicate="within")
    a = gg.groupby("ADM0_A3").agg(tests=("tests", "sum"), tiles16=("tiles", "sum")).reset_index()
    a["year"] = yy
    rows.append(a)
    try:
        f.unlink()
    except PermissionError:
        pass
A = pd.concat(rows).merge(W.drop(columns="geometry"), on="ADM0_A3")
A["pop"] = A.ADM0_A3.map(POP); A["income"] = A.ADM0_A3.map(INC)
A["tests_per_100k"] = A.tests / A["pop"] * 1e5
A.to_csv(ROOT / "outputs/tables/ookla_country_density.csv", index=False)
for yy in ["2019", "2025"]:
    s = A[(A.year == yy) & A.income.isin(["Low income", "Lower middle income"]) & (A["pop"] > 3e5)].sort_values("tests_per_100k", ascending=False).reset_index(drop=True)
    print(f"\n{yy} Q1 mobile tests per 100k, low & lower-middle income (n={len(s)}); median {s.tests_per_100k.median():.1f}")
    for iso in ["TLS", "NPL"]:
        r = s[s.ADM0_A3 == iso]
        if len(r):
            print(f"  {iso}: {r.tests_per_100k.iloc[0]:.1f} per 100k, rank {r.index[0] + 1}/{len(s)}, tests {r.tests.iloc[0]:.0f}")
    print(s.head(8)[["NAME", "tests_per_100k"]].round(1).to_string(index=False)); print(s.tail(5)[["NAME", "tests_per_100k"]].round(1).to_string(index=False))
