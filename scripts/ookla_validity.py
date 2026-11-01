"""Within-country validity of Ookla as a connectivity measure.

Timor-Leste: municipality Ookla tests/tiles per capita vs DHS 2009-10 and 2016 household mobile ownership (13 units);
             posto-level sparsity (zero-test postos), year-to-year stability, population share near any tested tile.
Nepal: district Ookla vs census 2011 household mobile / internet ownership (75 districts).
Output: outputs/tables/ookla_validity.csv, outputs/results_ookla.txt
"""
import math, requests
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import cKDTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = open(ROOT / "outputs/results_ookla.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
rows = []

# ---------- Timor-Leste ----------
P = pd.read_csv(ROOT / "data/processed/posto_covariates.csv")
dhs = pd.DataFrame(requests.get("https://api.dhsprogram.com/rest/dhs/data", params={"f": "json", "indicatorIds": "HC_HEFF_H_MPH", "countryIds": "TL",
                   "breakdown": "subnational", "perpage": 200}, timeout=120).json()["Data"])
dhs["mun"] = dhs.CharacteristicLabel.replace({"Cova Lima": "Covalima", "Liquica": "Liquiçá", "Lautem": "Lautém"})
M = P.groupby("municipio").agg(pop=("pop2022", "sum"), **{f"t{y}": (f"ook_mobile_{y}_tests", "sum") for y in [2019, 2022, 2025]},
                               **{f"k{y}": (f"ook_mobile_{y}_tiles", "sum") for y in [2019, 2022, 2025]}).reset_index()
for sv, yy in [("TL2009DHS", 2009), ("TL2016DHS", 2016)]:
    M = M.merge(dhs[dhs.SurveyId == sv][["mun", "Value"]].rename(columns={"mun": "municipio", "Value": f"dhs{yy}"}), on="municipio", how="left")
say(f"TL municipalities matched to DHS: {M.dhs2016.notna().sum()}/13")
for yy in [2019, 2022, 2025]:
    for sv in [2009, 2016]:
        for kind in ["t", "k"]:
            x = M[f"{kind}{yy}"] / M["pop"] * 1e4
            r, p = spearmanr(x, M[f"dhs{sv}"])
            rows.append({"country": "TL", "level": "municipality (13)", "ookla": f"{'tests' if kind == 't' else 'tiles'}/pop {yy}", "reference": f"DHS {sv} mobile ownership",
                         "spearman": r, "p": p})
    say(f"  Ookla {yy} tests/pop vs DHS 2009: rho={spearmanr(M[f't{yy}'] / M['pop'], M.dhs2009)[0]:+.2f};  vs DHS 2016: rho={spearmanr(M[f't{yy}'] / M['pop'], M.dhs2016)[0]:+.2f}")
nd = M[M.municipio != "Dili"]
say(f"  excl. Dili (12): Ookla 2019 tests/pop vs DHS 2016 rho={spearmanr(nd.t2019 / nd['pop'], nd.dhs2016)[0]:+.2f}; vs DHS 2009 rho={spearmanr(nd.t2019 / nd['pop'], nd.dhs2009)[0]:+.2f}")
for yy in [2019, 2022, 2025]:
    z = (P[f"ook_mobile_{yy}_tests"] == 0).sum()
    say(f"  postos with zero Ookla mobile tests in {yy}Q1: {z}/65; Dili 4 postos hold {P.loc[P.municipio == 'Dili', f'ook_mobile_{yy}_tests'].sum() / P[f'ook_mobile_{yy}_tests'].sum():.0%} of tests (pop share {P.loc[P.municipio == 'Dili', 'pop2022'].sum() / P.pop2022.sum():.0%})")
say(f"  stability across postos (Spearman tests/pop): 2019~2022 {spearmanr(P.ook_mobile_2019_tests / P.pop2022, P.ook_mobile_2022_tests / P.pop2022)[0]:+.2f}, "
    f"2019~2025 {spearmanr(P.ook_mobile_2019_tests / P.pop2022, P.ook_mobile_2025_tests / P.pop2022)[0]:+.2f}")


# population living within 1.5 km of any tested tile (TL)
def centres(csv):
    o = pd.read_csv(csv, dtype={"quadkey": str})
    out = []
    for qk in o.quadkey:
        x = y = 0; zz = len(qk)
        for i in range(zz):
            m = 1 << (zz - i - 1); c = int(qk[i])
            if c & 1: x |= m
            if c & 2: y |= m
        n = 2 ** zz
        out.append(((x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))))
    return np.array(out)


with rasterio.open(ROOT / "data/raw/worldpop/tls_ppp_2020_1km_UNadj.tif") as s:
    a = s.read(1); rr, cc = np.where(a > 0); xs, ys = rasterio.transform.xy(s.transform, rr, cc); pw = a[rr, cc]
kx = math.cos(math.radians(-8.8))
for yy in [2019, 2025]:
    c = centres(ROOT / f"data/raw/ookla/ookla_mobile_{yy}Q1_TL.csv")
    d, _ = cKDTree(np.c_[c[:, 0] * kx, c[:, 1]]).query(np.c_[np.array(xs) * kx, np.array(ys)])
    say(f"  TL population within 1.5 km of a tested tile, {yy}Q1: {pw[d * 111.32 <= 1.5].sum() / pw.sum():.0%}")

# ---------- Nepal ----------
N = pd.read_csv(ROOT / "data/processed/nepal/district_panel.csv")
for yy in ["2019", "2025"]:
    x = N[f"ook{yy}_tests"] / N.pop2011 * 1e5
    for ref in ["mobile_pct", "internet_pct"]:
        r, p = spearmanr(x, N[ref])
        rows.append({"country": "NP", "level": "district (75)", "ookla": f"tests/pop {yy}", "reference": f"census 2011 {ref}", "spearman": r, "p": p})
    say(f"NP Ookla {yy} tests/pop vs census 2011 mobile rho={spearmanr(x, N.mobile_pct)[0]:+.2f}, internet rho={spearmanr(x, N.internet_pct)[0]:+.2f}; zero-test districts {(N[f'ook{yy}_tests'] == 0).sum()}/75")
say(f"  Kathmandu valley share of Ookla 2019 tests {N.loc[N.capital == 1, 'ook2019_tests'].sum() / N.ook2019_tests.sum():.0%} vs population {N.loc[N.capital == 1, 'pop2011'].sum() / N.pop2011.sum():.0%}")
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/ookla_validity.csv", index=False)
LOG.close()
