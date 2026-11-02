"""External validation of the physical-access measures, and 2015 population weights.

1. Our travel times against the Malaria Atlas Project maps, clipped to Cambodia from the MAP GeoServer:
   - travel time to cities of 50,000+ in 2015 (Weiss et al. 2018, Nature): compared with our time to towns of 20,000+;
   - motorized travel time to healthcare in 2019 (Weiss et al. 2020, Nature Medicine): compared with our time to hospitals.
   Cell level (1 km cells of kh_cells_tt.csv, Spearman) and district level (population-weighted means, log, Pearson).
2. The headline model with an index built from someone else's map: -ln(Weiss travel time to cities), district means.
3. Our index with 2015 instead of 2020 population weights (kh_s03 KH_POPYEAR=2015, tag pop2015).
Effects: PPML as in kh_s04; p = largest of clustered and Conley 50/100 km.
Output: outputs/tables/kh_external_validation.txt
"""
import numpy as np, pandas as pd, rasterio
from scipy.stats import spearmanr, norm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCD = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
g = {"__file__": str(ROOT / "scripts/kh_s04_models.py")}; exec(src[:src.index("\nbase = A[A.level")], g)
A, outcome, ppml, conley_se = g["A"], g["outcome"], g["ppml"], g["conley_se"]
d0 = outcome(A[A.level == "adm2"])


def eff(series, label):
    d = d0.copy(); s_ = d.unit.map(series); d["xv"] = (s_ - s_.mean()) / s_.std(); d = d[d.xv.notna()]
    r = ppml("N ~ xv + lnpop + C(prov)", d); b = r.params["xv"]
    p = max([r.pvalues["xv"]] + [2 * norm.sf(abs(b / conley_se(r, d, "xv", c))) for c in (50, 100)])
    say(f"  {label:<62s} {100 * (np.exp(b) - 1):+6.0f}%  p {p:.3f}  n {len(d)}")


cells = pd.read_csv(PROCD / "kh_cells_tt.csv")
def sample(name):
    with rasterio.open(RAW / f"map_access/{name}.tif") as s:
        v = np.array([x[0] for x in s.sample(zip(cells.lon, cells.lat))], dtype=float)
        if s.nodata is not None:
            v[v == s.nodata] = np.nan
    v[v < 0] = np.nan
    return v
cells["w_city"] = sample("Accessibility__201501_Global_Travel_Time_to_Cities")
cells["w_hc"] = sample("Accessibility__202001_Global_Motorized_Travel_Time_to_Healthcare")
ok = cells.w_city.notna()
say("== 1. agreement with the Malaria Atlas Project maps")
say(f"  cells with both values: {ok.sum():,} of {len(cells):,}")
say(f"  cell level, Spearman(our time to towns 20k+, Weiss 2018 time to cities 50k+) = {spearmanr(cells.tt_town_dry[ok], cells.w_city[ok]).correlation:.2f}")
wm = lambda c: cells[cells[c].notna()].groupby("k").apply(lambda x: np.average(x[c], weights=x["pop"]), include_groups=False)
D = pd.DataFrame({"w_city": wm("w_city"), "w_hc": wm("w_hc")}).clip(lower=1)   # 0 min in city cores; 1 min floor for logs
D.index = D.index.astype(float).astype(int)
acc = pd.read_csv(PROCD / "kh_access.csv").set_index("k")
D = D.join(acc[["tt_town_dry", "tt_hosp_dry", "access_pc1"]], how="inner")
D[["w_city", "w_hc"]].rename_axis("unit").to_csv(PROCD / "kh_weiss_district.csv")   # used by kh_main.py
say(f"  district level, corr(log our time to towns, log Weiss time to cities) = {np.corrcoef(np.log(D.tt_town_dry), np.log(D.w_city))[0, 1]:.2f}  ({len(D)} districts)")
say(f"  district level, corr(log our time to hospital, log Weiss 2020 time to healthcare) = {np.corrcoef(np.log(D.tt_hosp_dry), np.log(D.w_hc))[0, 1]:.2f}")
say(f"  district level, corr(our physical-access index, -log Weiss time to cities) = {np.corrcoef(D.access_pc1, -np.log(D.w_city))[0, 1]:.2f}")
say(f"  median district travel time to the nearest destination: ours to towns {D.tt_town_dry.median():.0f} min, Weiss to cities {D.w_city.median():.0f} min")

say("\n== 2. the headline model with external accessibility measures")
eff(acc.access_pc1, "our physical-access index (main)")
eff(-np.log(D.w_city), "-ln Weiss et al. 2018 travel time to cities (district mean)")
eff(-np.log(D.w_hc), "-ln Weiss et al. 2020 travel time to healthcare")

say("\n== 3. 2015 population weights")
p15 = pd.read_csv(PROCD / "kh_access_pop2015.csv").set_index("k")
say(f"  corr with main index {p15.access_pc1.corr(acc.access_pc1):.3f}")
eff(p15.access_pc1, "index with 2015 WorldPop weights")
(ROOT / "outputs/tables/kh_external_validation.txt").write_text("\n".join(log), encoding="utf-8")
