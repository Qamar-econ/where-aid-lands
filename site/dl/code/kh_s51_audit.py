"""Pipeline audit (8 Nov 2026), post hoc checks run while auditing the final results, in pipeline order.

A1 towns: the main travel time uses GeoNames places of 20,000+, which lists villages with commune populations and misses six
   provincial capitals. Re-estimate with travel time to the 25 provincial capitals (kh_access_towncap.csv, same network).
A2 hubs: Phnom Penh's outer khans sit in two large mixed units counted as rural (116008001, 116008002); re-estimate the
   rural-only model with both counted as hubs, and with them dropped.
A3 sector codes: field/institutional/infrastructure use the first DAC code; 9 activities list several codes.
A4 persistence: past projects 2005-2012 carry the same town-name geocoding issue; re-estimate with past projects counted only
   from activities not flagged as town/province-name geocodes.
Output: outputs/tables/kh_audit.csv, .txt
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
src = (HERE / "kh_main.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_main.py"), "__name__": "kh_main_ns"}
sys.argv = sys.argv[:1]
exec(src[:src.index('say("== MAIN: one construct')], ns)
DF, MAIN, counts, T, PROC = (ns[k] for k in ("DF", "MAIN", "counts", "T", "PROC"))
ns["RNG"] = np.random.default_rng(20261116)
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
pc = lambda b: 100 * (np.exp(b) - 1)
DF["hub"] = DF.capital | DF.ADMIN_NAME.str.contains("Krong")


def est(check, label, d, x, ctrl=()):
    f, r = ns["fit"](d, [x] + list(ctrl)); b, s = r.params[x], r.bse[x]
    p = ns["ri_p"](d, x, list(ctrl), f, b, DRAWS)
    rows.append({"check": check, "model": label, "pct": pc(b), "ci_lo": pc(b - 1.96 * s), "ci_hi": pc(b + 1.96 * s), "p_ri": p, "p_cluster": r.pvalues[x], "n": len(d)})
    say(f"  {check} {label:<58s} {pc(b):+6.0f}% (95% CI {pc(b - 1.96 * s):+.0f}% to {pc(b + 1.96 * s):+.0f}%) | p RI {p:.3f} | p cluster {r.pvalues[x]:.3f} | n {len(d)}")


say("== A1 travel time to the 25 provincial capitals instead of GeoNames places of 20,000+")
tc = pd.read_csv(PROC / "kh_access_towncap.csv").set_index("k").tt_town_dry
DF["reach_cap"] = np.log2(DF.unit.map(tc))
d0 = counts(MAIN).reset_index(drop=True)
say(f"  corr(main travel time, capitals travel time) {d0.reach.corr(d0.reach_cap):.3f}; median {d0.tt_town_dry.median():.0f} vs {d0.unit.map(tc).median():.0f} min")
est("A1", "travel time x2, provincial capitals, all districts", d0, "reach_cap")
est("A1", "travel time x2, provincial capitals, rural districts", d0[~d0.hub].reset_index(drop=True), "reach_cap")

say("== A2 Phnom Penh's outer khans (mixed units 116008001, 116008002)")
OUT = {116008001, 116008002}
dr = d0[~d0.hub & ~d0.unit.isin(OUT)].reset_index(drop=True)
est("A2", "travel time x2, rural districts, outer-khan units dropped", dr, "reach")
say(f"  projects in the two outer units: {d0[d0.unit.isin(OUT)].N.sum():.1f}; hub share if counted as hubs: "
    f"{(d0[d0.hub | d0.unit.isin(OUT)].N.sum() / d0.N.sum()):.3f} of projects, {(d0[d0.hub | d0.unit.isin(OUT)].pop_2019.sum() / d0.pop_2019.sum()):.3f} of people")

say("== A4 persistence with the same town/province-name rule applied to 2005-2012 locations")
import re
lad = {"__file__": str(HERE / "kh_s06_ladder.py")}
exec((HERE / "kh_s06_ladder.py").read_text(encoding="utf-8").split("LADDER = [")[0], lad)
L = lad["L"].copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce"); g = lad["g"]
s38 = (HERE / "kh_s38_geocode_extract.py").read_text(encoding="utf-8")
PROV = eval(s38[s38.index("PROV = ") + 7:s38.index("]", s38.index("PROV = ")) + 1]); TOWN = re.search(r'TOWN = r"([^"]+)"', s38).group(1)
gg = g[g.dgroup == "EU/US"].copy(); gg["pn"] = gg.place_name.astype(str).str.split(",").str[0].str.strip()
pl = gg.pn.str.lower().str.replace(r"\s+province$", "", regex=True).str.replace("é", "e")
gg["flag"] = pl.isin([x.replace("é", "e") for x in PROV]) | gg.pn.str.lower().str.fullmatch(TOWN)
flagged = set(zip(gg[gg.flag].project_id, gg[gg.flag].unit))
PR = L[L.yr.between(2005, 2012) & (L.family != "ict") & L.unit.isin(DF.unit)].copy()
PR["flag"] = [(a, b) in flagged for a, b in zip(PR.project_id, PR.unit)]
say(f"  past activities {PR.project_id.nunique()}, weight {PR.w.sum():.1f}; flagged weight {PR[PR.flag].w.sum():.1f} ({PR[PR.flag].project_id.nunique()} activities)")
DF["prior_n"] = np.log1p(DF.unit.map(PR.groupby("unit").w.sum()).fillna(0))
DF["prior_n_drop"] = np.log1p(DF.unit.map(PR[~PR.flag].groupby("unit").w.sum()).fillna(0))
pop = DF.set_index("unit").pop_2019; prov = DF.set_index("unit").prov
sp = PR[~PR.flag].groupby("unit").w.sum()
for _, r in PR[PR.flag].iterrows():       # spread over the province by population, as the main rule does for PROVINCE/UNCLEAR
    pool = pop[prov == prov[r.unit]]; sp = sp.add(r.w * pool / pool.sum(), fill_value=0)
DF["prior_n_spread"] = np.log1p(DF.unit.map(sp).fillna(0))
d0 = counts(MAIN).reset_index(drop=True)
for v, lab in [("prior_n", "past projects as published (= kh_s45 model 6)"), ("prior_n_drop", "past projects, flagged locations dropped"),
               ("prior_n_spread", "past projects, flagged locations spread over province")]:
    est("A4", lab, d0, v, ["reach"])
pd.DataFrame(rows).to_csv(T / "kh_audit.csv", index=False)
(T / "kh_audit.txt").write_text("\n".join(log), encoding="utf-8")
