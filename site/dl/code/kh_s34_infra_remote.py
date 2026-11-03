"""Does infrastructure pull in later aid where districts were remote? (exploratory, added after the first results)

Same district-year panel as kh_s31 (2000-2020, PPML with district and year FE, clustered by district), GODAD transport and
energy activities as the treatment (lags 1-5, leads 1-2), estimated separately for districts below and above the median of
the physical-access index. Also counts how many infrastructure events fall in each half, and adds the hand-coded major
roads (kh_s32) as a check.
Output: outputs/tables/kh_infra_remote.txt
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, ROOT = ns["DF"], ns["L"], ns["ROOT"]
T, PROCD = ROOT / "outputs/tables", ROOT / "data/processed/cambodia"
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

L = L.copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce"); L = L[L.unit.isin(DF.unit) & L.yr.between(1990, 2022)]
grid = pd.MultiIndex.from_product([DF.unit, range(1990, 2024)], names=["unit", "yr"])
cnt = lambda f: f.groupby(["unit", "yr"]).w.sum().reindex(grid, fill_value=0)
ind = lambda f: f.groupby(["unit", "yr"]).size().reindex(grid, fill_value=0).clip(upper=1)
remote = DF.set_index("unit").access < DF.access.median()
MJ = pd.read_csv(PROCD / "kh_major_roads_units.csv")


def run(events, label):
    I = ind(events)
    for yname, Y in [("non-infrastructure", cnt(L[~L.family.isin(["infra", "ict"])])), ("social", cnt(L[L.family == "social"]))]:
        P = pd.DataFrame({"y": Y}).reset_index()
        for l in range(1, 6):
            P[f"l{l}"] = I.groupby(level=0).shift(l).values
        for l in (1, 2):
            P[f"f{l}"] = I.groupby(level=0).shift(-l).values
        P = P[P.yr.between(2000, 2020)].fillna(0)
        for half, keep in [("remote half", True), ("connected half", False)]:
            d = P[P.unit.map(remote) == keep]; d = d[d.groupby("unit").y.transform("sum") > 0]
            u_, y_ = I.index.get_level_values(0), I.index.get_level_values(1)
            n_ev = int(I[u_.isin(d.unit.unique()) & (y_ >= 1995) & (y_ <= 2019)].sum())
            m = smf.glm("y ~ l1 + l2 + l3 + l4 + l5 + f1 + f2 + C(unit) + C(yr)", data=d, family=sm.families.Poisson()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]}, maxiter=500)
            out = []
            for vs in (["l1", "l2", "l3", "l4", "l5"], ["f1", "f2"]):
                b = m.params[vs].sum(); s = float(np.sqrt(np.ones(len(vs)) @ m.cov_params().loc[vs, vs].values @ np.ones(len(vs))))
                out.append(f"{100 * (np.exp(b) - 1):+.0f}% (95% CI {100 * (np.exp(b - 1.96 * s) - 1):+.0f}% to {100 * (np.exp(b + 1.96 * s) - 1):+.0f}%, p {2 * norm.sf(abs(b / s)):.2f})")
            say(f"  {label:<26s} {yname:<19s} {half:<15s} events {n_ev:>3d} | lags 1-5 {out[0]} | leads {out[1]}")


say("districts split at the median of the physical-access index (remote = below)")
ev = L[L.family == "infra"][["unit", "yr"]]
say(f"GODAD transport/energy activity-district events 1995-2019: remote half {int(ev[ev.unit.map(remote)].query('1995 <= yr <= 2019').shape[0])}, "
    f"connected half {int(ev[~ev.unit.map(remote)].query('1995 <= yr <= 2019').shape[0])}")
say(f"major ADB/Japan road districts: remote half {int(remote.reindex(MJ.unit.unique()).sum())}, connected half {int((~remote.reindex(MJ.unit.unique())).sum())}")
run(ev, "GODAD infrastructure")
run(pd.concat([ev, MJ[["unit", "start_year"]].rename(columns={"start_year": "yr"})]), "GODAD + major roads")
(T / "kh_infra_remote.txt").write_text("\n".join(log), encoding="utf-8")
