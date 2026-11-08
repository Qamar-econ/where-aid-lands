"""Spec 9-38 (post hoc consistency check, run while finalising the site, 8 Nov 2026).

Question: are the travel-time, neighbour-pair and distance results (specs 9-27, 9-36) one contrast between the hubs
(Phnom Penh's four khans and the 25 provincial towns, Krong) and the countryside, or a gradient inside the countryside?
Each design is re-estimated on all districts and on the countryside alone:
  1 travel time x2, province FE (kh_main model): 141 districts vs the 112 rural districts
  2 neighbour pairs, pair FE (kh_s46 A): all 361 pairs vs rural-rural pairs vs pairs that include a hub
  3 straight-line distance and road detour in one model (kh_s46 B): all districts vs rural districts
Inference: Freedman-Lane randomisation inference within province, as in the source scripts (2,000 draws).
Output: outputs/tables/kh_hub_consistency.csv, .txt
"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
src = (HERE / "kh_s46_designs.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_s46_designs.py"), "__name__": "kh_s46_ns"}
sys.argv = [sys.argv[0], "0"]          # rebuild kh_s46's data and pairs only; its own randomisation draws are skipped
exec(src[:src.index("Mc = M[M.type.isin")].replace("for _ in range(DRAWS):", "for _ in range(0):"), ns)
M0 = ns["ns"]; M0["RNG"] = np.random.default_rng(20261113); RNG = np.random.default_rng(20261114)
DF, MAIN, counts, T, ppml = ns["DF"], ns["MAIN"], ns["counts"], ns["T"], ns["ppml"]
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
pc = lambda b: 100 * (np.exp(b) - 1)
DF["hub"] = DF.capital | DF.ADMIN_NAME.str.contains("Krong")
hub = DF.set_index("unit").hub


def add(test, sample, b, s, p_ri, p_cl, n):
    r = {"test": test, "sample": sample, "pct": pc(b), "ci_lo": pc(b - 1.96 * s), "ci_hi": pc(b + 1.96 * s), "p_ri": p_ri, "p_cluster": p_cl, "n": n}
    rows.append(r)
    say(f"  {test:<34s} {sample:<26s} {r['pct']:+6.0f}% (95% CI {r['ci_lo']:+.0f}% to {r['ci_hi']:+.0f}%) | p RI {p_ri:.3f} | p cluster {p_cl:.3f} | n {n}")


d0 = counts(MAIN).reset_index(drop=True); d0["hub"] = d0.unit.map(hub).astype(bool)
dr = d0[~d0.hub].reset_index(drop=True)
say(f"hubs {int(d0.hub.sum())} districts ({d0[d0.hub].N.sum():.1f} projects) | rural {len(dr)} districts ({dr.N.sum():.1f} projects)")

say("== 1 travel time x2, province fixed effects")
for lab, d in [("all districts", d0), ("rural districts only", dr)]:
    f, r = M0["fit"](d, ["reach"])
    add("travel time x2", lab, r.params["reach"], r.bse["reach"], M0["ri_p"](d, "reach", [], f, r.params["reach"], DRAWS), r.pvalues["reach"], len(d))

say("== 2 neighbour pairs, pair fixed effects")
S, P = ns["S"], ns["P"]
P["nhub"] = hub.reindex(P.a).values.astype(int) + hub.reindex(P.b).values.astype(int)
fv, e0, idx = ns["fv"], ns["e0"], ns["idx"]
umap = pd.Series(np.arange(len(ns["d0"])), index=ns["units"])
for lab, keep in [("all pairs", P.pair), ("rural-rural pairs", P.pair[P.nhub == 0]), ("pairs with a hub", P.pair[P.nhub >= 1])]:
    Ss = S[S.pair.isin(keep)]
    b, s = ns["fitA"](Ss, Ss.reach.values)
    rowpos = umap.reindex(Ss.unit).values; bs = []
    for _ in range(DRAWS):              # same permutation as kh_s46 A: travel-time residual shuffled within province
        e = e0.copy()
        for ix in idx:
            e[ix] = RNG.permutation(e0[ix])
        bs.append(ppml(Ss.N.values.astype(float), np.c_[(fv + e)[rowpos], Ss.lnpop.values], Ss.pair.values, Ss.c.values, se=False)[0][0])
    add("travel time x2, neighbour pairs", lab, b, s, (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs)), 2 * norm.sf(abs(b / s)), len(keep))

say("== 3 straight-line distance vs road detour, one model")
for lab, d in [("all districts", d0), ("rural districts only", dr)]:
    f, r = M0["fit"](d, ["dist", "detour"])
    for v, t in [("dist", "straight-line distance x2"), ("detour", "road detour x2")]:
        p = M0["ri_p"](d, v, [x for x in ["dist", "detour"] if x != v], f, r.params[v], DRAWS)
        add(t, lab, r.params[v], r.bse[v], p, r.pvalues[v], len(d))

# the all-district rows repeat estimates already published (kh_main_simple, kh_designs): same point estimates, but their
# randomisation p comes from other random draws, so the published p is carried over to keep one number per estimate
R = pd.DataFrame(rows); MS = pd.read_csv(T / "kh_main_simple.csv"); DS = pd.read_csv(T / "kh_designs.csv").set_index("model")
canon = {("travel time x2", "all districts"): MS[(MS.model == "physical access alone") & (MS["var"] == "reach")].iloc[0],
         ("travel time x2", "rural districts only"): MS[(MS.model == "rural districts: physical access alone") & (MS["var"] == "reach")].iloc[0],
         ("travel time x2, neighbour pairs", "all pairs"): DS.loc["travel time x2, neighbour pairs"],
         ("straight-line distance x2", "all districts"): DS.loc["straight-line distance x2 (detour fixed)"],
         ("road detour x2", "all districts"): DS.loc["road detour x2 (distance fixed)"]}
for (t, s), c in canon.items():
    i = R.index[(R.test == t) & (R["sample"] == s)][0]
    assert abs(R.at[i, "pct"] - c.pct) < 1e-6, (t, R.at[i, "pct"], c.pct)
    say(f"  published p RI carried over for {t} / {s}: {R.at[i, 'p_ri']:.3f} -> {c.p_ri:.3f}")
    R.at[i, "p_ri"] = c.p_ri
R.to_csv(T / "kh_hub_consistency.csv", index=False)
(T / "kh_hub_consistency.txt").write_text("\n".join(log), encoding="utf-8")
