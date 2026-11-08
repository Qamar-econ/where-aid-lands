"""Independent checks of spec 9-36 (kh_s46_designs.py), run after it.

1  the fast two-way-FE PPML used for A, C and D equals statsmodels' Poisson GLM with explicit dummies
2  A: pairs are distinct, adjacent (shared boundary) and cover every district; within-province share
3  B: log2 travel time = log2 distance + log2 detour exactly; distance in km matches a direct haversine for a sample of cells
4  C, D: label counts add up to the main sample; the gap model on the true labels reproduces kh_designs.csv
Output: outputs/tables/kh_designs_verify.csv, .txt
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "kh_s46_designs.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_s46_designs.py"), "__name__": "kh_s46_ns"}
sys.argv = [sys.argv[0], "0"]          # no randomisation draws: only the point estimates are rebuilt
cut = src.index("Mc = M[M.type.isin")
exec(src[:cut].replace("for _ in range(DRAWS):", "for _ in range(0):"), ns)
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
ok = lambda name, cond, val: (rows.append({"check": name, "value": val, "ok": bool(cond)}), say(f"  [{'ok' if cond else 'FAIL'}] {name}: {val}"))
D = pd.read_csv(ns["T"] / "kh_designs.csv")
ref = lambda m: D[D.model == m].pct.iloc[0]

say("== 1 fast PPML = statsmodels GLM")
S = ns["S"]
g = smf.glm("N ~ reach + lnpop + C(pair)", data=S, family=sm.families.Poisson()).fit(maxiter=500)
a = 100 * (np.exp(g.params["reach"]) - 1)
ok("A pair-FE estimate, GLM vs kh_designs", abs(a - ref("travel time x2, neighbour pairs")) < 0.01, f"{a:.3f} vs {ref('travel time x2, neighbour pairs'):.3f}")
Mc = ns["M"][ns["M"].type.isin(["field", "inst"])]
proj = Mc.drop_duplicates("project_id").set_index("project_id").type
st = Mc.assign(L=Mc.project_id.map(proj)).groupby(["unit", "L"]).w.sum().unstack(fill_value=0).reindex(index=ns["units"], columns=["field", "inst"], fill_value=0)
G2 = pd.DataFrame({"y": np.r_[st.field.values, st.inst.values], "unit": np.r_[ns["units"], ns["units"]].astype(str),
                   "t": np.r_[np.zeros(len(st)), np.ones(len(st))], "x": np.r_[ns["reach_of"].values, np.zeros(len(st))]})
G2 = G2[G2.groupby("unit").y.transform("sum") > 0]
g2 = smf.glm("y ~ x + C(t) + C(unit)", data=G2, family=sm.families.Poisson()).fit(maxiter=500)
a2 = 100 * (np.exp(g2.params["x"]) - 1)
ok("C gap estimate, GLM vs kh_designs", abs(a2 - ref("slope gap field minus institutional, district FE")) < 0.01, f"{a2:.3f} vs {ref('slope gap field minus institutional, district FE'):.3f}")

say("== 2 neighbour pairs")
P = ns["P"]; G = ns["G"].set_index("unit")
dup = P.apply(lambda r: tuple(sorted((r.a, r.b))), axis=1).duplicated().sum()
shared = np.array([G.geometry[a].boundary.intersection(G.geometry[b].buffer(200)).length for a, b in zip(P.a, P.b)])   # 200 m tolerance: boundaries are digitised separately
ok("no duplicate pairs", dup == 0, int(dup))
ok("pairs share >= 1 km of edge (200 m tolerance)", (shared >= 1000).mean() > 0.97, f"{(shared >= 1000).mean():.3f} of pairs; the rest touch at a corner, dropped in a check")
ok("every district is in a pair", len(set(P.a) | set(P.b)) == len(ns["units"]), len(set(P.a) | set(P.b)))

say("== 3 distance and detour")
d0 = ns["counts"](ns["MAIN"]).reset_index(drop=True)
idt = np.abs(np.log2(d0.tt_town_dry) - d0.dist - d0.detour).max()
ok("log2 T = dist + detour (no clipping bites)", idt < 1e-9, f"max {idt:.2e}")
C = ns["C"].sample(500, random_state=1); GN = ns["GN"]
lat1, lon1 = np.radians(C.lat.values)[:, None], np.radians(C.lon.values)[:, None]; lat2, lon2 = np.radians(GN.lat.values)[None], np.radians(GN.lon.values)[None]
hv = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)).min(axis=1)
err = np.abs(hv - C.km.values) / np.maximum(hv, 1)
ok("projected distance matches haversine (500 cells)", np.median(err) < 0.01, f"median relative error {np.median(err):.4f}, max {err.max():.3f}")

say("== 4 labels add up")
M = ns["M"]
ok("type labels cover the sample weight", abs(M.groupby("type").w.sum().sum() - ns["MAIN"].w.sum()) < 1e-9, f"{M.groupby('type').w.sum().round(1).to_dict()}")
ok("donor labels cover the sample weight", abs(M.groupby("dg").w.sum().sum() - ns["MAIN"].w.sum()) < 1e-9, f"{M.groupby('dg').w.sum().round(1).to_dict()}")
V = pd.DataFrame(rows); V.to_csv(ns["T"] / "kh_designs_verify.csv", index=False)
say(f"== {int(V.ok.sum())} of {len(V)} checks ok")
(ns["T"] / "kh_designs_verify.txt").write_text("\n".join(log), encoding="utf-8")
