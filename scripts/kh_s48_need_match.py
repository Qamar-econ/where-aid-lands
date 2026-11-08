"""Spec 9-37 (exploratory): within a district, do projects go to the sector whose need is larger?

Sectors and needs (census 2019, z across districts): water and sanitation (DAC 140) - no flush toilet; education (110-114) -
child out of school; energy (230-236) - no electricity. Outcome and sample as kh_main.py.
(1) by sector: need + travel time + ln population + province FE, Freedman-Lane randomisation inference
(2) stacked: district FE + sector FE, regressor = the district's need for that sector; permutation of sector labels across
    projects; robustness with province x sector FE and ln population x sector.
Output: outputs/tables/kh_needmatch.csv, .txt
"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
src = (HERE / "kh_main.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_main.py"), "__name__": "kh_main_ns"}
sys.argv = sys.argv[:1]
exec(src[:src.index('say("== MAIN: one construct')], ns)
DF, MAIN, A0, counts, T, PROC = (ns[k] for k in ("DF", "MAIN", "A0", "counts", "T", "PROC"))
ns["RNG"] = np.random.default_rng(20261111)
s35 = (HERE / "kh_s35_infra_stacked.py").read_text(encoding="utf-8"); pp = {"np": np, "pd": pd}
exec(s35[s35.index("def _demean"):s35.index("# ---------------- stacked panel")], pp)
ppml = pp["ppml"]; RNG = np.random.default_rng(20261112)
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
pc = lambda b: 100 * (np.exp(b) - 1)

c = pd.read_csv(PROC / "kh_census.csv"); c = c[c.year == 2019].set_index("unit")
SECT = {"water": ({140}, "dep_sanitation"), "education": ({110, 111, 112, 113, 114}, "dep_school"), "energy": ({230, 231, 232, 233, 234, 235, 236}, "dep_electric")}
for k, (_, col) in SECT.items():
    v = DF.unit.map(c[col]); DF[f"need_{k}"] = (v - v.mean()) / v.std()
s3 = pd.to_numeric(A0.drop_duplicates("project_id").set_index("project_id").sector_code.astype(str).str.extract(r"(\d{3})")[0], errors="coerce")
lab = pd.Series(np.select([s3.isin(SECT[k][0]) for k in SECT], list(SECT), "none"), index=s3.index)
M = MAIN.assign(sec=MAIN.project_id.map(lab)); M = M[M.sec != "none"]
say("activities: " + ", ".join(f"{k} {M[M.sec == k].project_id.nunique()} (weight {M[M.sec == k].w.sum():.1f})" for k in SECT))


def add(label, b, s, p, n, extra=""):
    r = {"model": label, "pct": pc(b), "ci_lo": pc(b - 1.96 * s), "ci_hi": pc(b + 1.96 * s), "mde_pct": pc(2.8 * s), "p_ri": p, "p_cluster": 2 * norm.sf(abs(b / s)), "n": n}
    rows.append(r)
    say(f"  {label:<54s} {r['pct']:+6.0f}% per SD (95% CI {r['ci_lo']:+.0f}% to {r['ci_hi']:+.0f}%) MDE {r['mde_pct']:.0f}% | p RI {p:.3f} | "
        f"p cluster {r['p_cluster']:.3f} | n {n} {extra}")


say("== (1) by sector: own need, holding travel time")
for k in SECT:
    dk = counts(M[M.sec == k]).reset_index(drop=True); v = f"need_{k}"
    say(f"  corr(need_{k}, travel time) {dk[v].corr(dk.reach):+.2f}")
    f, r = ns["fit"](dk, [v, "reach"])
    add(f"{k}: own need (+ travel time)", r.params[v], r.bse[v], ns["ri_p"](dk, v, ["reach"], f, r.params[v], DRAWS), round(dk.N.sum(), 1))

say("== (2) within district: sector with the larger need")
d0 = counts(MAIN).reset_index(drop=True); units = d0.unit.values; K = list(SECT)
NEED = np.column_stack([d0[f"need_{k}"].values for k in K]); LNP = d0.lnpop.values; PRV = d0.prov.values.astype(object)


def fit2(lmap, full=False, se=True):
    g = M.assign(L=M.project_id.map(lmap)).groupby(["unit", "L"]).w.sum().unstack(fill_value=0).reindex(index=units, columns=K, fill_value=0)
    y = np.concatenate([g[k].values for k in K]); uu = np.tile(units, len(K)); x = np.concatenate([NEED[:, j] for j in range(len(K))])[:, None]
    sec = np.repeat(np.arange(len(K)), len(units))
    if full:
        x = np.column_stack([x] + [np.where(sec == j, np.tile(LNP, len(K)), 0) for j in range(1, len(K))])
        fe2 = np.tile(PRV, len(K)) + "|" + sec.astype(str)
    else:
        fe2 = sec
    return ppml(y.astype(float), x, uu, fe2, uu, se=se)


proj = M.drop_duplicates("project_id").set_index("project_id").sec
for full, label in [(False, "need of the sector, district + sector FE"), (True, "same + province x sector FE, ln pop x sector")]:
    b, V = fit2(proj, full); b, s = b[0], np.sqrt(V[0, 0]); bs = []
    for _ in range(DRAWS):
        bs.append(fit2(pd.Series(RNG.permutation(proj.values), index=proj.index), full, se=False)[0][0])
    add(label, b, s, (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs)), M.project_id.nunique())
pd.DataFrame(rows).to_csv(T / "kh_needmatch.csv", index=False)
(T / "kh_needmatch.txt").write_text("\n".join(log), encoding="utf-8")
