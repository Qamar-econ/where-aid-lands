"""Pre-specified estimation (사전분석계획_네팔2019_2024.md): district x planning-year PPML, M1 / M2 + R1-R10, Holm, RI, MDE.
Output: outputs/results_np2_panel.txt, outputs/tables/np2_panel_results.csv, data/processed/nepal/np2_panel.csv
"""
import re, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PL = pd.read_csv(ROOT / "data/processed/nepal/np2_planning.csv")
ST = pd.read_csv(ROOT / "data/processed/nepal/np2_project_status.csv")
G = pd.read_csv(ROOT / "data/processed/nepal/np2_geocodes.csv")
CONN = pd.read_csv(ROOT / "data/processed/nepal/np2_connectivity.csv")
D = pd.read_csv(ROOT / "data/processed/nepal/districts.csv")
A = pd.read_csv(ROOT / "data/processed/nepal/district_access.csv")
YEARS = range(2019, 2025)

prj = PL.merge(ST, on="uid")
prj = prj[prj.best_precision <= 4].copy()
prj["covid"] = prj.title.str.contains(r"(?i)covid|coronavirus|pandemic|vaccin")
asg = G[G.uid.isin(prj.uid)].groupby(["uid", "district"]).precision.min().reset_index()
asg = asg.merge(prj[["uid", "planning_year", "method", "dclass", "usd", "covid", "n_districts", "best_precision"]], on="uid")
asg["w"] = 1 / asg.n_districts

base = CONN[["k", "year", "net_idx", "mob_tests_pc", "mob_mbps", "mob_tiles_km2", "fix_tests_pc", "pop"]].copy()
base = base.merge(D[["k", "province", "capital", "deprivation"]], on="k").merge(A[["k", "tt_dry_min", "paved_km"]], on="k")
base = base.merge(D[["k", "area_km2"]], on="k")
z = lambda s: (s - s.mean()) / s.std()
base["road"] = z(-np.log(base.tt_dry_min))
base["paved"] = z(np.log1p(base.paved_km / base.area_km2 * 100))
base["dep"] = z(base.deprivation)
base["lnpop"] = np.log(base["pop"])
for c in ["mob_tests_pc", "mob_mbps", "mob_tiles_km2", "fix_tests_pc"]:
    base[c + "_z"] = z(np.log1p(base[c]))
base["pt"] = base.province.astype(str) + "_" + base.year.astype(str)


def outcome(a, shift=0, weight="w"):
    a = a.assign(year=a.planning_year + shift)
    y = a.groupby(["district", "year"])[weight].sum().rename("N").reset_index().rename(columns={"district": "k"})
    p = base[base.year.isin(YEARS)].merge(y, on=["k", "year"], how="left")
    p["N"] = p.N.fillna(0)
    return p


def ppml(f, p, cl="k"):
    m = smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p[cl])[0]}, maxiter=200)
    return m


M1 = "N ~ net_idx + road + paved + dep + lnpop + capital + C(pt)"
M2 = "N ~ net_idx + lnpop + C(k) + C(year)"
res = []


def run(tag, a, f=M1, shift=0, weight="w", terms=("net_idx", "road", "paved", "dep"), drop_cap=False):
    p = outcome(a, shift, weight)
    if drop_cap:
        p = p[p.capital == 0]
    if "C(k)" in f:  # districts never treated contribute nothing under district FE
        p = p[p.groupby("k").N.transform("sum") > 0]
    try:
        m = ppml(f, p)
    except Exception as ex:
        print(tag, "FAILED", ex); return None
    for t in terms:
        if t in m.params:
            res.append({"spec": tag, "term": t, "b": m.params[t], "se": m.bse[t], "p": m.pvalues[t], "n_cells": len(p), "sumN": p.N.sum(),
                        "pct_per_sd": 100 * (np.exp(m.params[t]) - 1)})
    return m, p


a0 = asg
m1, p1 = run("M1 main", a0)
m2, p2 = run("M2 main (district+year FE)", a0, M2, terms=("net_idx",))
run("R1 binary", a0.assign(one=1), weight="one")
run("R1 binary M2", a0.assign(one=1), M2, weight="one", terms=("net_idx",))
run("R2 palika-precise", a0[a0.best_precision <= 3])
run("R3 observed dates", a0[a0.method.isin(["O1", "O1a", "O2", "O3"])])
run("R4 L-1", a0, shift=-1)
run("R4 L+1", a0, shift=1)
run("R4 L-1 M2", a0, M2, shift=-1, terms=("net_idx",))
run("R4 L+1 M2", a0, M2, shift=1, terms=("net_idx",))
run("R5 no covid", a0[~a0.covid])
run("R5 no covid M2", a0[~a0.covid], M2, terms=("net_idx",))
run("R6 <=10 districts", a0[a0.n_districts <= 10])
for c in ["mob_tests_pc_z", "mob_mbps_z", "mob_tiles_km2_z", "fix_tests_pc_z"]:
    run(f"R7 {c}", a0, M1.replace("net_idx", c), terms=(c, "road"))
    run(f"R7 {c} M2", a0, M2.replace("net_idx", c), terms=(c,))
for g in ["MDB", "bilateral", "UN-other"]:
    run(f"R8 {g}", a0[a0.dclass == g])
run("R9 USD-weighted", a0.assign(uw=a0.usd.fillna(0) * a0.w / 1e6), weight="uw")
run("R10 no capital", a0, M1.replace(" + capital", ""), drop_cap=True)
R = pd.DataFrame(res)

# Holm on the three confirmatory tests
conf = R[((R.spec == "M1 main") & R.term.isin(["net_idx", "road"])) | ((R.spec.str.startswith("M2 main")) & (R.term == "net_idx"))].copy()
order = conf.p.sort_values().index
k = len(order)
adj, run_max = {}, 0
for i, ix in enumerate(order):
    run_max = max(run_max, min(1, (k - i) * conf.loc[ix, "p"])); adj[ix] = run_max
R["p_holm"] = pd.Series(adj)

# randomisation inference for M1 net_idx: permute district connectivity trajectories across districts within province
rng = np.random.default_rng(42)
b_obs = m1.params["net_idx"]
traj = p1.pivot(index="k", columns="year", values="net_idx")
prov = p1.drop_duplicates("k").set_index("k").province
bs = []
for _ in range(500):
    perm = {}
    for pv, ks in prov.groupby(prov).groups.items():
        ks = list(ks); perm.update(dict(zip(ks, rng.permutation(ks))))
    q = p1.copy(); q["net_idx"] = [traj.loc[perm[k_], y] for k_, y in zip(q.k, q.year)]
    try:
        bs.append(smf.glm(M1, data=q, family=sm.families.Poisson()).fit(maxiter=100).params["net_idx"])
    except Exception:
        pass
ri_p = np.mean(np.abs(bs) >= abs(b_obs))

R.to_csv(ROOT / "outputs/tables/np2_panel_results.csv", index=False)
p1.to_csv(ROOT / "data/processed/nepal/np2_panel.csv", index=False)
with open(ROOT / "outputs/results_np2_panel.txt", "w", encoding="utf-8") as fh:
    def w(s=""):
        print(s); fh.write(s + "\n")
    w(f"located projects (planning 2019-2024): {prj.planning_year.between(2019, 2024).sum()} | cells: {len(p1)} | mean N per cell: {p1.N.mean():.2f} | zero share: {(p1.N == 0).mean():.2f}")
    w(f"MDE (2.8 x SE), % per SD: M1 net {100 * (np.exp(2.8 * m1.bse['net_idx']) - 1):.0f}% | M1 road {100 * (np.exp(2.8 * m1.bse['road']) - 1):.0f}% | "
      f"M2 net {100 * (np.exp(2.8 * m2.bse['net_idx']) - 1):.0f}%")
    w(f"RI p (M1 net_idx, within-province permutation of trajectories, 500 draws): {ri_p:.3f}")
    w(R.assign(b=R.b.round(3), se=R.se.round(3), p=R.p.round(3), pct_per_sd=R.pct_per_sd.round(1), p_holm=R.p_holm.round(3)).to_string(index=False))
