"""Spec 9-35 (exploratory): persistence. Holding travel time fixed, do districts that had projects in 2005-2012 get more in 2017-2020?

prior   = 1 if the district had a district-located non-ICT activity starting 2005-2012 (GODAD geocodes as published)
prior_n = log(1 + those activities)
Outcome, sample and model as kh_main.py (2017-2020, spec 9-26 reclassification, PPML, province FE, ln population),
Freedman-Lane randomisation inference within province (primary), province-clustered SE, Conley 100 km.
Model 5 uses only other donors' past projects: EU/US outcomes get non-EU/US prior presence and vice versa (two-group stack).
Output: outputs/tables/kh_persistence.csv, kh_persistence.txt
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
src = (HERE / "kh_main.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_main.py"), "__name__": "kh_main_ns"}
sys.argv = sys.argv[:1]
exec(src[:src.index('say("== MAIN: one construct')], ns)
DF, MAIN, A0, counts, fit, ri_p, conley_p, T = (ns[k] for k in ("DF", "MAIN", "A0", "counts", "fit", "ri_p", "conley_p", "T"))
ns["RNG"] = np.random.default_rng(20261107)
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))

lad = {"__file__": str(HERE / "kh_s06_ladder.py")}
exec((HERE / "kh_s06_ladder.py").read_text(encoding="utf-8").split("LADDER = [")[0], lad)
L = lad["L"].copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce")
PR = L[L.yr.between(2005, 2012) & (L.family != "ict") & L.unit.isin(DF.unit)]
DF["prior"] = DF.unit.isin(PR.unit).astype(int)
DF["prior_n"] = np.log1p(DF.unit.map(PR.groupby("unit").w.sum()).fillna(0))
DF["capkhan"] = DF.unit.isin({116008004, 116008005, 116008007, 116008011})
DF["rural"] = ~DF.capkhan & ~DF.ADMIN_NAME.str.contains("Krong")


def estimate(label, xs, d=None, draws=DRAWS):
    d = (counts(MAIN) if d is None else d).reset_index(drop=True)
    f, r = fit(d, xs)
    for v in xs:
        b, s = r.params[v], r.bse[v]
        row = {"model": label, "var": v, "pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * s) - 1), "ci_hi": 100 * (np.exp(b + 1.96 * s) - 1),
               "p_ri": ri_p(d, v, [x for x in xs if x != v], f, b, draws) if draws else np.nan, "p_cluster": r.pvalues[v],
               "p_conley100": conley_p(r, d, v), "districts": len(d), "projects": d.N.sum()}
        rows.append(row)
        say(f"  {label:<44s} {v:<8s} {row['pct']:+6.0f}% (95% CI {row['ci_lo']:+.0f}% to {row['ci_hi']:+.0f}%) | p RI {row['p_ri']:.3f} | "
            f"cluster {row['p_cluster']:.3f} | Conley {row['p_conley100']:.3f} | districts {len(d)}, projects {d.N.sum():.1f}")


d0 = counts(MAIN)
say(f"districts {len(d0)} | with 2005-2012 projects {DF.prior.sum()} ({PR.project_id.nunique()} activities) | corr(reach, prior) {DF.reach.corr(DF.prior):+.2f}, "
    f"corr(reach, prior_n) {DF.reach.corr(DF.prior_n):+.2f}")
say(f"descriptive: projects per 100,000 people 2017-2020, prior 1: {1e5 * d0[d0.prior == 1].N.sum() / d0[d0.prior == 1].pop_2019.sum():.2f}, "
    f"prior 0: {1e5 * d0[d0.prior == 0].N.sum() / d0[d0.prior == 0].pop_2019.sum():.2f}")
say("== MAIN (spec 9-35)")
estimate("1 prior alone", ["prior"])
estimate("2 prior + travel time", ["prior", "reach"])
say("== CHECKS")
estimate("3 rural districts: prior + travel time", ["prior", "reach"], d=d0[d0.rural], draws=min(DRAWS, 999))
estimate("4 Phnom Penh khans excluded", ["prior", "reach"], d=d0[~d0.capkhan], draws=min(DRAWS, 999))
estimate("6 prior_n + travel time", ["prior_n", "reach"], draws=min(DRAWS, 999))

say("== POST HOC (added after seeing model 6): is the count of past projects only the hubs?")
estimate("7 post hoc: prior_n + travel time, rural only", ["prior_n", "reach"], d=d0[d0.rural], draws=min(DRAWS, 999))
estimate("8 post hoc: prior_n + travel time, no PP khans", ["prior_n", "reach"], d=d0[~d0.capkhan], draws=min(DRAWS, 999))
say(f"  mean prior_n: Phnom Penh khans {d0[d0.capkhan].prior_n.mean():.2f}, towns {d0[~d0.capkhan & ~d0.rural].prior_n.mean():.2f}, rural {d0[d0.rural].prior_n.mean():.2f}")
say("== 5 other donors' past projects only (two-group stack)")
grp = A0.drop_duplicates("project_id").set_index("project_id").dgroup.eq("EU/US")
pr_eu = set(PR[PR.dgroup == "EU/US"].unit); pr_ot = set(PR[PR.dgroup != "EU/US"].unit)
parts = []
for g, own in [("EU/US", True), ("other", False)]:
    sub = MAIN[MAIN.project_id.map(grp).eq(own)]
    dg = counts(sub).assign(g=g); dg["prior_other"] = dg.unit.isin(pr_ot if own else pr_eu).astype(int); parts.append(dg)
S = pd.concat(parts, ignore_index=True); S["pg"] = S.prov + "|" + S.g
f5 = "N ~ prior_other + reach:C(g) + lnpop:C(g) + C(pg)"
r5 = smf.glm(f5, data=S, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.prov)[0]}, maxiter=300)
b5, s5 = r5.params["prior_other"], r5.bse["prior_other"]
aux = smf.ols("prior_other ~ reach:C(g) + lnpop:C(g) + C(pg)", data=S).fit(); fv, e0 = aux.fittedvalues.values, aux.resid.values
idx = [np.where(S.pg.values == k)[0] for k in S.pg.unique()]; bs = []
rng = np.random.default_rng(20261108)
for _ in range(min(DRAWS, 999)):
    e = e0.copy()
    for ix in idx:
        e[ix] = rng.permutation(e0[ix])
    bs.append(smf.glm(f5, data=S.assign(prior_other=fv + e), family=sm.families.Poisson()).fit(maxiter=100).params["prior_other"])
p5 = (1 + np.sum(np.abs(bs) >= abs(b5))) / (1 + len(bs))
row = {"model": "5 other donors' past projects + travel time", "var": "prior_other", "pct": 100 * (np.exp(b5) - 1), "ci_lo": 100 * (np.exp(b5 - 1.96 * s5) - 1),
       "ci_hi": 100 * (np.exp(b5 + 1.96 * s5) - 1), "p_ri": p5, "p_cluster": 2 * norm.sf(abs(b5 / s5)), "p_conley100": np.nan, "districts": len(S) // 2, "projects": S.N.sum()}
rows.append(row)
say(f"  {row['model']:<44s} prior_o  {row['pct']:+6.0f}% (95% CI {row['ci_lo']:+.0f}% to {row['ci_hi']:+.0f}%) | p RI {p5:.3f} | cluster {row['p_cluster']:.3f} | "
    f"EU/US projects {parts[0].N.sum():.1f}, other {parts[1].N.sum():.1f}")
pd.DataFrame(rows).to_csv(T / "kh_persistence.csv", index=False)
(T / "kh_persistence.txt").write_text("\n".join(log), encoding="utf-8")
