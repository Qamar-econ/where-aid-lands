"""Pre-registered H-C3 (사전등록_센서스분석.md section 4; execution details in spec 9-31): two-period district panel.

E[N_dt] = exp(mu_d + lambda_t + beta P_dt)
  early  activities starting 2009-2012, P = households with a mobile phone, census 2008
  late   activities starting 2017-2020, P = census 2019
mu_d district fixed effects, lambda_t period effect; beta per 10 percentage points of phone ownership.
Outcome: non-ICT activities located at district level, GODAD geocodes as published in both periods (spec 9-31).
Inference: district-clustered SE (pre-registered) and randomisation inference permuting the change in ownership within
province (2,000 draws, primary). Holm over the pre-registered family {H-C1, H-C2, H-C3}.
Robustness: EU/US donors only; early period 2005-2008; late period with the spec 9-26 reclassification.
Output: outputs/tables/kh_hc3.csv, kh_hc3.txt
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent
ns = {"__file__": str(HERE / "kh_s06_ladder.py")}
exec((HERE / "kh_s06_ladder.py").read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, ROOT = ns["DF"], ns["L"], ns["ROOT"]
src = (HERE / "kh_s35_infra_stacked.py").read_text(encoding="utf-8")
exec(src[src.index("def _demean"):src.index("# ---------------- stacked panel")], ns)   # PPML with two high-dimensional FE (checked in kh_s35)
ppml = ns["ppml"]
from kh_common import reclassify_geocodes
PROC, T = ROOT / "data/processed/cambodia", ROOT / "outputs/tables"
import os; DRAWS = int(os.environ.get("HC3_DRAWS", 2000))
RNG = np.random.default_rng(20261104)
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

c = pd.read_csv(PROC / "kh_census.csv")
P08, P19 = (c[c.year == y].set_index("unit").phone_share * 100 for y in (2008, 2019))
L = L.copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce")
L = L[L.unit.isin(DF.unit) & (L.family != "ict")]
units = DF.unit.values; prov = DF.set_index("unit").prov


def counts(frame, lo, hi):
    return frame[frame.yr.between(lo, hi)].groupby("unit").w.sum().reindex(units, fill_value=0)


def panel(early, late):
    P = pd.DataFrame({"unit": np.r_[units, units], "t": np.r_[np.zeros(len(units)), np.ones(len(units))],
                      "y": np.r_[early.values, late.values], "P": np.r_[P08.reindex(units).values, P19.reindex(units).values]})
    P["prov"] = P.unit.map(prov)
    return P.dropna(subset=["P"]).reset_index(drop=True)


def fit(P, draws=DRAWS):
    X = (P.P / 10).values[:, None]
    b, V = ppml(P.y.values.astype(float), X, P.unit.values, P.t.values, P.unit.values)
    b, s = b[0], np.sqrt(V[0, 0])
    p_ri = np.nan
    if draws:
        d = P.pivot(index="unit", columns="t", values="P"); dP = (d[1] - d[0]); pv = d.index.map(prov)
        bs = []
        for _ in range(draws):
            perm = dP.copy()
            for g in np.unique(pv):
                ix = np.where(pv == g)[0]; perm.iloc[ix] = RNG.permutation(dP.values[ix])
            late = d[0] + perm
            Pp = P.copy(); Pp.loc[Pp.t == 1, "P"] = Pp.loc[Pp.t == 1, "unit"].map(late).values
            bs.append(ppml(Pp.y.values.astype(float), (Pp.P / 10).values[:, None], Pp.unit.values, Pp.t.values, se=False)[0][0])
        p_ri = (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs))
    n_inf = P.groupby("unit").y.transform("sum").gt(0).sum() // 2
    return {"pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * s) - 1), "ci_hi": 100 * (np.exp(b + 1.96 * s) - 1),
            "p_cluster": 2 * norm.sf(abs(b / s)), "p_ri": p_ri, "districts_informative": int(n_inf), "projects": P.y.sum()}


rows = []


def run(label, early, late, draws=DRAWS):
    r = {"model": label, **fit(panel(early, late), draws)}; rows.append(r)
    say(f"  {label:<52s} {r['pct']:+5.1f}% per 10 pp (95% CI {r['ci_lo']:+.0f}% to {r['ci_hi']:+.0f}%) | p RI {r['p_ri']:.3f} | cluster {r['p_cluster']:.3f} | "
        f"districts {r['districts_informative']} | projects {r['projects']:.0f}")


dP = (P19 - P08).reindex(units)
say(f"phone ownership: 2008 mean {P08.reindex(units).mean():.1f}%, 2019 mean {P19.reindex(units).mean():.1f}% | change mean {dP.mean():.1f} pp, SD {dP.std():.1f} pp, "
    f"range {dP.min():.1f} to {dP.max():.1f} | corr(change, log travel time) = {np.corrcoef(dP, np.log(DF.set_index('unit').tt_town_dry.reindex(units)))[0, 1]:.2f}")
say("== H-C3 MAIN (pre-registered): activities 2009-2012 vs 2017-2020, district and period fixed effects")
run("H-C3 main", counts(L, 2009, 2012), counts(L, 2017, 2020))
say("== ROBUSTNESS (pre-registered and spec 9-31)")
eu = L[L.dgroup == "EU/US"]
run("EU/US donors only", counts(eu, 2009, 2012), counts(eu, 2017, 2020), draws=999)
run("early period 2005-2008", counts(L, 2005, 2008), counts(L, 2017, 2020), draws=999)
late_rc = reclassify_geocodes(L[L.yr.between(2017, 2020)], DF, PROC)
run("late period with geocode reclassification", counts(L, 2009, 2012), late_rc.groupby("unit").w.sum().reindex(units, fill_value=0), draws=999)
R = pd.DataFrame(rows); R.to_csv(T / "kh_hc3.csv", index=False)

M = pd.read_csv(T / "kh_main_simple.csv")
fam = {"H-C1 deprivation": M[(M.model == "poverty alone") & (M["var"] == "deprivation")].p_ri.iloc[0],
       "H-C2 phone ownership": M[(M.model == "digital access alone") & (M["var"] == "phone")].p_ri.iloc[0],
       "H-C3 panel": R.loc[0, "p_ri"]}
run_, holm = 0, {}
for i, (k, v) in enumerate(sorted(fam.items(), key=lambda kv: kv[1])):
    run_ = max(run_, min(1, (3 - i) * v)); holm[k] = run_
say("== Holm over the pre-registered family: " + ", ".join(f"{k} p {fam[k]:.3f} -> {holm[k]:.3f}" for k in fam))
(T / "kh_hc3.txt").write_text("\n".join(log), encoding="utf-8")
