"""Spec 9-33 (exploratory): distance to Phnom Penh, where donor offices sit, vs distance to the nearest town.

capital = log2 travel time to Phnom Penh (tt_ktm_dry), town = log2 travel time to the nearest town of 20,000+ (main 'reach').
Province fixed effects absorb 77% of the capital-distance variation, so the comparison uses the four CSES zones as fixed
effects. Same outcome, sample and ln population as kh_main.py. Freedman-Lane RI within zone (primary), province-clustered
SE, Conley 100 km. The two are correlated at 0.78, so they are compared by cross-validated deviance gain and a two-way
Shapley split, not by a joint model.
Output: outputs/tables/kh_capital.csv, kh_capital_compare.csv, kh_capital.txt
"""
import sys, os, itertools, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DR2, DR3 = min(DRAWS, 999), min(DRAWS, 499)
src = (HERE / "kh_main.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_main.py"), "__name__": "kh_main_ns"}
sys.argv = sys.argv[:1]
exec(src[:src.index('say("== MAIN: one construct')], ns)
DF, MAIN, A0, counts, conley_p, T = (ns[k] for k in ("DF", "MAIN", "A0", "counts", "conley_p", "T"))
RNG = np.random.default_rng(20261105)
log = []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))

ZONE = {**{p: "plains" for p in (3, 8, 14, 20, 21)}, **{p: "tonle_sap" for p in (1, 2, 4, 6, 15, 17, 24)},
        **{p: "coastal" for p in (7, 9, 23)}, **{p: "plateau" for p in (5, 10, 11, 13, 16, 19)}}
DF["zone"] = DF.PARENT.map(ZONE)
assert DF.zone.notna().all(), DF[DF.zone.isna()].PARENT.unique()
DF["capital"] = np.log2(DF.tt_ktm_dry); DF["town"] = DF.reach
DF["capkhan"] = DF.unit.isin({116008004, 116008005, 116008007, 116008011})
FE = {"zone": "C(zone)", "province": "C(prov)", "none": None}


def formula(xs, fe):
    return "N ~ " + " + ".join(list(xs) + ["lnpop"] + ([FE[fe]] if FE[fe] else []))


def ri_p(d, v, f, b, fe, draws):
    """Freedman-Lane: v regressed on ln pop + FE; residuals permuted within zone (within province for the province-FE check)."""
    aux = smf.ols(f"{v} ~ " + " + ".join(["lnpop"] + ([FE[fe]] if FE[fe] else [])), data=d).fit()
    fv, e0 = aux.fittedvalues.values, aux.resid.values
    g = d.prov.values if fe == "province" else d.zone.values
    idx = [np.where(g == k)[0] for k in np.unique(g)]; bs = []
    for _ in range(draws):
        e = e0.copy()
        for ix in idx:
            e[ix] = RNG.permutation(e0[ix])
        bs.append(smf.glm(f, data=d.assign(**{v: fv + e}), family=sm.families.Poisson()).fit(maxiter=100).params[v])
    return (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs))


rows = []


def estimate(label, a, v, fe="zone", draws=DRAWS, d=None):
    d = (counts(a) if d is None else d).reset_index(drop=True)
    f = formula([v], fe)
    r = smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)
    b, s = r.params[v], r.bse[v]
    row = {"model": label, "var": v, "fe": fe, "pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * s) - 1),
           "ci_hi": 100 * (np.exp(b + 1.96 * s) - 1), "p_ri": ri_p(d, v, f, b, fe, draws) if draws else np.nan,
           "p_cluster": r.pvalues[v], "p_conley100": conley_p(r, d, v), "projects": d.N.sum(), "districts": len(d)}
    rows.append(row)
    say(f"  {label:<40s} {v:<8s} FE {fe:<8s} {row['pct']:+5.0f}% per doubling (95% CI {row['ci_lo']:+.0f}% to {row['ci_hi']:+.0f}%) | "
        f"p RI {row['p_ri']:.3f} | cluster {row['p_cluster']:.3f} | Conley {row['p_conley100']:.3f} | projects {row['projects']:.1f}")
    return row


CVR = np.random.default_rng(12)


def cv_gain(d, xs, fe="zone", reps=20, k=5):
    """Out-of-sample Poisson deviance gain over FE + ln population; folds stratified by zone."""
    gains = []
    strat = d.zone.values
    for _ in range(reps):
        fold = np.full(len(d), -1)
        for z in np.unique(strat):
            ix = np.where(strat == z)[0]; fold[ix] = CVR.permutation(np.arange(len(ix)) % k)
        db = dm = 0.0
        for j in range(k):
            tr, te = d[fold != j], d[fold == j]
            if fe == "province":
                te = te[te.prov.isin(tr.prov)]
            for xs_, which in (([], "b"), (list(xs), "m")):
                m = smf.glm(formula(xs_, fe), data=tr, family=sm.families.Poisson()).fit(maxiter=200)
                mu = np.clip(m.predict(te), 1e-9, None); y = te.N.values
                dv = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
                db, dm = (db + dv, dm) if which == "b" else (db, dm + dv)
        gains.append(100 * (db - dm) / db)
    return float(np.mean(gains)), float(np.std(gains))


def shapley2(d, fe="zone"):
    dev = lambda xs: smf.glm(formula(xs, fe), data=d, family=sm.families.Poisson()).fit(maxiter=300).deviance
    v0, vt, vc, vb = dev([]), dev(["town"]), dev(["capital"]), dev(["town", "capital"])
    st = 0.5 * ((v0 - vt) + (vc - vb)); sc = 0.5 * ((v0 - vc) + (vt - vb))
    return 100 * st / (st + sc), 100 * sc / (st + sc)


comp = []


def compare(label, d, fe="zone"):
    cvt, cvc = cv_gain(d, ["town"], fe), cv_gain(d, ["capital"], fe); sh_t, sh_c = shapley2(d, fe)
    comp.append({"sample": label, "fe": fe, "cv_town": cvt[0], "cv_town_sd": cvt[1], "cv_capital": cvc[0], "cv_capital_sd": cvc[1],
                 "shapley_town": sh_t, "shapley_capital": sh_c, "districts": len(d), "projects": d.N.sum()})
    say(f"  COMPARE {label:<32s} FE {fe:<8s} CV gain: town {cvt[0]:+.1f}% (sd {cvt[1]:.1f}), capital {cvc[0]:+.1f}% (sd {cvc[1]:.1f}) | "
        f"Shapley: town {sh_t:.0f}%, capital {sh_c:.0f}%")


d0 = counts(MAIN)
say(f"districts {len(d0)} | projects {d0.N.sum():.1f} | corr(town, capital) = {d0.town.corr(d0.capital):.2f} | "
    f"median travel time to Phnom Penh {d0.tt_ktm_dry.median():.0f} min | zones: " + ", ".join(f"{z} {n}" for z, n in d0.zone.value_counts().items()))
say("== MAIN (spec 9-33): zone fixed effects")
estimate("town alone", MAIN, "town")
estimate("capital alone", MAIN, "capital")
compare("all districts", d0)
say("== ROBUSTNESS")
dn = d0[~d0.capkhan]
estimate("Phnom Penh khans excluded: town", MAIN, "town", draws=DR2, d=dn)
estimate("Phnom Penh khans excluded: capital", MAIN, "capital", draws=DR2, d=dn)
compare("Phnom Penh khans excluded", dn)
for fe in ("province", "none"):
    estimate(f"{fe} FE: town", MAIN, "town", fe=fe, draws=DR2)
    estimate(f"{fe} FE: capital", MAIN, "capital", fe=fe, draws=DR2)
    compare("all districts", d0, fe=fe)
say("== MECHANISM (exploratory): donor groups and sectors, zone FE")
fam = A0.assign(s3=pd.to_numeric(A0.sector_code.astype(str).str.extract(r"(\d{3})")[0], errors="coerce"))
fam["family"] = np.select([fam.s3.isin([210, 230]), fam.s3.between(110, 169), fam.s3.between(310, 339)], ["infra", "social", "production"], "other")
fam = fam.drop_duplicates("project_id").set_index("project_id")
for lab, keep in [("donors EU/US", fam.dgroup == "EU/US"), ("donors World Bank", fam.dgroup == "World Bank"), ("donors China", fam.dgroup == "China"),
                  ("sector social", fam.family == "social"), ("sector production", fam.family == "production"),
                  ("sector infrastructure", fam.family == "infra"), ("sector other", fam.family == "other")]:
    sub = MAIN[MAIN.project_id.isin(fam.index[keep])]
    if sub.w.sum() < 10:
        say(f"  {lab}: {sub.w.sum():.1f} projects, skipped"); continue
    for v in ("town", "capital"):
        estimate(f"{lab}: {v}", sub, v, draws=DR3)
    compare(lab, counts(sub))
pd.DataFrame(rows).to_csv(T / "kh_capital.csv", index=False)
pd.DataFrame(comp).to_csv(T / "kh_capital_compare.csv", index=False)
(T / "kh_capital.txt").write_text("\n".join(log), encoding="utf-8")
