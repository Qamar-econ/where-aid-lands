"""Briggs (2017) replication on DHS regions + network extension + Gelbach decomposition.

Step 1  replicate: projects ~ poverty (share in lowest wealth quintile) + log pop + exposure + country FE  (expect b_poverty < 0)
Step 2  extend   : + household mobile-phone ownership (observed, measured before the projects) [+ electricity, night lights]
Step 3  Gelbach (2016) decomposition of the change in the poverty / night-light coefficient (OLS, log(1+projects))
Outputs: outputs/tables/briggs_*.csv, outputs/results_briggs.txt
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
LOG = open(ROOT / "outputs" / "results_briggs.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()

R = gpd.read_file(ROOT / "data/processed/briggs/regions.gpkg").drop(columns="geometry")
R = R[R.pop2010 > 0].copy()
# countries with no post-survey projects are perfectly predicted by their FE (PPML separation) -> dropped
R = R[R.groupby("DHS_CountryCode").n_proj.transform("sum") > 0].copy()
R["poor"], R["mob"], R["elec"] = z(R.poor_q1), z(R.mobile), z(R.electricity)
R["ntl"] = z(np.log1p(R.ntl2010 / R.area_km2))
R["lpop"] = np.log(R.pop2010)
R["lexp"] = np.log(R.years_obs.clip(lower=1))
R["capital_reg"] = (R.groupby("DHS_CountryCode").pop2010.transform(lambda s: s / s.sum()) * 0 +
                    (R.ntl2010 / R.area_km2 == R.groupby("DHS_CountryCode").apply(lambda d: d.ntl2010 / d.area_km2, include_groups=False).groupby(level=0).transform("max").values)).astype(int)
R["ly"] = np.log1p(R.n_proj)
say(f"Sample: {len(R)} DHS regions in {R.DHS_CountryCode.nunique()} Sub-Saharan countries; surveys 2006-2011; WB projects approved after survey through 2014 = {R.n_proj.sum():.0f} region-project links")


def ppml(f, d, clus=True):
    m = smf.glm(f, d, family=sm.families.Poisson())
    return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.DHS_CountryCode)[0]}) if clus else m.fit(cov_type="HC1")


rows = []
SPECS = [("B1 Briggs replication: poverty", "n_proj ~ poor + lpop + C(DHS_CountryCode)"),
         ("B2 + mobile phone ownership", "n_proj ~ poor + mob + lpop + C(DHS_CountryCode)"),
         ("B3 + electricity", "n_proj ~ poor + mob + elec + lpop + C(DHS_CountryCode)"),
         ("B4 night lights only", "n_proj ~ ntl + lpop + C(DHS_CountryCode)"),
         ("B5 night lights + mobile", "n_proj ~ ntl + mob + lpop + C(DHS_CountryCode)"),
         ("B6 night lights + mobile + electricity", "n_proj ~ ntl + mob + elec + lpop + C(DHS_CountryCode)"),
         ("B7 all", "n_proj ~ poor + ntl + mob + elec + lpop + C(DHS_CountryCode)"),
         ("B8 B2 excl. brightest (capital) region", "n_proj ~ poor + mob + lpop + C(DHS_CountryCode)")]
say("=" * 100 + "\nPPML, country FE, SE clustered by country (1 SD)")
for nm, f in SPECS:
    d = R[R.capital_reg == 0] if "excl." in nm else R
    m = ppml(f, d)
    rec = {"spec": nm, "n": int(m.nobs)}
    for v in ["poor", "ntl", "mob", "elec"]:
        if v in m.params:
            rec[f"b_{v}"], rec[f"se_{v}"], rec[f"p_{v}"] = m.params[v], m.bse[v], m.pvalues[v]
    rows.append(rec)
    say(f"  {nm:<42} N={int(m.nobs)}  " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in ["poor", "ntl", "mob", "elec"] if v in m.params))
pd.DataFrame(rows).to_csv(TAB / "briggs_models.csv", index=False)

m = smf.logit("any_aid ~ poor + mob + lpop + C(DHS_CountryCode)", R).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(R.DHS_CountryCode)[0]}) if R.any_aid.mean() < 0.98 else None
if m is not None:
    m0 = smf.logit("any_aid ~ poor + lpop + C(DHS_CountryCode)", R).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(R.DHS_CountryCode)[0]})
    say(f"  Logit any aid: poverty alone {m0.params['poor']:+.3f} (p={m0.pvalues['poor']:.3f}) -> with mobile {m.params['poor']:+.3f} (p={m.pvalues['poor']:.3f}); mobile {m.params['mob']:+.3f} (p={m.pvalues['mob']:.3f})")


def gelbach(y, focal, added, base, data):
    """Gelbach (2016): decompose b_focal(base) - b_focal(full) into contributions of each added covariate."""
    fe = " + C(DHS_CountryCode)"
    b_base = smf.ols(f"{y} ~ {focal} + {base}{fe}", data).fit().params[focal]
    full = smf.ols(f"{y} ~ {focal} + {' + '.join(added)} + {base}{fe}", data).fit()
    out = {}
    for a in added:
        delta = smf.ols(f"{a} ~ {focal} + {base}{fe}", data).fit().params[focal]
        out[a] = full.params[a] * delta
    # bootstrap by country for SE of each contribution and of total
    rng = np.random.default_rng(0); cs = data.DHS_CountryCode.unique(); boots = []
    for _ in range(400):
        samp = pd.concat([data[data.DHS_CountryCode == c].assign(DHS_CountryCode=f"{c}_{i}") for i, c in enumerate(rng.choice(cs, len(cs)))])
        try:
            fb = smf.ols(f"{y} ~ {focal} + {' + '.join(added)} + {base}{fe}", samp).fit()
            boots.append({a: fb.params[a] * smf.ols(f"{a} ~ {focal} + {base}{fe}", samp).fit().params[focal] for a in added})
        except Exception:
            continue
    Bt = pd.DataFrame(boots)
    return b_base, full.params[focal], out, Bt.std(), Bt.sum(axis=1).std()


say("\n" + "=" * 100 + "\nGELBACH DECOMPOSITION (OLS log(1+projects), country FE; bootstrap SE by country, 400 draws)")
dec = []
for focal, added in [("poor", ["mob"]), ("poor", ["mob", "elec"]), ("ntl", ["mob"]), ("ntl", ["mob", "elec"])]:
    b0, b1, contrib, se, se_tot = gelbach("ly", focal, added, "lpop", R)
    tot = b0 - b1
    say(f"  {focal}: base {b0:+.3f} -> full {b1:+.3f}; explained {tot:+.3f} ({tot / b0:.0%} of base, SE {se_tot:.3f})  " +
        "  ".join(f"{a}: {c:+.3f} (SE {se[a]:.3f}, {c / b0:.0%})" for a, c in contrib.items()))
    dec.append({"focal": focal, "added": "+".join(added), "b_base": b0, "b_full": b1, "explained": tot, "share": tot / b0, "se_total": se_tot,
                **{f"c_{a}": c for a, c in contrib.items()}, **{f"se_{a}": se[a] for a in contrib}})
pd.DataFrame(dec).to_csv(TAB / "briggs_gelbach.csv", index=False)
say("\n  Spearman correlations: " + ", ".join(f"{a}~{b}={R[[a, b]].corr('spearman').iloc[0, 1]:+.2f}" for a, b in
                                                [("poor_q1", "mobile"), ("poor_q1", "electricity"), ("mobile", "electricity"), ("ntl", "mobile"), ("ntl", "poor_q1")]))
LOG.close()
