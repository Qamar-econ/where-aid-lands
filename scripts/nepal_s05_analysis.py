"""Nepal: district dataset + protocol v1 primary model, robustness, extensions (roads vs network, reverse causality,
sector/ICT family with multiple-testing correction, crowd-in panel, era split).
Outputs: data/processed/nepal/district_panel.csv, outputs/tables/nepal_*.csv, outputs/results_nepal.txt
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT, TAB = ROOT / "data" / "processed" / "nepal", ROOT / "outputs" / "tables"
LOG = open(ROOT / "outputs" / "results_nepal.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}

D = gpd.read_file(OUT / "districts.gpkg").drop(columns="geometry").merge(pd.read_csv(OUT / "district_access.csv"), on="k")
PR, AS = pd.read_csv(OUT / "oda_projects.csv"), pd.read_csv(OUT / "oda_assignments.csv")
PR = PR[PR.start_year.between(2000, 2026)]
dv = PR[PR.cat.isin(DEV) & (PR.usd > 0)]; q1, q2 = dv.usd.quantile([1 / 3, 2 / 3])
PR["size"] = np.select([PR.usd.isna() | (PR.usd <= 0), PR.usd <= q1, PR.usd <= q2], ["unknown", "small", "medium"], "large")
PR["dur"] = np.select([PR.duration.isna(), PR.duration <= 2, PR.duration <= 5], ["unknown", "short", "medium"], "long")
X = AS[AS.level != "national"].merge(PR[["uid", "cat", "start_year", "usd", "donor_group", "ict", "size", "dur", "era", "eq"]], on="uid")
X["ab"] = X.tier.isin(["A", "B"])
X["grp"] = np.select([X.cat.isin(DEV), X.cat.str.startswith("ACCESS"), X.cat == "HUMAN"], ["dev", "access", "human"], "other")


def cnt(mask, name, ab=True):
    s = X[mask & (X.ab if ab else True)].groupby("unit").uid.nunique()
    D[name] = D.k.map(s).fillna(0)


g = X.grp == "dev"
cnt(g, "y_AB"); cnt(g & (X.tier == "A"), "y_A"); cnt(g, "y_all", ab=False)
cnt(g & (X.era == "e1"), "y_e1"); cnt(g & (X.era == "e2"), "y_e2"); cnt(g & ~X["eq"].astype(bool), "y_noeq")
cnt(g & X.start_year.between(2005, 2010), "y_0510")
cnt(X.grp == "access", "y_access"); cnt(X.grp == "human", "y_human")
for c in ["SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI", "ACCESS_transport", "ACCESS_energy"]:
    cnt(X.cat == c, f"y_{c}")
cnt(g & X.ict, "y_ict"); cnt(g & ~X.ict, "y_nonict")
for s in ["small", "medium", "large"]:
    cnt(g & (X["size"] == s), f"y_{s}")
for s in ["short", "medium", "long"]:
    cnt(g & (X["dur"] == s), f"y_dur_{s}")
from oda_common import BILAT
cnt(g & X.donor_group.isin(BILAT), "y_bilat"); cnt(g & ~X.donor_group.isin(BILAT), "y_multi")

# indices
comp = pd.DataFrame({"neg_log_tt": -np.log(D.tt_wet_min), "paved": D.paved_km / D.road_km, "roadden": np.log(D.road_km / D.area_km2),
                     "neg_steep": -D.steep_share, "neg_wetpen": -(D.wet_penalty_min / D.tt_wet_min)})
D["phys_idx"] = ((comp - comp.mean()) / comp.std()).mean(axis=1)
D["ook_idx"] = (z(np.log1p((D.ook2019_tiles + D.ook2025_tiles) / D.pop2011 * 1e5)) + z(np.log1p((D.ook2019_tests + D.ook2025_tests) / D.pop2011 * 1e5))) / 2
for c, s in [("phys", D.phys_idx), ("net", D.net_census), ("ook", D.ook_idx), ("dep", D.deprivation), ("rwi_poor", -D.rwi), ("ntl", np.log1p(D.ntl2010))]:
    D[c] = z(s)
D["lpop"] = np.log(D.pop2011)
D.to_csv(OUT / "district_panel.csv", index=False, encoding="utf-8-sig")
S = D[D.capital == 0].copy()
say(f"districts {len(D)}, analysis sample (excl. Kathmandu valley) {len(S)}; located dev projects (A/B) total {D.y_AB.sum():.0f}, all tiers {D.y_all.sum():.0f}")
cap = D[D.capital == 1]
say(f"Kathmandu valley: {cap.pop2011.sum() / D.pop2011.sum():.1%} of population, {cap.y_all.sum() / D.y_all.sum():.1%} of located dev projects; "
    f"central-institution projects = {(PR.target_type == 'central_institution').mean():.1%} of projects, "
    f"{PR.loc[PR.target_type == 'central_institution', 'usd'].sum() / PR.usd.sum():.1%} of USD "
    f"(excl. IFC GTFP outlier: {PR.loc[(PR.target_type == 'central_institution') & (PR.usd < 5e9), 'usd'].sum() / PR.loc[PR.usd < 5e9, 'usd'].sum():.1%})")


def ppml(f, d, clus=False):
    m = smf.glm(f, d, family=sm.families.Poisson())
    return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.province)[0]}) if clus else m.fit(cov_type="HC1")


BASE = "phys + net + dep + lpop"
SPECS = [("P  PRIMARY (census network, province FE)", f"y_AB ~ {BASE} + C(province)", S, False),
         ("R1 tier A only", f"y_A ~ {BASE} + C(province)", S, False),
         ("R2 all tiers, no FE", f"y_all ~ {BASE}", S, False),
         ("R3 + night lights 2010", f"y_AB ~ {BASE} + ntl + C(province)", S, False),
         ("R5 province-clustered SE (7)", f"y_AB ~ {BASE} + C(province)", S, True),
         ("R6 RWI instead of deprivation", "y_AB ~ phys + net + rwi_poor + lpop + C(province)", S, False),
         ("R7a era 1 (<=2014, AIMS)", f"y_e1 ~ {BASE} + C(province)", S, False),
         ("R7b era 2 (>=2015)", f"y_e2 ~ {BASE} + C(province)", S, False),
         ("R8 incl. Kathmandu valley + dummy", f"y_AB ~ {BASE} + capital + C(province)", D, False),
         ("R9 excl. earthquake-related projects", f"y_noeq ~ {BASE} + C(province)", S, False),
         ("R10 no FE", f"y_AB ~ {BASE}", S, False),
         ("R11 Ookla instead of census network", "y_AB ~ phys + ook + dep + lpop + C(province)", S, False),
         ("R12 census + Ookla jointly", "y_AB ~ phys + net + ook + dep + lpop + C(province)", S, False)]
rows = []
say("=" * 100 + "\nPRIMARY + ROBUSTNESS (PPML, 1 SD)")
for nm, f, d, cl in SPECS:
    m = ppml(f, d, cl); rec = {"spec": nm, "n": int(m.nobs)}
    for v in ["phys", "net", "ook", "dep", "rwi_poor", "ntl", "capital", "lpop"]:
        if v in m.params:
            rec[f"b_{v}"], rec[f"se_{v}"], rec[f"p_{v}"] = m.params[v], m.bse[v], m.pvalues[v]
    rows.append(rec)
    say(f"  {nm:<42} N={int(m.nobs)}  " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in ["phys", "net", "ook", "dep", "rwi_poor", "ntl", "capital"] if v in m.params))
pd.DataFrame(rows).to_csv(TAB / "nepal_robustness.csv", index=False)
lo = [ppml(f"y_AB ~ {BASE} + C(province)", S[S.province != p]).params[["phys", "net", "dep"]].rename(p) for p in S.province.unique()]
LO = pd.DataFrame(lo); say(f"  leave-one-province-out: net [{LO.net.min():+.3f},{LO.net.max():+.3f}]  phys [{LO.phys.min():+.3f},{LO.phys.max():+.3f}]  dep [{LO.dep.min():+.3f},{LO.dep.max():+.3f}]")
rng = np.random.default_rng(1); b0 = ppml(f"y_AB ~ {BASE} + C(province)", S).params["net"]; perm = []
for _ in range(500):
    Sp = S.copy(); Sp["net"] = Sp.groupby("province").net.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    perm.append(ppml(f"y_AB ~ {BASE} + C(province)", Sp).params["net"])
say(f"  randomization inference (network permuted within province): p = {(np.sum(np.abs(perm) >= abs(b0)) + 1) / 501:.3f}")

say("\n" + "=" * 100 + "\nROADS vs NETWORK (model comparison, primary outcome)")
cmp = []
for nm, rhs in [("road only", "phys"), ("network only", "net"), ("both", "phys + net"), ("interaction", "phys * net"), ("Ookla only", "ook")]:
    m = ppml(f"y_AB ~ {rhs} + dep + lpop + C(province)", S)
    cmp.append({"model": nm, "aic": m.aic, **{f"b_{v}": m.params[v] for v in m.params.index if v in ("phys", "net", "phys:net", "ook")},
                **{f"p_{v}": m.pvalues[v] for v in m.params.index if v in ("phys", "net", "phys:net", "ook")}})
    say(f"  {nm:<13} AIC={m.aic:7.1f}  " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in ["phys", "net", "phys:net", "ook"] if v in m.params))
pd.DataFrame(cmp).to_csv(TAB / "nepal_roads_vs_network.csv", index=False)
CM = S[["tt_wet_min", "phys_idx", "net_census", "mobile_pct", "internet_pct", "ook_idx", "deprivation", "rwi", "ntl2010", "pop2011", "y_AB"]].copy()
CM["tt_wet_min"] = np.log(CM.tt_wet_min); CM["y_AB_pc"] = S.y_AB / S.pop2011 * 1e5
CM.corr("spearman").to_csv(TAB / "nepal_corr_spearman.csv")
say("  Spearman: net_census~phys_idx {:.2f}, net_census~ookla {:.2f}, net~deprivation {:.2f}, phys~deprivation {:.2f}".format(
    *[CM.corr("spearman").loc[a, b] for a, b in [("net_census", "phys_idx"), ("net_census", "ook_idx"), ("net_census", "deprivation"), ("phys_idx", "deprivation")]]))

say("\n" + "=" * 100 + "\nREVERSE CAUSALITY (projects 2005-10 -> population growth 2011-2023; excl. valley)")
S["g_pct"] = S.g_11_23 * 100; S["p0510_pc"] = S.y_0510 / S.pop2011 * 1e5
m = smf.ols("g_pct ~ np.log1p(p0510_pc) + lpop + phys + net + dep + C(province)", S).fit(cov_type="HC1")
say(f"  growth ~ log(1+projects 2005-10 per 100k): b={m.params['np.log1p(p0510_pc)']:+.3f} (p={m.pvalues['np.log1p(p0510_pc)']:.3f})")
m = ppml("y_e2 ~ g_pct + lpop + phys + net + dep + C(province)", S)
say(f"  projects 2015+ ~ growth 2011-23: b={m.params['g_pct']:+.3f} (p={m.pvalues['g_pct']:.3f})")

say("\n" + "=" * 100 + "\nSECTOR / ICT / SIZE / DURATION / DONOR FAMILY (PPML no FE, all tiers; Holm & BH)")
fam = []
for y, lab in [("y_SOC_education", "Education"), ("y_SOC_health", "Health"), ("y_SOC_water", "Water & sanitation"), ("y_SOC_other", "Other social"),
               ("y_GOV", "Governance"), ("y_PROD", "Productive"), ("y_ENV_MULTI", "Env./multisector"), ("y_ACCESS_transport", "Transport infra"),
               ("y_ACCESS_energy", "Energy infra"), ("y_human", "Humanitarian"), ("y_ict", "ICT component"), ("y_nonict", "Non-ICT"),
               ("y_small", "Small"), ("y_medium", "Medium"), ("y_large", "Large"), ("y_dur_short", "Short"), ("y_dur_medium", "Medium duration"),
               ("y_dur_long", "Long"), ("y_bilat", "Bilateral"), ("y_multi", "Multilateral/UN")]:
    if S[y].sum() < 10:
        continue
    m = ppml(f"{y} ~ phys + net + dep + lpop", S)
    for v in ["phys", "net", "dep"]:
        fam.append({"group": lab, "term": v, "coef": m.params[v], "se": m.bse[v], "p": m.pvalues[v], "n_proj": S[y].sum()})
F = pd.DataFrame(fam); F["p_holm"] = multipletests(F.p, method="holm")[1]; F["q_bh"] = multipletests(F.p, method="fdr_bh")[1]
F.to_csv(TAB / "nepal_sector_family.csv", index=False)
for r in F[F.p < 0.1].itertuples():
    say(f"  {r.group:<18} {r.term:<5} b={r.coef:+.3f} p={r.p:.3f} Holm={r.p_holm:.3f} BH q={r.q_bh:.3f}")
cap_ict = cap.y_ict.sum() / D.y_ict.sum(); cap_non = cap.y_nonict.sum() / D.y_nonict.sum()
ict_central = (PR[PR.cat.isin(DEV)].groupby("ict").target_type.apply(lambda s: (s == "central_institution").mean()))
say(f"  ICT: valley share of located ICT dev {cap_ict:.1%} vs non-ICT {cap_non:.1%}; central-institution share ICT {ict_central.get(True, 0):.1%} vs non-ICT {ict_central.get(False, 0):.1%}")

say("\n" + "=" * 100 + "\nCROWD-IN PANEL (district-year 2005-2025, PPML district & year FE, cluster district; excl. valley)")
yrs = range(2005, 2026)
Y = X[X.ab | (X.method == "manual")].copy()
pan = pd.MultiIndex.from_product([S.k, yrs], names=["unit", "year"]).to_frame(index=False)
dn = Y[Y.grp == "dev"].groupby(["unit", "start_year"]).uid.nunique().rename("dev").reset_index().rename(columns={"start_year": "year"})
an = Y[Y.grp == "access"].groupby(["unit", "start_year"]).uid.nunique().rename("acc").reset_index().rename(columns={"start_year": "year"})
pan = pan.merge(dn, how="left").merge(an, how="left").fillna(0).sort_values(["unit", "year"])
pan["acc_l13"] = pan.groupby("unit").acc.transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0).gt(0).astype(int)
pan["acc_f13"] = pan.groupby("unit").acc.transform(lambda s: s[::-1].shift(1).rolling(3, min_periods=1).sum()[::-1]).fillna(0).gt(0).astype(int)
for nm, d in [("2005-2025", pan), ("AIMS era 2005-2014", pan[pan.year <= 2014])]:
    m = smf.glm("dev ~ acc_l13 + acc_f13 + C(unit) + C(year)", d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]})
    say(f"  {nm}: lag(1-3y) b={m.params['acc_l13']:+.3f} (p={m.pvalues['acc_l13']:.3f})   lead placebo b={m.params['acc_f13']:+.3f} (p={m.pvalues['acc_f13']:.3f})")
LOG.close()
