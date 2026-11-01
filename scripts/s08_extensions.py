"""Extended analyses requested by the PI.

A. Reverse causality: do projects attract population? (census 2010->2015 growth; WorldPop 2015->2020)
B. Roads vs network: separate / joint / interaction models, component correlation matrix, 2x2 groups
C. Road upgrades with a measurement-consistent panel: Budget Book 5 editions 2013-2025 (municipality x edition),
   counts and planned amounts, DiD + event study; East (Baucau, Lautem) descriptive
D. Sensitivity by category, ICT component, project size and duration (PPML) + stacked tests (unit FE)
E. Infrastructure crowd-in: do other projects follow access-infrastructure projects? (panel PPML, leads/lags)
F. Municipal-capital ("hub") posto test
Outputs: outputs/tables/ext_*.csv, outputs/results_extensions.txt
"""
import re, sys, warnings
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from gazetteer import geoparse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, INT, RAW = ROOT / "data" / "processed", ROOT / "data" / "interim", ROOT / "data" / "raw"
TAB = ROOT / "outputs" / "tables"
LOG = open(ROOT / "outputs" / "results_extensions.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")


def ppml(formula, data, cluster=None):
    m = smf.glm(formula, data, family=sm.families.Poisson())
    if cluster is not None:
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster])[0]})
    return m.fit(cov_type="HC1")


def row(m, v, **kw):
    return {"term": v, "coef": m.params[v], "se": m.bse[v], "p": m.pvalues[v], **kw}


z = lambda s: (s - s.mean()) / s.std()
U = gpd.read_file(PROC / "units.gpkg").drop(columns="geometry")
U["dili"] = (U.unit == "TL06U").astype(int)
for c, s in [("phys_z", U.access_phys), ("net_z", U.access_net), ("acc_z", U.access_pc1), ("dep_z", U.deprivation),
             ("poor_z", -U.rwi), ("dens_z", np.log(U.density))]:
    U[c] = z(s)
U["log_cpop10"] = np.log(U.cpop2010)
U["log_pop"] = np.log(U.pop2022)
D = U[U.dili == 0].copy()

# =====================================================================================
say("=" * 100 + "\nA. REVERSE CAUSALITY: DO DEVELOPMENT PROJECTS ATTRACT POPULATION?")
D["proj0610_pc"] = D.n_dev_0610 / D.cpop2010 * 1e4
U["proj0610_pc"] = U.n_dev_0610 / U.cpop2010 * 1e4
D["g1015_pct"] = D.g_10_15 * 100
U["g1015_pct"] = U.g_10_15 * 100
A = []
for name, f, d in [
    ("A1 growth 2010-15 ~ projects 2006-10 per 10k (excl Dili)", "g1015_pct ~ np.log1p(proj0610_pc) + log_cpop10 + acc_z + dep_z", D),
    ("A2 + municipality FE", "g1015_pct ~ np.log1p(proj0610_pc) + log_cpop10 + acc_z + dep_z + C(mun_pcode)", D),
    ("A3 incl Dili + dummy", "g1015_pct ~ np.log1p(proj0610_pc) + log_cpop10 + acc_z + dep_z + dili", U),
]:
    m = smf.ols(f, d).fit(cov_type="HC1")
    v = "np.log1p(proj0610_pc)"
    A.append(row(m, v, model=name, n=int(m.nobs), r2=m.rsquared))
    say(f"  {name}: b={m.params[v]:+.3f} (se {m.bse[v]:.3f}, p={m.pvalues[v]:.3f}), R2={m.rsquared:.2f}  "
        f"[access b={m.params['acc_z']:+.3f} p={m.pvalues['acc_z']:.3f}]")
r, p = stats.spearmanr(D.proj0610_pc, D.g_10_15)
say(f"  simple Spearman (projects 2006-10 per capita vs growth 2010-15): rho={r:+.2f} p={p:.3f}")
# reverse: does growth attract later projects?
m = smf.glm("n_dev_1625 ~ g1015_pct + np.log(cpop2015) + acc_z + dep_z", D, family=sm.families.Poisson()).fit(cov_type="HC1")
A.append(row(m, "g1015_pct", model="A4 projects 2016-25 ~ growth 2010-15 (PPML)", n=int(m.nobs)))
say(f"  A4 projects 2016-25 ~ growth 2010-15 (PPML): b={m.params['g1015_pct']:+.3f} per pp (p={m.pvalues['g1015_pct']:.3f})")
# WorldPop 2015->2020 growth (modelled) on projects 2006-15
D["gwp_1520"] = ((D.pop2020 / D.pop2015) ** 0.2 - 1) * 100
D["proj0615_pc"] = (D.n_dev_e1) / D.pop2015 * 1e4
m = smf.ols("gwp_1520 ~ np.log1p(proj0615_pc) + np.log(pop2015) + acc_z + dep_z", D).fit(cov_type="HC1")
A.append(row(m, "np.log1p(proj0615_pc)", model="A5 WorldPop growth 2015-20 ~ projects 2000-15", n=int(m.nobs)))
say(f"  A5 WorldPop growth 2015-20 ~ log projects 2000-15 pc: b={m.params['np.log1p(proj0615_pc)']:+.3f} (p={m.pvalues['np.log1p(proj0615_pc)']:.3f})")
# population elasticity with pre-period population (avoids pop responding to projects)
m = ppml("n_dev_0610 ~ log_cpop10 + acc_z + dep_z", D)
say(f"  A6 projects 2006-10 ~ log census pop 2010: elasticity={m.params['log_cpop10']:.2f} (se {m.bse['log_cpop10']:.2f}); "
    f"access b={m.params['acc_z']:+.3f} p={m.pvalues['acc_z']:.3f}; deprivation b={m.params['dep_z']:+.3f} p={m.pvalues['dep_z']:.3f}")
A.append(row(m, "log_cpop10", model="A6 elasticity projects 2006-10 to pop 2010", n=int(m.nobs)))
pd.DataFrame(A).to_csv(TAB / "ext_A_reverse_causality.csv", index=False)

# =====================================================================================
say("\n" + "=" * 100 + "\nB. ROADS vs NETWORK (excl. Dili; PPML; controls log pop + census deprivation)")
B = []
specs = {"B1 physical only": "phys_z", "B2 network only": "net_z", "B3 both": "phys_z + net_z",
         "B4 interaction": "phys_z * net_z", "B5 composite PC1": "acc_z"}
for y, ylab in [("np_dev", "precise"), ("n_dev", "all"), ("n_dev_strict", "strict (no Dili-only)"), ("n_dev_1625", "2016-25 only")]:
    say(f"  outcome: {ylab}")
    for name, rhs in specs.items():
        m = ppml(f"{y} ~ {rhs} + log_pop + dep_z", D)
        ll, k = m.llf, len(m.params)
        rec = {"outcome": ylab, "model": name, "aic": m.aic, "llf": ll, "k": k, "pseudo_r2": 1 - m.deviance / m.null_deviance}
        for v in ["phys_z", "net_z", "phys_z:net_z", "acc_z"]:
            if v in m.params:
                rec[f"b_{v}"] = m.params[v]; rec[f"p_{v}"] = m.pvalues[v]; rec[f"se_{v}"] = m.bse[v]
        B.append(rec)
        say("    " + f"{name:<20} " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in ["phys_z", "net_z", "phys_z:net_z", "acc_z"] if v in m.params)
            + f"   devR2={rec['pseudo_r2']:.3f} AIC={m.aic:.1f}")
Bdf = pd.DataFrame(B); Bdf.to_csv(TAB / "ext_B_roads_vs_network.csv", index=False)
# 2x2 groups
D["grp2x2"] = np.select([(D.phys_z >= D.phys_z.median()) & (D.net_z >= D.net_z.median()), (D.phys_z >= D.phys_z.median()),
                          (D.net_z >= D.net_z.median())], ["both high", "road high only", "network high only"], "both low")
G = D.groupby("grp2x2").agg(units=("unit", "size"), pop=("pop2022", "sum"), precise=("np_dev", "sum"), all_=("n_dev", "sum"),
                            dep=("deprivation", "mean"), tt_h=("tt_wet_min", lambda s: s.mean() / 60))
G["precise_per10k"] = G.precise / G["pop"] * 1e4; G["all_per10k"] = G.all_ / G["pop"] * 1e4
G.to_csv(TAB / "ext_B_2x2.csv"); say("  2x2 groups (median splits):\n" + G.round(2).to_string())
m = ppml("np_dev ~ C(grp2x2, Treatment('both low')) + log_pop + dep_z", D)
say("  PPML 2x2 dummies vs 'both low':\n" + "\n".join(f"    {k.split('[T.')[1][:-1]:<20} b={m.params[k]:+.3f} p={m.pvalues[k]:.3f} IRR={np.exp(m.params[k]):.2f}"
                                                      for k in m.params.index if "grp2x2" in k))
# all components simultaneously
compcols = {"log_tt": np.log(D.tt_wet_min), "paved": D.paved_share, "roadden": np.log(D.road_density), "steep": D.steep_share,
            "wetpen": D.wet_penalty_min / D.tt_wet_min, "net_tiles": np.log1p(D.net_tiles_per10k), "net_tests": np.log1p(D.net_tests_per1k)}
for k, v in compcols.items():
    D[f"c_{k}"] = z(v)
m = ppml("np_dev ~ " + " + ".join(f"c_{k}" for k in compcols) + " + log_pop + dep_z", D)
Xc = D[[f"c_{k}" for k in compcols]]
vif = pd.Series(np.diag(np.linalg.inv(np.corrcoef(Xc.T))), index=Xc.columns)
say("  all components jointly (precise outcome):\n" + "\n".join(f"    {k:<12} b={m.params[k]:+.3f} p={m.pvalues[k]:.3f}  VIF={vif[k]:.1f}" for k in Xc.columns))
pd.DataFrame({"coef": m.params[Xc.columns], "se": m.bse[Xc.columns], "p": m.pvalues[Xc.columns], "vif": vif}).to_csv(TAB / "ext_B_components_joint.csv")
# correlation matrix of components + need + outcomes
CM = pd.DataFrame({
    "우기 이동시간(log)": np.log(D.tt_wet_min), "포장률": D.paved_share, "도로밀도(log)": np.log(D.road_density), "급경사 비율": D.steep_share,
    "우기 지연 비율": D.wet_penalty_min / D.tt_wet_min, "Ookla 타일/인구": np.log1p(D.net_tiles_per10k), "Ookla 측정/인구": np.log1p(D.net_tests_per1k),
    "물리 하위지수": D.access_phys, "네트워크 하위지수": D.access_net, "종합지수 PC1": D.access_pc1,
    "인구(log)": D.log_pop, "인구밀도(log)": np.log(D.density), "부 지수 RWI": D.rwi, "센서스 박탈지수": D.deprivation,
    "개선 식수 %": D.water_improved, "개선 위생 %": D.sanit_improved, "인구증가 10-15": D.g_10_15,
    "정밀 사업 수": D.np_dev, "정밀 사업/1만": D.np_dev / D.pop2022 * 1e4, "전체 사업/1만": D.n_dev / D.pop2022 * 1e4})
CM.corr(method="spearman").to_csv(TAB / "ext_B_corr_spearman.csv", encoding="utf-8-sig")
CM.corr().to_csv(TAB / "ext_B_corr_pearson.csv", encoding="utf-8-sig")

# =====================================================================================
say("\n" + "=" * 100 + "\nC. ROAD UPGRADES — measurement-consistent Budget Book panel (municipality x edition, 2013-2025)")
bb = pd.read_csv(INT / "bb5_rows.csv")
bb = bb[(bb.donors.fillna("") != "") & ~bb.text.str.contains(r"\d+(\.\d+)?\s?%", regex=True)]
bb.loc[bb.amount_k > 100000, "amount_k"] = np.nan
MUN = {"TL02": "Aileu", "TL01": "Ainaro", "TL03": "Baucau", "TL04": "Bobonaro", "TL05": "Covalima", "TL06": "Dili", "TL07": "Ermera",
       "TL08": "Liquiçá", "TL09": "Lautém", "TL10": "Manufahi", "TL11": "Manatuto", "TL12": "Oecussi", "TL13": "Viqueque"}
recs = []
for r in bb.itertuples():
    ps, ms = geoparse(r.text)
    muns = {p[:4] for p in ps} | ms
    for mcode in muns:
        recs.append({"edition": r.book_year, "mun": MUN[mcode], "n": 1 / 1, "amt": (r.amount_k or 0) / len(muns) if pd.notna(r.amount_k) else 0,
                     "infra": bool(re.search(r"\broads?\b|bridge|highway|airport|port\b|electric|power", r.text, re.I))})
M = pd.DataFrame(recs)
full = pd.MultiIndex.from_product([sorted(bb.book_year.unique()), sorted(MUN.values())], names=["edition", "mun"]).to_frame(index=False)
Mp = M[~M.infra].groupby(["edition", "mun"]).agg(n=("n", "sum"), amt=("amt", "sum")).reset_index()
Mp = full.merge(Mp, how="left").fillna(0)
Mp["share"] = Mp.n / Mp.groupby("edition").n.transform("sum")
TREAT = {"Liquiçá": 2016, "Ermera": 2016, "Manatuto": 2018, "Baucau": 2020, "Aileu": 2020, "Ainaro": 2020, "Lautém": 2021}
Mp["treat_year"] = Mp.mun.map(TREAT)
Mp["post"] = (Mp.edition >= Mp.treat_year.fillna(9999)).astype(int)
Mp = Mp[Mp.mun != "Dili"]
Mp.to_csv(TAB / "ext_C_bb5_panel.csv", index=False)
say(f"  panel: {Mp.mun.nunique()} municipalities x {Mp.edition.nunique()} editions; non-infra project mentions = {Mp.n.sum():.0f}")
Crows = []
m = ppml("n ~ post + C(mun) + C(edition)", Mp, cluster="mun"); Crows.append(row(m, "post", model="C1 counts PPML TWFE"))
say(f"  C1 counts (PPML, mun+edition FE, cluster mun): post b={m.params['post']:+.3f} (se {m.bse['post']:.3f}, p={m.pvalues['post']:.3f}) IRR={np.exp(m.params['post']):.2f}")
m = smf.ols("share ~ post + C(mun) + C(edition)", Mp).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Mp.mun)[0]})
Crows.append(row(m, "post", model="C2 share of mentions OLS TWFE"))
say(f"  C2 share of all municipal mentions: post b={m.params['post']*100:+.2f} pp (p={m.pvalues['post']:.3f})")
m = smf.ols("np.log1p(amt) ~ post + C(mun) + C(edition)", Mp).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Mp.mun)[0]})
Crows.append(row(m, "post", model="C3 log planned amount OLS TWFE"))
say(f"  C3 log(1+planned amount, USD k): post b={m.params['post']:+.3f} (p={m.pvalues['post']:.3f})")
# event study (relative edition year, binned -4..+4)
Mp["rel"] = (Mp.edition - Mp.treat_year).clip(-4, 4)
ev = []
for k in range(-4, 5):
    if k == -1:
        continue
    Mp[f"r{k + 10}"] = ((Mp.rel == k) & Mp.treat_year.notna()).astype(int)
f = "n ~ " + " + ".join(f"r{k + 10}" for k in range(-4, 5) if k != -1) + " + C(mun) + C(edition)"
m = ppml(f, Mp, cluster="mun")
for k in range(-4, 5):
    ev.append({"rel": k, "coef": 0 if k == -1 else m.params[f"r{k + 10}"], "se": 0 if k == -1 else m.bse[f"r{k + 10}"]})
pd.DataFrame(ev).to_csv(TAB / "ext_C_event_bb5.csv", index=False)
say("  event study coefs: " + ", ".join(f"{e['rel']:+d}:{e['coef']:+.2f}" for e in ev))
# East descriptive
Mp["region"] = np.where(Mp.mun.isin(["Baucau", "Lautém"]), "East corridor (Baucau, Lautém)",
                        np.where(Mp.treat_year.notna(), "Other upgraded", "Not upgraded"))
Mp["period"] = np.where(Mp.edition <= 2019, "2013-2019", "2020-2025")
E = Mp.pivot_table(index="region", columns="period", values=["n", "amt", "share"], aggfunc="mean")
E.to_csv(TAB / "ext_C_east_descriptive.csv"); say("  mean per municipality-edition:\n" + E.round(3).to_string())
pd.DataFrame(Crows).to_csv(TAB / "ext_C_did.csv", index=False)

# posto-level budget (our DB) before/after for the east corridor
X = pd.read_csv(PROC / "unit_project_links.csv")
pan = pd.read_csv(PROC / "unit_year_panel.csv")
tre = pan.drop_duplicates("unit").set_index("unit").treat_year
X["treat_year"] = X.unit.map(tre)
east_units = set(U.loc[U.mun_pcode.isin(["TL03", "TL09"]), "unit"])
X["grp"] = np.where(X.unit.isin(east_units), "East (Baucau+Lautém postos)", np.where(X.treat_year.notna(), "Other upgraded postos", "Not upgraded postos"))
X = X[(X.unit != "TL06U") & (X.cat.isin(["SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"]))]
X["period"] = pd.cut(X.start_year, [2005, 2010, 2015, 2020, 2026], labels=["2006-10", "2011-15", "2016-20", "2021-26"])
T = X.groupby(["grp", "period"], observed=True).agg(projects=("cnt", "sum"), usd_m=("usd_alloc", lambda s: s.sum() / 1e6)).unstack()
T.to_csv(TAB / "ext_C_db_periods.csv"); say("  our DB, non-infra dev projects by period:\n" + T.round(1).to_string())

# =====================================================================================
say("\n" + "=" * 100 + "\nD. SENSITIVITY BY CATEGORY / ICT / SIZE / DURATION (PPML, excl. Dili; phys & net jointly; controls log pop, deprivation)")
Dd = []
GROUPS = [("n_SOC_education", "교육", "category"), ("n_SOC_health", "보건", "category"), ("n_SOC_water", "식수·위생", "category"),
          ("n_SOC_other", "기타 사회", "category"), ("n_GOV", "거버넌스", "category"), ("n_PROD", "생산부문", "category"),
          ("n_ENV_MULTI", "환경·복합", "category"), ("n_ACCESS_transport", "교통 인프라", "category"), ("n_ACCESS_energy", "에너지 인프라", "category"),
          ("n_ACCESS_comms", "통신 인프라", "category"), ("n_human", "인도적 지원", "category"),
          ("n_dev_ict", "ICT 요소 포함", "ict"), ("n_dev_nonict", "ICT 미포함", "ict"),
          ("n_dev_small", "소규모 (<$0.12m)", "size"), ("n_dev_medium", "중규모", "size"), ("n_dev_large", "대규모 (>$0.82m)", "size"),
          ("n_dur_short", "단기 (≤2년)", "duration"), ("n_dur_medium", "중기 (3-5년)", "duration"), ("n_dur_long", "장기 (≥6년)", "duration")]
for y, lab, fam in GROUPS:
    if D[y].sum() < 8:
        continue
    m = ppml(f"{y} ~ phys_z + net_z + log_pop + dep_z", D)
    Dd.append({"group": lab, "family": fam, "total": D[y].sum(),
               "b_phys": m.params["phys_z"], "se_phys": m.bse["phys_z"], "p_phys": m.pvalues["phys_z"],
               "b_net": m.params["net_z"], "se_net": m.bse["net_z"], "p_net": m.pvalues["net_z"],
               "b_dep": m.params["dep_z"], "p_dep": m.pvalues["dep_z"]})
    say(f"  {lab:<18} N={D[y].sum():6.0f}  road/phys b={m.params['phys_z']:+.3f}(p={m.pvalues['phys_z']:.3f})  "
        f"network b={m.params['net_z']:+.3f}(p={m.pvalues['net_z']:.3f})  deprivation b={m.params['dep_z']:+.3f}(p={m.pvalues['dep_z']:.3f})")
pd.DataFrame(Dd).to_csv(TAB / "ext_D_sensitivity.csv", index=False)


def stacked(cols, labels, ref):
    L = D.melt(id_vars=["unit", "phys_z", "net_z"], value_vars=cols, var_name="type", value_name="y")
    L["type"] = L.type.map(dict(zip(cols, labels)))
    for t in labels:
        if t != ref:
            L[f"net_x_{t}"] = L.net_z * (L.type == t); L[f"phys_x_{t}"] = L.phys_z * (L.type == t)
    rhs = " + ".join(c for c in L.columns if c.startswith(("net_x_", "phys_x_")))
    return ppml(f"y ~ C(unit) + C(type) + {rhs}", L, cluster="unit")


m = stacked(["n_dev_ict", "n_dev_nonict"], ["ict", "nonict"], "nonict")
say(f"  STACKED (unit FE) ICT vs non-ICT: extra network sensitivity of ICT projects = {m.params['net_x_ict']:+.3f} (p={m.pvalues['net_x_ict']:.3f}); "
    f"extra road sensitivity = {m.params['phys_x_ict']:+.3f} (p={m.pvalues['phys_x_ict']:.3f})")
st = [row(m, "net_x_ict", test="ICT vs non-ICT"), row(m, "phys_x_ict", test="ICT vs non-ICT")]
m = stacked(["n_dev_large", "n_dev_small"], ["large", "small"], "small")
say(f"  STACKED large vs small: network {m.params['net_x_large']:+.3f} (p={m.pvalues['net_x_large']:.3f}); road {m.params['phys_x_large']:+.3f} (p={m.pvalues['phys_x_large']:.3f})")
st += [row(m, "net_x_large", test="large vs small"), row(m, "phys_x_large", test="large vs small")]
m = stacked(["n_dur_long", "n_dur_short"], ["long", "short"], "short")
say(f"  STACKED long vs short: network {m.params['net_x_long']:+.3f} (p={m.pvalues['net_x_long']:.3f}); road {m.params['phys_x_long']:+.3f} (p={m.pvalues['phys_x_long']:.3f})")
st += [row(m, "net_x_long", test="long vs short"), row(m, "phys_x_long", test="long vs short")]
pd.DataFrame(st).to_csv(TAB / "ext_D_stacked_tests.csv", index=False)

# =====================================================================================
say("\n" + "=" * 100 + "\nE. INFRASTRUCTURE CROWD-IN: do other projects follow access-infrastructure projects? (unit-year PPML, excl. Dili)")
P = pan[pan.unit != "TL06U"].sort_values(["unit", "year"]).copy()
P["cum_access_l1"] = P.groupby("unit").n_access_new.cumsum().groupby(P.unit).shift(1).fillna(0)
P["any_access_l1_3"] = P.groupby("unit").n_access_new.transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0).gt(0).astype(int)
P["any_access_f1_3"] = P.groupby("unit").n_access_new.transform(lambda s: s[::-1].shift(1).rolling(3, min_periods=1).sum()[::-1]).fillna(0).gt(0).astype(int)
Erows = []
for nm, f, d in [("E1 cumulative infra projects (t-1), 2006-25", "n_dev_new ~ np.log1p(cum_access_l1) + C(unit) + C(year)", P),
                 ("E2 any infra start in t-1..t-3, 2006-25", "n_dev_new ~ any_access_l1_3 + C(unit) + C(year)", P),
                 ("E3 lags + leads (placebo), 2006-25", "n_dev_new ~ any_access_l1_3 + any_access_f1_3 + C(unit) + C(year)", P),
                 ("E4 lags + leads, AMP era 2006-15 only", "n_dev_new ~ any_access_l1_3 + any_access_f1_3 + C(unit) + C(year)", P[P.year <= 2015])]:
    m = ppml(f, d, cluster="unit")
    for v in [v for v in m.params.index if "access" in v]:
        Erows.append(row(m, v, model=nm))
    say(f"  {nm}: " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in m.params.index if "access" in v))
m = ppml("n_dev ~ np.log1p(n_access) + log_pop + dep_z + acc_z", D)
say(f"  E5 cross-section: dev projects ~ log(1+infra projects): b={m.params['np.log1p(n_access)']:+.3f} (p={m.pvalues['np.log1p(n_access)']:.3f})")
Erows.append(row(m, "np.log1p(n_access)", model="E5 cross-section"))
pd.DataFrame(Erows).to_csv(TAB / "ext_E_crowd_in.csv", index=False)

# =====================================================================================
say("\n" + "=" * 100 + "\nF. HUB TEST: is the within-municipality access effect a 'municipal capital' effect?")
F = []
for nm, f in [("F1 access PC1, muni FE", "np.log1p(np_dev) ~ acc_z + log_pop + dep_z + C(mun_pcode)"),
              ("F2 + capital dummy", "np.log1p(np_dev) ~ acc_z + mun_capital + log_pop + dep_z + C(mun_pcode)"),
              ("F3 network + road, muni FE", "np.log1p(np_dev) ~ phys_z + net_z + log_pop + dep_z + C(mun_pcode)"),
              ("F4 + capital dummy", "np.log1p(np_dev) ~ phys_z + net_z + mun_capital + log_pop + dep_z + C(mun_pcode)")]:
    m = smf.ols(f, D).fit(cov_type="HC1")
    vs = [v for v in ["acc_z", "phys_z", "net_z", "mun_capital", "dep_z"] if v in m.params]
    say(f"  {nm}: " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in vs))
    F += [row(m, v, model=nm) for v in vs]
pd.DataFrame(F).to_csv(TAB / "ext_F_hub.csv", index=False)

# deprivation vs RWI as need measure in main model
say("\n" + "=" * 100 + "\nG. MAIN MODEL WITH CENSUS DEPRIVATION (replacing RWI) and strict outcome")
for y in ["np_dev", "n_dev", "n_dev_strict"]:
    for d, lab in [(U, "all 62"), (D, "excl Dili")]:
        m = ppml(f"{y} ~ acc_z + log_pop + dep_z", d)
        say(f"  {y:<13} {lab:<9}: access={m.params['acc_z']:+.3f}(p={m.pvalues['acc_z']:.3f})  deprivation={m.params['dep_z']:+.3f}(p={m.pvalues['dep_z']:.3f})  "
            f"pop elasticity={m.params['log_pop']:.2f}")
LOG.close()
