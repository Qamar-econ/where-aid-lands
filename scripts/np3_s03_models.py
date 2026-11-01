"""Paper models, Nepal 2010-2025: parsimonious specification, one variable per construct.

  E[N_{d,t}] = exp( b_A Access_d + b_C Conn_{d,t} + b_D Need_d + theta ln Pop_{d,t} + g_{p(d),t} )

  N_{d,t}  new projects in district d with planning year t (= start year - 1), multi-district projects split equally
  Access   physical-access index (PC1: travel time to town / HQ / hospital, monsoon travel time, walking time, share within
           30 min of a road, road density, paved share, monsoon penalty, terrain) -- roads and paved roads enter only here
  Conn     household internet share (census 2011 and 2021, logit-linear interpolation; trend-extrapolated to 2010, 2022-2025)
  Need     census-2011 deprivation (no improved water, no toilet, illiteracy) -- contains no communication item
  theta    population elasticity (theta = 1: projects proportional to population)
  g        province x year fixed effects; PPML; SE clustered by district (75)
Each construct alone, then jointly (VIF reported). Capital is not a regressor; Kathmandu-valley exclusion is a robustness test.
Outputs: outputs/tables/np3_*.csv, outputs/results_np3_models.txt, data/processed/nepal/np3_panel.csv
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs/tables"
P = pd.read_csv(ROOT / "data/processed/nepal/np3_projects.csv")
AS = pd.read_csv(ROOT / "data/processed/nepal/np3_assign.csv")
AC = pd.read_csv(ROOT / "data/processed/nepal/np3_access.csv")
D = pd.read_csv(ROOT / "data/processed/nepal/districts.csv")
H = pd.read_csv(ROOT / "data/processed/nepal/census2021_district_hh.csv")
YEARS = range(2009, 2025)  # planning years (start 2010-2025)
z = lambda s: (s - s.mean()) / s.std()

# ---------------- district-year covariates ----------------
X = D[["k", "province", "pop2011", "pop2023", "internet_pct", "deprivation", "capital"]].merge(H[["k", "internet_pct21"]], on="k").merge(
    AC[["k", "access_pc1", "tt_ktm_dry", "tt_town_dry", "tt_hq_dry", "tt_hosp_dry", "share_walk30", "road_density", "paved_share"]], on="k")
logit = lambda p: np.log(p / (1 - p))
l11, l21 = logit(np.clip(X.internet_pct / 100, 0.002, 0.99)), logit(np.clip(X.internet_pct21 / 100, 0.002, 0.99))
rows = []
for t in YEARS:
    r = X[["k", "province", "capital"]].copy(); r["year"] = t
    r["conn_raw"] = 100 / (1 + np.exp(-(l11 + (l21 - l11) * (t - 2011) / 10)))
    r["lnpop"] = np.log(X.pop2011) + (np.log(X.pop2023) - np.log(X.pop2011)) * (t - 2011) / 12
    rows.append(r)
B = pd.concat(rows)
B = B.merge(X[["k", "access_pc1", "deprivation", "tt_ktm_dry", "tt_town_dry", "tt_hq_dry", "tt_hosp_dry", "share_walk30", "road_density", "paved_share"]], on="k")
B["access"] = z(B.access_pc1); B["conn"] = z(B.conn_raw); B["need"] = z(B.deprivation)
for c, s in [("tt_town_dry", -1), ("tt_hq_dry", -1), ("tt_hosp_dry", -1), ("tt_ktm_dry", -1), ("road_density", 1), ("paved_share", 1), ("share_walk30", 1)]:
    B["c_" + c] = z(s * np.log(B[c]) if c.startswith("tt") or c == "road_density" else s * B[c])
B["pt"] = B.province.astype(str) + "_" + B.year.astype(str)
B.to_csv(ROOT / "data/processed/nepal/np3_panel.csv", index=False)

AS = AS.merge(P[["uid", "planning_year", "planning_year_doc", "family", "category", "size", "dclass", "source", "usd", "quake", "covid", "n_districts"]], on="uid")


def panel(a, weight="w", year="planning_year"):
    y = a.groupby(["k", year])[weight].sum().rename("N").reset_index().rename(columns={year: "year"})
    p = B.merge(y, on=["k", "year"], how="left"); p["N"] = p.N.fillna(0)
    return p


def fit(f, p):
    return smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.k)[0]}, maxiter=300)


def row(tag, m, terms):
    out = {"spec": tag}
    for v in terms:
        if v in m.params:
            out[v + "_b"], out[v + "_se"], out[v + "_p"] = m.params[v], m.bse[v], m.pvalues[v]
            out[v + "_pct"] = 100 * (np.exp(m.params[v]) - 1) if v != "lnpop" else np.nan
    if "lnpop" in m.params:
        out["theta_p_eq1"] = 2 * norm.sf(abs(m.params["lnpop"] - 1) / m.bse["lnpop"])
    return out


TERMS = ["access", "conn", "need", "lnpop"]
JOINT = "N ~ access + conn + need + lnpop + C(pt)"


def obo(p, fe="C(pt)", cov=None):
    """Primary specification: each construct in its own model (+ ln pop + FE); theta from the population-only model."""
    f_ = (lambda f: fit(f, p)) if cov is None else (lambda f: smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type=cov, maxiter=300))
    o = {}
    for v in ["access", "conn", "need"]:
        m = f_(f"N ~ {v} + lnpop + {fe}")
        o[v + "_b"], o[v + "_se"], o[v + "_p"], o[v + "_pct"] = m.params[v], m.bse[v], m.pvalues[v], 100 * (np.exp(m.params[v]) - 1)
    m = f_(f"N ~ lnpop + {fe}")
    o["lnpop_b"], o["lnpop_se"] = m.params["lnpop"], m.bse["lnpop"]
    o["theta_p_eq1"] = 2 * norm.sf(abs(m.params["lnpop"] - 1) / m.bse["lnpop"])
    return o
dev = AS[AS.family == "development"]
txt = []
w = txt.append

# ---- Table 2: one construct at a time, then jointly (development projects)
p0 = panel(dev)
T2 = [row("(0) population only", fit("N ~ lnpop + C(pt)", p0), TERMS)] + [row(f"({i + 1}) {v} only", fit(f"N ~ {v} + lnpop + C(pt)", p0), TERMS) for i, v in enumerate(["access", "conn", "need"])]
T2.append(row("(4) joint", fit(JOINT, p0), TERMS))
pa = panel(AS)
T2.append(row("(5) joint, all sectors", fit(JOINT, pa), TERMS))
T2 = pd.DataFrame(T2); T2.to_csv(T / "np3_table2_main.csv", index=False)
cs = B[B.year == 2016][["access", "conn", "need", "lnpop"]]
R = np.corrcoef(cs.values.T); vif = dict(zip(cs.columns, np.diag(np.linalg.inv(R)).round(2)))
w(f"development projects located, planning 2009-2024: {dev.uid.nunique()} | district-years {len(p0)} | mean N {p0.N.mean():.2f}")
w("correlations (2016 cross-section):\n" + pd.DataFrame(R, index=cs.columns, columns=cs.columns).round(2).to_string())
w(f"VIF: {vif}")
w(T2[["spec"] + [c for c in T2.columns if c.endswith("_pct") or c.endswith("_p") or c == "lnpop_b"]].round(3).to_string(index=False))

# ---- access components one at a time (with conn, need, lnpop)
COMP = []
for c in [c for c in B.columns if c.startswith("c_")]:
    m = fit(f"N ~ {c} + conn + need + lnpop + C(pt)", p0)
    COMP.append({"component": c[2:], "pct_per_sd": 100 * (np.exp(m.params[c]) - 1), "p": m.pvalues[c]})
COMP = pd.DataFrame(COMP); COMP.to_csv(T / "np3_access_components.csv", index=False)
w("access components, one at a time:\n" + COMP.round(3).to_string(index=False))

# ---- heterogeneity (joint model)
HET = []
groups = [("All development", dev)] + [(f"Category: {c}", AS[AS.category == c]) for c in
          ["Education", "Health", "Water & sanitation", "Governance", "Agriculture & production", "Environment & multisector", "Other social"]] + \
         [(f"Size: {s}", dev[dev["size"] == s]) for s in ["< $1M", "$1-10M", "> $10M"]] + \
         [(f"Donor: {s}", dev[dev.dclass == s]) for s in ["Bilateral", "MDB", "UN", "NGO/foundation"]] + \
         [(f"Period: {a}-{b}", dev[dev.planning_year.between(a - 1, b - 1)]) for a, b in [(2010, 2014), (2015, 2019), (2020, 2025)]] + \
         [("Placebo: Humanitarian", AS[AS.family == "humanitarian"]), ("Placebo: Infrastructure", AS[AS.family == "infrastructure"])]
for tag, a in groups:
    if a.uid.nunique() < 30:
        continue
    p = panel(a)
    if tag.startswith("Period"):
        yrs = sorted(a.planning_year.unique()); p = p[p.year.isin(yrs)]
    try:
        HET.append({"group": tag, "n_projects": a.uid.nunique(), **obo(p)})
    except Exception as e:
        w(f"{tag} failed: {e}")
HET = pd.DataFrame(HET); HET.to_csv(T / "np3_heterogeneity.csv", index=False)
w("heterogeneity (one construct at a time, % per SD; theta from population-only model):\n" + HET[["group", "n_projects", "access_pct", "access_p", "conn_pct", "conn_p", "need_pct", "need_p", "lnpop_b", "theta_p_eq1"]].round(3).to_string(index=False))

# ---- year-by-year coefficients (province FE within year)
YY = []
for t in YEARS:
    p = p0[p0.year == t]
    try:
        YY.append({"year": t + 1, **obo(p, "C(province)", "HC1"), "n": p.N.sum()})
    except Exception:
        pass
pd.DataFrame(YY).to_csv(T / "np3_yearly.csv", index=False)

# ---- robustness (joint model, development)
RB = []
RB.append({"spec": "R0 main", **obo(p0)})
RB.append({"spec": "R1 IATI/WB only (single source 2010-2025)", **obo(panel(dev[~dev.source.str.startswith("AidData")]))})
RB.append({"spec": "R2 binary counting", **obo(panel(dev.assign(one=1.0), "one"))})
RB.append({"spec": "R3 USD-weighted ($m)", **obo(panel(dev.assign(um=dev.usd.fillna(0) * dev.w / 1e6), "um"))})
pv = panel(dev); pv = pv[pv.capital == 0]
RB.append({"spec": "R4 excluding Kathmandu valley", **obo(pv)})
RB.append({"spec": "R5 excluding earthquake & COVID projects", **obo(panel(dev[~dev.quake & ~dev.covid]))})
RB.append({"spec": "R6 single-district projects", **obo(panel(dev[dev.n_districts == 1]))})
RB.append({"spec": "R7 projects in <= 10 districts", **obo(panel(dev[dev.n_districts <= 10]))})
dd = dev.assign(py=dev.planning_year_doc.fillna(dev.planning_year).astype(int))
RB.append({"spec": "R8 documented planning year where available", **obo(panel(dd, year="py"))})
pf = p0[p0.groupby("k").N.transform("sum") > 0]
RB.append(row("R10 within: district + year FE (conn only)", fit("N ~ conn + lnpop + C(k) + C(year)", pf), ["conn", "lnpop"]))
RB = pd.DataFrame(RB); RB.to_csv(T / "np3_robustness.csv", index=False)
w("robustness (one construct at a time):\n" + RB[["spec"] + [c for c in RB.columns if c.endswith("_pct") or c.endswith("_p") or c == "lnpop_b" or c == "conn11_pct"]].round(3).to_string(index=False))

# ---- portfolio accounting by start year
acc = P.groupby(["start_year", "status"]).agg(n=("uid", "size"), usd=("usd", "sum")).reset_index()
acc.to_csv(T / "np3_portfolio_status.csv", index=False)
w("portfolio status (all years): " + str(P.status.value_counts(normalize=True).round(3).to_dict()) +
  " | by USD: " + str((P.groupby("status").usd.sum() / P.usd.sum()).round(3).to_dict()))
(ROOT / "outputs/results_np3_models.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
