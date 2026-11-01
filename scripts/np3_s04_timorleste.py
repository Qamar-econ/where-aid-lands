"""Timor-Leste comparison, 2010-2025, same parsimonious model as Nepal (np3_s03_models.py):
  E[N_{u,t}] = exp( b_A Access_u + b_C Conn_u + b_D Need_u + theta ln Pop_{u,t} + g_{m(u),t} )
  u = posto (Dili's urban postos merged), m = municipality, t = planning year (start - 1), 2009-2024.
  Access = PC1 of travel time to Dili (dry, wet), walking time to road, share of population within 2 km of a road, road density,
           paved share, wet-season penalty (physical only); Conn = posto network index from the original study (access_net);
  Need = census-2010 deprivation. Municipality-level projects are split equally across the municipality's postos.
Outputs: outputs/tables/tl3_table_main.csv, tl3_heterogeneity.csv, outputs/results_tl3.txt
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
z = lambda s: (s - s.mean()) / s.std()
P = pd.read_csv(ROOT / "data/processed/oda_projects.csv")
A = pd.read_csv(ROOT / "data/processed/oda_assignments.csv")
W = pd.read_csv(ROOT / "data/processed/windows_TL.csv")
PA = pd.read_csv(ROOT / "data/processed/posto_access.csv").rename(columns={"pcode": "unit"})
UN = pd.read_csv(ROOT / "data/processed/units.csv")
U = W[W.window == "W1"][["unit", "mun_pcode", "access_net", "deprivation"]].merge(UN[["unit", "pop2010", "pop2015", "pop2020", "area_km2"]], on="unit").merge(PA, on="unit")
comp = pd.DataFrame({"neg_ln_tt_dry": -np.log(U.tt_dry_min), "neg_ln_tt_wet": -np.log(U.tt_wet_min), "neg_ln_walk": -np.log(U.walk_min.clip(lower=0.5)),
                     "share_2km": U.share_pop_2km_road, "ln_road_density": np.log(U.road_density), "paved_share": U.paved_share,
                     "neg_wet_penalty": -U.wet_penalty_min / U.tt_dry_min})
Zc = (comp - comp.mean()) / comp.std()
w_, V = np.linalg.eigh(np.cov(Zc.T)); pc = V[:, -1] * np.sign(V[:, -1].sum())
U["access"] = z(Zc.values @ pc); U["conn"] = z(U.access_net); U["need"] = z(U.deprivation)
YEARS = range(2009, 2025)
B = pd.concat([U.assign(year=t, lnpop=np.interp(t, [2010, 2015, 2020], [0, 0, 0]) + np.log(U.pop2010) +
                        (np.log(U.pop2020) - np.log(U.pop2010)) * (t - 2010) / 10) for t in YEARS])
B["mt"] = B.mun_pcode.astype(str) + "_" + B.year.astype(str)

P = P[P.start_year.between(2010, 2025)].copy()
GRP = {"SOC_education": "Education", "SOC_health": "Health", "SOC_water": "Water & sanitation", "GOV": "Governance", "PROD": "Agriculture & production",
       "ENV_MULTI": "Environment & multisector", "SOC_other": "Other social", "HUMAN": "Humanitarian"}
P["category"] = P["cat"].map(GRP).fillna(pd.Series(np.where(P["cat"].fillna("").str.startswith("ACCESS"), "Infrastructure", "Other/unclassified"), index=P.index))
P["family"] = np.select([P.category.eq("Humanitarian"), P.category.eq("Infrastructure"), P.category.eq("Other/unclassified")],
                        ["humanitarian", "infrastructure", "other"], "development")
P["size"] = pd.cut(P.usd, [-1, 1e6, 1e7, np.inf], labels=["< $1M", "$1-10M", "> $10M"]).astype(str)
mun_units = U.groupby("mun_pcode").unit.apply(list).to_dict()
rows = []
for r in A[A.uid.isin(P.uid) & A.level.isin(["posto", "municipio"])].itertuples():
    us = [r.unit] if r.level == "posto" else mun_units.get(r.unit, [])
    for u in us:
        rows.append((r.uid, u, r.weight / len(us)))
AS = pd.DataFrame(rows, columns=["uid", "unit", "w"])
AS = AS[AS.unit.isin(U.unit)].groupby(["uid", "unit"]).w.sum().reset_index()
AS["w"] = AS.w / AS.groupby("uid").w.transform("sum")
AS = AS.merge(P[["uid", "start_year", "family", "category", "size", "source"]], on="uid")
AS["year"] = AS.start_year.astype(int) - 1


def panel(a):
    y = a.groupby(["unit", "year"]).w.sum().rename("N").reset_index()
    p = B.merge(y, on=["unit", "year"], how="left"); p["N"] = p.N.fillna(0)
    return p


def fit(f, p):
    return smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.unit)[0]}, maxiter=300)


def row(tag, m):
    o = {"spec": tag}
    for v in ["access", "conn", "need", "lnpop"]:
        if v in m.params and v in m.params.index:
            o[v + "_b"], o[v + "_se"], o[v + "_p"] = m.params[v], m.bse[v], m.pvalues[v]
            o[v + "_pct"] = 100 * (np.exp(m.params[v]) - 1) if v != "lnpop" else np.nan
    o["theta_p_eq1"] = 2 * norm.sf(abs(m.params["lnpop"] - 1) / m.bse["lnpop"])
    return o


JOINT = "N ~ access + conn + need + lnpop + C(mt)"
dev = AS[AS.family == "development"]
p0 = panel(dev)
T1 = [row("(0) population only", fit("N ~ lnpop + C(mt)", p0))] + [row(f"({i + 1}) {v} only", fit(f"N ~ {v} + lnpop + C(mt)", p0)) for i, v in enumerate(["access", "conn", "need"])]
T1 += [row("(4) joint", fit(JOINT, p0)), row("(5) joint, all sectors", fit(JOINT, panel(AS))),
       row("(6) joint, excluding Dili", fit(JOINT, panel(dev).query("unit != @B.sort_values('pop2010').unit.iloc[-1]")))]
T1 = pd.DataFrame(T1); T1.to_csv(ROOT / "outputs/tables/tl3_table_main.csv", index=False)
HET = []
for tag, a in [("All development", dev)] + [(f"Category: {c}", AS[AS.category == c]) for c in GRP.values() if c != "Humanitarian"] + \
        [(f"Size: {s}", dev[dev["size"] == s]) for s in ["< $1M", "$1-10M", "> $10M"]] + [("Placebo: Humanitarian", AS[AS.family == "humanitarian"]),
                                                                                        ("Placebo: Infrastructure", AS[AS.family == "infrastructure"])]:
    if a.uid.nunique() >= 30:
        try:
            pp = panel(a); o = {"group": tag, "n_projects": a.uid.nunique()}
            for v in ["access", "conn", "need"]:
                m = fit(f"N ~ {v} + lnpop + C(mt)", pp); o.update({v + "_pct": 100 * (np.exp(m.params[v]) - 1), v + "_p": m.pvalues[v]})
            m = fit("N ~ lnpop + C(mt)", pp); o.update({"lnpop_b": m.params["lnpop"], "theta_p_eq1": 2 * norm.sf(abs(m.params["lnpop"] - 1) / m.bse["lnpop"])})
            HET.append(o)
        except Exception as e:
            print(tag, "failed", e)
HET = pd.DataFrame(HET); HET.to_csv(ROOT / "outputs/tables/tl3_heterogeneity.csv", index=False)
pd.DataFrame({"unit": U.unit, "pop": U.pop2015, "N": U.unit.map(dev.groupby("unit").w.sum()).fillna(0)}).to_csv(
    ROOT / "outputs/tables/tl3_units_totals.csv", index=False)  # for the population-scaling figure
cs = U[["access", "conn", "need"]].assign(lnpop=np.log(U.pop2015))
R = np.corrcoef(cs.values.T)
out = [f"TL postos {len(U)} | development projects located 2010-2025: {dev.uid.nunique()} | all located: {AS.uid.nunique()} | projects in DB 2010-2025: {len(P)}",
       f"access PC1 var share {w_[-1] / w_.sum():.2f} loadings {dict(zip(comp.columns, pc.round(2)))}",
       "correlations:\n" + pd.DataFrame(R, index=cs.columns, columns=cs.columns).round(2).to_string(),
       f"VIF: {dict(zip(cs.columns, np.diag(np.linalg.inv(R)).round(2)))}",
       T1[["spec", "access_pct", "access_p", "conn_pct", "conn_p", "need_pct", "need_p", "lnpop_b", "theta_p_eq1"]].round(3).to_string(index=False),
       HET[["group", "n_projects", "access_pct", "access_p", "conn_pct", "conn_p", "need_pct", "need_p", "lnpop_b", "theta_p_eq1"]].round(3).to_string(index=False)]
(ROOT / "outputs/results_tl3.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
