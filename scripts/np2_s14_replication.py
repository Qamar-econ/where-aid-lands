"""Pre-registered replication of the size gradient (사전등록_규모기울기_복제.md) in two independent samples:
R-A Nepal 2009-2013 starts (75 districts; AidData-AIMS + IATI; census-2011 connectivity)
R-B Timor-Leste 2009-2015 starts (postos; existing DB; posto network index access_net)
Model: stacked size groups, E[N^s_{u,t}] = exp(sum_v (b_v + b_vS s) X_v + (theta + theta_S s) lnPop + kappa Cap + g_{upper,t,s}), PPML, cluster by unit.
Output: outputs/results_replication_gradient.txt
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
z = lambda s: (s - s.mean()) / s.std()
DEVX = {"HUMAN", "ACCESS_transport", "ACCESS_energy", "ACCESS_comms", "ACCESS_water_infra"}
out = []


def stacked(asg, units, years, groups, upper, tag):
    """asg: uid, unit, w, year, size_grp ; units: unit, upper, Xs, lnpop, cap"""
    st = []
    for s in sorted(groups):
        y = asg[asg.sgrp == s].groupby(["unit", "year"]).w.sum().rename("N").reset_index()
        grid = pd.MultiIndex.from_product([units.unit, years], names=["unit", "year"]).to_frame(index=False)
        p = grid.merge(units, on="unit").merge(y, on=["unit", "year"], how="left").fillna({"N": 0})
        p["grp"] = s; p["gfe"] = p[upper].astype(str) + "_" + p.year.astype(str) + "_" + str(s)
        st.append(p)
    st = pd.concat(st)
    X = ["net", "road", "paved", "dep"]
    f = "N ~ " + " + ".join(f"{v} + {v}:grp" for v in X) + " + lnpop + lnpop:grp + cap + C(gfe)"
    m = smf.glm(f, data=st, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(st.unit)[0]}, maxiter=300)
    n = asg.drop_duplicates("uid").sgrp.value_counts().sort_index().to_dict()
    res = {}
    for v, side in [("road", 1), ("net", -1)]:
        b, se = m.params[f"{v}:grp"], m.bse[f"{v}:grp"]
        res[v] = (b, se, norm.sf(side * b / se))  # one-sided p in the hypothesised direction
    holm = sorted([res["road"][2], res["net"][2]])
    out.append(f"{tag}: projects per size group {n} | H1 road:size {res['road'][0]:+.3f} (se {res['road'][1]:.3f}, one-sided p {res['road'][2]:.3f}) | "
               f"H2 net:size {res['net'][0]:+.3f} (se {res['net'][1]:.3f}, one-sided p {res['net'][2]:.3f}) | Holm min-p {min(1, 2 * holm[0]):.3f}")
    for s in sorted(groups):  # group-specific slopes for display
        q = st[st.grp == s]
        try:
            mm = smf.glm("N ~ net + road + paved + dep + lnpop + cap + C(gfe)", data=q, family=sm.families.Poisson()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(q.unit)[0]}, maxiter=300)
            out.append(f"    size group {s}: road {100 * (np.exp(mm.params['road']) - 1):+.0f}% (p {mm.pvalues['road']:.3f}) | "
                       f"net {100 * (np.exp(mm.params['net']) - 1):+.0f}% (p {mm.pvalues['net']:.3f})")
        except Exception as e:
            out.append(f"    size group {s}: failed {e}")


def size_grp(usd, three=True):
    return pd.cut(usd, [-1, 1e6, 1e7, np.inf], labels=[1, 2, 3]).astype(int) if three else pd.cut(usd, [-1, 1e6, np.inf], labels=[1, 2]).astype(int)


# ---------------- R-A Nepal 2009-2013 ----------------
P = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv")
A = pd.read_csv(ROOT / "data/processed/nepal/oda_assignments.csv")
DP = pd.read_csv(ROOT / "data/processed/nepal/district_panel.csv")
P = P[P.start_year.between(2009, 2013) & P.usd.notna() & (P.usd > 0) & ~P["cat"].isin(DEVX) & (P.ict != True)]
a = A[A.level.isin(["district", "split"]) & A.uid.isin(P.uid)].merge(P[["uid", "start_year", "usd"]], on="uid")
a = a.rename(columns={"unit": "unit_", "weight": "w"}).assign(unit=lambda d: d.unit_, year=lambda d: d.start_year - 1)
a["sgrp"] = size_grp(a.usd)
U = DP[["k", "province", "capital", "net_census", "tt_dry_min", "paved_km", "area_km2", "deprivation", "pop2011"]].rename(columns={"k": "unit"})
U = U.assign(net=z(U.net_census), road=z(-np.log(U.tt_dry_min)), paved=z(np.log1p(U.paved_km / U.area_km2 * 100)), dep=z(U.deprivation),
             lnpop=np.log(U.pop2011), cap=U.capital)
a = a[a.unit.isin(U.unit)]
stacked(a, U[["unit", "province", "net", "road", "paved", "dep", "lnpop", "cap"]], range(2008, 2013), [1, 2, 3], "province", "R-A Nepal 2009-2013")

# ---------------- R-B Timor-Leste 2009-2015 ----------------
P = pd.read_csv(ROOT / "data/processed/oda_projects.csv")
A = pd.read_csv(ROOT / "data/processed/oda_assignments.csv")
W = pd.read_csv(ROOT / "data/processed/windows_TL.csv")
PA = pd.read_csv(ROOT / "data/processed/posto_access.csv")
UN = pd.read_csv(ROOT / "data/processed/units.csv")
P = P[P.start_year.between(2009, 2015) & P.usd.notna() & (P.usd > 0) & ~P["cat"].isin(DEVX) & (P.ict != True)]
U = W[W.window == "W1"][["unit", "mun_pcode", "access_net", "deprivation", "pop2010", "area_km2"]].copy()
out.append(f"TL units: {len(U)} | posto_access pcodes matched: {U.unit.isin(PA.pcode).sum()} | units.csv cols: {UN.columns.tolist()[:12]}")
pa = PA.set_index("pcode")
if U.unit.isin(PA.pcode).sum() < len(U) * 0.8 and "unit" in UN.columns:  # merged Dili unit: aggregate postos to units
    m = UN.set_index("pcode").unit if "pcode" in UN.columns else None
    if m is not None:
        PA["unit"] = PA.pcode.map(m)
        pa = PA.groupby("unit").agg(tt_dry_min=("tt_dry_min", "mean"), paved_km=("paved_km", "sum"))
U["tt_dry_min"] = U.unit.map(pa.tt_dry_min); U["paved_km"] = U.unit.map(pa.paved_km)
U = U.dropna(subset=["tt_dry_min"])
dili = U.sort_values("pop2010").unit.iloc[-1]
U = U.assign(net=z(U.access_net), road=z(-np.log(U.tt_dry_min)), paved=z(np.log1p(U.paved_km / U.area_km2 * 100)), dep=z(U.deprivation),
             lnpop=np.log(U.pop2010), cap=(U.unit == dili).astype(int))
a = A[A.uid.isin(P.uid) & A.level.isin(["posto", "municipio"])].merge(P[["uid", "start_year", "usd"]], on="uid")
# municipio-level rows are split equally over the postos of the municipality (as in the original study)
mun_units = U.groupby("mun_pcode").unit.apply(list).to_dict()
rows = []
for r in a.itertuples():
    if r.level == "posto":
        rows.append((r.uid, r.unit, r.weight, r.start_year - 1, r.usd))
    else:
        us = mun_units.get(r.unit, [])
        rows += [(r.uid, u, r.weight / len(us), r.start_year - 1, r.usd) for u in us]
a = pd.DataFrame(rows, columns=["uid", "unit", "w", "year", "usd"])
a = a[a.unit.isin(U.unit)]
nlarge = a[a.usd > 1e7].uid.nunique()
three = nlarge >= 15
a["sgrp"] = size_grp(a.usd, three)
out.append(f"TL located projects: {a.uid.nunique()} | > $10M: {nlarge} -> {'3' if three else '2'} size groups (pre-registered rule)")
stacked(a, U[["unit", "mun_pcode", "net", "road", "paved", "dep", "lnpop", "cap"]], range(2008, 2015), [1, 2, 3] if three else [1, 2],
        "mun_pcode", "R-B Timor-Leste 2009-2015")

(ROOT / "outputs/results_replication_gradient.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
