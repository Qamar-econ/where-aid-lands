"""Diagnose the Nepal 'lag effect' (2011 network -> 2019-23 new projects) against competing explanations A-G.

A geocoding method   B development trajectory   C chance / multiple testing   D earthquake composition
E collinearity       F donor composition         G generic hub (placebo outcomes)
Output: outputs/results_diagnose.txt, outputs/tables/diagnose_nepal.csv
"""
import warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
LOG = open(ROOT / "outputs/results_diagnose.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}
EQ14 = {"gorkha", "dhading", "nuwakot", "rasuwa", "sindhupalchok", "dolakha", "kavre", "kathmandu", "lalitpur", "bhaktapur", "makwanpur", "ramechhap", "okhaldhunga", "sindhuli"}

PR = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv")
bb = PR.source.eq("BudgetBook5")
med = PR.assign(d=PR.end_year - PR.start_year).query("d>=0 and d<=30").groupby("source").d.median()
PR["end_year"] = PR.end_year.where(PR.end_year >= PR.start_year, PR.start_year + PR.source.map(med).fillna(4))
AS = pd.read_csv(ROOT / "data/processed/nepal/oda_assignments.csv")
AS = AS[(AS.level == "district") & AS.tier.isin(["A", "B"])]
L = AS.merge(PR[["uid", "cat", "start_year", "end_year", "source", "donor_group", "eq"]], on="uid")
D = pd.read_csv(ROOT / "data/processed/nepal/district_panel.csv")
G = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")[["k", "geometry"]]
with rasterio.open(ROOT / "data/raw/nepal/ntl/ntl_2020_NPL.tif") as s:
    G["ntl2020"] = [mask(s, [g], crop=True, filled=True, nodata=0)[0][0].astype(float).sum() for g in G.geometry]
D = D.merge(G[["k", "ntl2020"]], on="k")
S = D[D.capital == 0].copy()
S = S.assign(Net=z(S.net_census), Mob=z(S.mobile_pct), Int=z(np.log1p(S.internet_pct)), Road=z(S.phys_idx), Dep=z(S.deprivation),
             lpop=np.log(S.pop2011), fe=S.province, gpop=z(S.g_11_23), gntl=z(np.log1p(S.ntl2020) - np.log1p(S.ntl2010)),
             ntl=z(np.log1p(S.ntl2010 / S.area_km2)), dens=z(np.log(S.density)))


def counts(mask_, a, b, new=True):
    sub = L[mask_ & (L.start_year.between(a, b) if new else ((L.start_year <= b) & (L.end_year >= a)))]
    return S.k.map(sub.groupby("unit").uid.nunique()).fillna(0).values


dev = L.cat.isin(DEV)
S["new1"], S["act1"] = counts(dev, 2009, 2013), counts(dev, 2009, 2013, False)
S["new2"], S["act2"] = counts(dev, 2019, 2023), counts(dev, 2019, 2023, False)
S["y1"] = np.log1p(S.act1)
BASE = "Net + Road + Dep + lpop + C(fe)"


def ppml(y, rhs, d=None):
    d = S if d is None else d
    return smf.glm(f"{y} ~ {rhs}", d, family=sm.families.Poisson()).fit(cov_type="HC1")


def show(tag, m, v="Net"):
    say(f"  {tag:<58} {v}={m.params[v]:+.3f} (p={m.pvalues[v]:.3f})")
    return {"test": tag, "coef": m.params[v], "p": m.pvalues[v]}


rows = []
say("BASELINE")
rows += [show("W1 new (2009-13)", ppml("new1", BASE)), show("W1 active", ppml("act1", BASE)),
         show("W2 new (2019-23) lag", ppml("new2", BASE)), show("W2 active lag", ppml("act2", BASE))]

say("\nA. geocoding method (W2 new): coordinate-based vs text-based assignments; source mix")
for nm, mk in [("coordinates only", L.method.eq("coord")), ("text only", L.method.eq("text")), ("manual", L.method.eq("manual"))]:
    S["tmp"] = counts(dev & mk, 2019, 2023)
    if S.tmp.sum() >= 15:
        rows.append(show(f"W2 new, {nm} (n={S.tmp.sum():.0f})", ppml("tmp", BASE)))
for nm, mk in [("coordinates only", L.method.eq("coord")), ("text only", L.method.eq("text"))]:
    S["tmp"] = counts(dev & mk, 2009, 2013)
    if S.tmp.sum() >= 15:
        rows.append(show(f"W1 new, {nm} (n={S.tmp.sum():.0f})", ppml("tmp", BASE)))
say("  share of W1 new links by method: " + str(L[dev & L.start_year.between(2009, 2013)].method.value_counts(normalize=True).round(2).to_dict()))
say("  share of W2 new links by method: " + str(L[dev & L.start_year.between(2019, 2023)].method.value_counts(normalize=True).round(2).to_dict()))

say("\nB. development trajectory controls (W2 new)")
for rhs, tag in [(BASE + " + gpop", "+ population growth 2011-23"), (BASE + " + gntl", "+ night-light growth 2010-20"),
                 (BASE + " + ntl + dens", "+ 2010 lights + density"), (BASE + " + gpop + gntl + ntl + dens + y1", "+ all trajectory + W1 projects")]:
    rows.append(show(tag, ppml("new2", rhs)))

say("\nC. chance: randomization inference (Net permuted within province, 1000 draws) and Holm over the 4 baseline tests")
rng = np.random.default_rng(7); b0 = ppml("new2", BASE).params["Net"]; perm = []
for _ in range(1000):
    Sp = S.copy(); Sp["Net"] = Sp.groupby("fe").Net.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    perm.append(ppml("new2", BASE, Sp).params["Net"])
p_ri = (np.sum(np.abs(perm) >= abs(b0)) + 1) / 1001
holm = multipletests([r["p"] for r in rows[:4]], method="holm")[1]
say(f"  RI p (W2 new) = {p_ri:.3f};  Holm-adjusted p of the 4 baseline tests: " + ", ".join(f"{r['test']}={h:.3f}" for r, h in zip(rows[:4], holm)))

say("\nD. earthquake composition")
S["tmp"] = counts(dev & ~L["eq"].astype(bool), 2019, 2023)
rows.append(show("W2 new excl. earthquake-related projects", ppml("tmp", BASE)))
rows.append(show("W2 new excl. 14 earthquake-affected districts", ppml("new2", BASE, S[~S.k.isin(EQ14)])))

say("\nE. collinearity (VIF Net/Dep) and alternative network measures")
X = S[["Net", "Road", "Dep", "lpop"]]; vif = np.diag(np.linalg.inv(np.corrcoef(X.T)))
say(f"  VIF: Net {vif[0]:.1f}, Road {vif[1]:.1f}, Dep {vif[2]:.1f}, lpop {vif[3]:.1f}")
rows.append(show("W2 new, without deprivation", ppml("new2", "Net + Road + lpop + C(fe)")))
rows.append(show("W2 new, mobile ownership only", ppml("new2", "Mob + Road + Dep + lpop + C(fe)"), "Mob"))
rows.append(show("W2 new, internet only", ppml("new2", "Int + Road + Dep + lpop + C(fe)"), "Int"))
rows.append(show("W2 new, no province FE", ppml("new2", "Net + Road + Dep + lpop")))

say("\nF. donor composition (W2 new)")
for nm, grp in [("USA+UK", {"USA", "UK"}), ("UN agencies", {"UN agencies"}), ("other donors", None)]:
    mk = L.donor_group.isin(grp) if grp else ~L.donor_group.isin({"USA", "UK", "UN agencies"})
    S["tmp"] = counts(dev & mk, 2019, 2023)
    if S.tmp.sum() >= 15:
        rows.append(show(f"W2 new, {nm} (n={S.tmp.sum():.0f})", ppml("tmp", BASE)))
say("  W2 new links by donor: " + str(L[dev & L.start_year.between(2019, 2023)].donor_group.value_counts().head(6).to_dict()))

say("\nG. placebo outcomes (W2 new): does 2011 network predict non-development projects too?")
for nm, mk in [("access infrastructure", L.cat.str.startswith("ACCESS")), ("humanitarian", L.cat.eq("HUMAN")), ("other/unallocated", L.cat.eq("OTHER"))]:
    S["tmp"] = counts(mk, 2019, 2023)
    if S.tmp.sum() >= 15:
        rows.append(show(f"W2 new, {nm} (n={S.tmp.sum():.0f})", ppml("tmp", BASE)))

say("\nPOWER: smallest detectable Net effect (80% power, alpha .05) given SE of W1 active model")
se = ppml("act1", BASE).bse["Net"]
say(f"  SE={se:.3f} -> MDE ~ {2.8 * se:.3f} (i.e. {(np.exp(2.8 * se) - 1) * 100:.0f}% per SD)")
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/diagnose_nepal.csv", index=False)
LOG.close()
