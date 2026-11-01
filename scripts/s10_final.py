"""Pre-specified primary model + fixed robustness battery + multiple-testing correction (methodology protocol v1).

PRIMARY (P):  PPML,  Y = # non-infrastructure development projects with posto-level location (tier A/B), start 2000-2026
              X = road/physical sub-index (z), network sub-index (z), census deprivation 2010 (z), log census pop 2010,
              municipality fixed effects; sample = 61 non-Dili-urban units; HC1 SE.
Hypotheses:   H1a b_net > 0,  H1b b_phys > 0,  H1c b_dep < 0 (need-reverse allocation).
Outputs: outputs/tables/final_*.csv, outputs/results_final.txt
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data" / "processed", ROOT / "outputs" / "tables"
LOG = open(ROOT / "outputs" / "results_final.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")


z = lambda s: (s - s.mean()) / s.std()
U = pd.read_csv(PROC / "units.csv")
X = pd.read_csv(PROC / "unit_project_links.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
PRJ = pd.read_csv(PROC / "oda_projects.csv")
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}

# tier-specific precise outcomes (unit-level counts)
tier = AS[AS.level == "posto"].groupby("uid").tier.min()
X["tier"] = X.uid.map(tier)
d = X[(X.grp == "dev") & (X.precise == 1)]
U = U.merge(d.groupby("unit").cnt.sum().rename("y_AB"), on="unit", how="left")
U = U.merge(d[d.tier == "A"].groupby("unit").cnt.sum().rename("y_A"), on="unit", how="left")
U[["y_AB", "y_A"]] = U[["y_AB", "y_A"]].fillna(0)
U["dili"] = (U.unit == "TL06U").astype(int)
U["oe_at"] = ((U.mun_pcode == "TL12") | (U.unit_name == "Atauro")).astype(int)
for c, s in [("phys", U.access_phys), ("net", U.access_net), ("dep", U.deprivation), ("rwi_poor", -U.rwi), ("ntl", U.log_ntl2010)]:
    U[c] = z(s)
U["lpop"] = np.log(U.cpop2010)
D = U[U.dili == 0].copy()
BASE = "phys + net + dep + lpop"


def ppml(f, data, cov="HC1"):
    m = smf.glm(f, data, family=sm.families.Poisson())
    if cov == "cluster":
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.mun_pcode)[0]})
    return m.fit(cov_type="HC1")


SPECS = [
    ("P  PRIMARY", "y_AB ~ " + BASE + " + C(mun_pcode)", D, "HC1"),
    ("R1 tier A (exact points) only", "y_A ~ " + BASE + " + C(mun_pcode)", D, "HC1"),
    ("R2 all located incl. municipal split, no FE", "n_dev ~ " + BASE, D, "HC1"),
    ("R3 + pre-period hub: log night lights 2010", "y_AB ~ " + BASE + " + ntl + C(mun_pcode)", D, "HC1"),
    ("R4 + municipal-capital dummy", "y_AB ~ " + BASE + " + mun_capital + C(mun_pcode)", D, "HC1"),
    ("R5 municipality-clustered SE (12 clusters)", "y_AB ~ " + BASE + " + C(mun_pcode)", D, "cluster"),
    ("R6 RWI instead of census deprivation", "y_AB ~ phys + net + rwi_poor + lpop + C(mun_pcode)", D, "HC1"),
    ("R7a projects started 2000-2015", "np_dev_e1 ~ " + BASE + " + C(mun_pcode)", D, "HC1"),
    ("R7b projects started 2016-2026", "np_dev_e2 ~ " + BASE + " + C(mun_pcode)", D, "HC1"),
    ("R8 all 62 incl. Dili (dummy)", "y_AB ~ " + BASE + " + dili + C(mun_pcode)", U, "HC1"),
    ("R9 excl. Oecusse & Atauro (sea access)", "y_AB ~ " + BASE + " + C(mun_pcode)", D[D.oe_at == 0], "HC1"),
    ("R10 no municipality FE", "y_AB ~ " + BASE, D, "HC1"),
]
rows = []
say("=" * 100 + "\nPRIMARY MODEL AND PRE-SPECIFIED ROBUSTNESS (coefficients per 1 SD; PPML)")
for name, f, data, cov in SPECS:
    m = ppml(f, data, cov)
    rec = {"spec": name, "n": int(m.nobs)}
    for v in ["phys", "net", "dep", "rwi_poor", "lpop", "ntl", "mun_capital", "dili"]:
        if v in m.params:
            rec[f"b_{v}"], rec[f"se_{v}"], rec[f"p_{v}"] = m.params[v], m.bse[v], m.pvalues[v]
    rows.append(rec)
    need = "dep" if "dep" in m.params else "rwi_poor"
    extra = "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in ["ntl", "mun_capital", "dili"] if v in m.params)
    say(f"  {name:<46} N={int(m.nobs):>2}  road={m.params['phys']:+.3f}(p={m.pvalues['phys']:.3f})  "
        f"network={m.params['net']:+.3f}(p={m.pvalues['net']:.3f})  need={m.params[need]:+.3f}(p={m.pvalues[need]:.3f})  {extra}")
R = pd.DataFrame(rows); R.to_csv(TAB / "final_robustness.csv", index=False)

# leave-one-municipality-out
lo = []
for mc in sorted(D.mun_pcode.unique()):
    m = ppml("y_AB ~ " + BASE + " + C(mun_pcode)", D[D.mun_pcode != mc])
    lo.append({"dropped": mc, "b_net": m.params["net"], "p_net": m.pvalues["net"], "b_phys": m.params["phys"], "b_dep": m.params["dep"]})
LO = pd.DataFrame(lo); LO.to_csv(TAB / "final_jackknife.csv", index=False)
say(f"\n  leave-one-municipality-out: network b in [{LO.b_net.min():+.3f}, {LO.b_net.max():+.3f}], p<0.05 in {(LO.p_net < .05).sum()}/{len(LO)}; "
    f"road b in [{LO.b_phys.min():+.3f}, {LO.b_phys.max():+.3f}]; deprivation b in [{LO.b_dep.min():+.3f}, {LO.b_dep.max():+.3f}]")

# randomization inference for the primary network coefficient (permute net within municipality)
rng = np.random.default_rng(42)
b0 = ppml("y_AB ~ " + BASE + " + C(mun_pcode)", D).params["net"]
perm = []
for _ in range(500):
    Dp = D.copy()
    Dp["net"] = Dp.groupby("mun_pcode").net.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    perm.append(ppml("y_AB ~ " + BASE + " + C(mun_pcode)", Dp).params["net"])
p_ri = (np.sum(np.abs(perm) >= abs(b0)) + 1) / (len(perm) + 1)
say(f"  randomization inference (network permuted within municipality, 500 draws): two-sided p = {p_ri:.3f}")

# ---------- category family with multiple-testing correction ----------
say("\n" + "=" * 100 + "\nSECTOR FAMILY (no FE, all located incl. municipal split; Holm and Benjamini-Hochberg)")
fam = []
for y, lab in [("n_SOC_education", "Education"), ("n_SOC_health", "Health"), ("n_SOC_water", "Water & sanitation"),
               ("n_SOC_other", "Other social"), ("n_GOV", "Governance"), ("n_PROD", "Productive"), ("n_ENV_MULTI", "Env./multisector"),
               ("n_ACCESS_transport", "Transport infra"), ("n_human", "Humanitarian"), ("n_dev_ict", "ICT component"), ("n_dev_nonict", "Non-ICT")]:
    m = ppml(f"{y} ~ {BASE}", D)
    for v in ["phys", "net", "dep"]:
        fam.append({"group": lab, "term": v, "coef": m.params[v], "se": m.bse[v], "p": m.pvalues[v]})
F = pd.DataFrame(fam)
F["p_holm"] = multipletests(F.p, method="holm")[1]
F["q_bh"] = multipletests(F.p, method="fdr_bh")[1]
F.to_csv(TAB / "final_sector_family.csv", index=False)
for r in F[F.p < 0.1].itertuples():
    say(f"  {r.group:<20} {r.term:<5} b={r.coef:+.3f} p={r.p:.3f}  Holm={r.p_holm:.3f}  BH q={r.q_bh:.3f}")
LOG.close()
