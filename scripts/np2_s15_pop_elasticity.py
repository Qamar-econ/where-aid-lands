"""Population elasticity of project counts across the three samples (exploratory; not pre-registered).
theta < 1: projects grow less than proportionally with population -> more projects per person in small units.
Specs: (a) ln pop + upper x year FE only; (b) + connectivity, road, paved, deprivation, capital.
Output: outputs/results_pop_elasticity.txt
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).with_name("np2_s14_replication.py").read_text(encoding="utf-8")
out = []


def theta(p, upper, tag):
    p = p.assign(gfe=p[upper].astype(str) + "_" + p.year.astype(str))
    for lab, f in [("pop only", "N ~ lnpop + C(gfe)"), ("+ covariates", "N ~ lnpop + net + road + paved + dep + cap + C(gfe)")]:
        m = smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.unit)[0]}, maxiter=300)
        t, se = m.params["lnpop"], m.bse["lnpop"]
        out.append(f"{tag:<28} {lab:<13} theta {t:.2f} (95% CI {t - 1.96 * se:.2f}-{t + 1.96 * se:.2f}; p[theta=1] {2 * norm.sf(abs(t - 1) / se):.3f})")


# capture the unit tables and assignments built in the replication script without running its estimation
ns = {"__file__": str(Path(__file__).with_name("np2_s14_replication.py"))}
exec(SRC.split("# ---------------- R-A")[0], ns)
exec("stacked = lambda *a, **k: None\n#" + SRC.split("# ---------------- R-A")[1].split("(ROOT / \"outputs/results_replication")[0]
     .replace("stacked(a, U[[\"unit\", \"province\"", "NP_A, NP_U = a.copy(), U.copy(); stacked(a, U[[\"unit\", \"province\"")
     .replace("stacked(a, U[[\"unit\", \"mun_pcode\"", "TL_A, TL_U = a.copy(), U.copy(); stacked(a, U[[\"unit\", \"mun_pcode\""), ns)


def build(a, U, years):
    y = a.groupby(["unit", "year"]).w.sum().rename("N").reset_index()
    g = pd.MultiIndex.from_product([U.unit, years], names=["unit", "year"]).to_frame(index=False)
    return g.merge(U, on="unit").merge(y, on=["unit", "year"], how="left").fillna({"N": 0})


theta(build(ns["NP_A"], ns["NP_U"], range(2008, 2013)), "province", "Nepal 2009-2013 (districts)")
theta(build(ns["TL_A"], ns["TL_U"], range(2008, 2015)), "mun_pcode", "Timor-Leste 2009-2015 (postos)")
B = pd.read_csv(ROOT / "data/processed/nepal/np2_panel.csv").rename(columns={"k": "unit", "net_idx": "net", "capital": "cap"})
theta(B, "province", "Nepal 2019-2024 (districts)")
(ROOT / "outputs/results_pop_elasticity.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
