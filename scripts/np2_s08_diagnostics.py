"""POST-HOC diagnostics for the pre-specified M1 connectivity result (not part of the pre-analysis plan; reported as such).
D1  urbanisation / growth controls: density, NTL 2010, RWI, population growth 2011-23, urban-hub dummy (district has a city > 100k)
D2  placebo outcomes: infrastructure and humanitarian projects (same geocoder; planning year = start year - 1)
D3  text-salience check: does connectivity predict how often a district is NAMED in project text that is coded national?
Output: outputs/results_np2_diagnostics.txt
"""
import re, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from np2_geo import geocode_text, best_by_district  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_csv(ROOT / "data/processed/nepal/np2_panel.csv")
D = pd.read_csv(ROOT / "data/processed/nepal/districts.csv")
A = pd.read_csv(ROOT / "data/processed/nepal/district_access.csv")
z = lambda s: (s - s.mean()) / s.std()
P = P.merge(D[["k", "density", "g_11_23"]], on="k").merge(A[["k", "ntl2010", "rwi"]], on="k")
P["ldens"], P["lntl"], P["grow"], P["rwi_z"] = z(np.log(P.density)), z(np.log1p(P.ntl2010)), z(P.g_11_23), z(P.rwi)
HUB = {"kathmandu", "lalitpur", "bhaktapur", "kaski", "morang", "parsa", "rupandehi", "chitawan", "chitwan", "banke", "kailali", "sunsari", "dhanusha", "jhapa"}
P["hub"] = P.k.isin(HUB).astype(int)
M1 = "N ~ net_idx + road + paved + dep + lnpop + capital + C(pt)"
out = []


def fit(tag, f, p):
    m = smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.k)[0]}, maxiter=200)
    out.append(f"{tag:<44} net {m.params['net_idx']:+.3f} (se {m.bse['net_idx']:.3f}, p {m.pvalues['net_idx']:.3f})"
               + "".join(f" | {t} {m.params[t]:+.3f} (p {m.pvalues[t]:.3f})" for t in ["road", "ldens", "lntl", "grow", "rwi_z", "hub"] if t in m.params))
    return m


fit("D0 pre-specified M1", M1, P)
fit("D1a + log density", M1 + " + ldens", P)
fit("D1b + NTL 2010", M1 + " + lntl", P)
fit("D1c + pop growth 2011-23", M1 + " + grow", P)
fit("D1d + RWI", M1 + " + rwi_z", P)
fit("D1e + hub dummy", M1 + " + hub", P)
fit("D1f + all urbanisation controls", M1 + " + ldens + lntl + grow + rwi_z + hub", P)
fit("D1g non-hub districts only", M1.replace(" + capital", ""), P[P.hub == 0])

# D2 placebo: infra / humanitarian projects
I = pd.read_csv(ROOT / "data/interim/np2_inventory.csv")
I = I[~I.dev & I.start_year.between(2020, 2025)]
ft = pd.read_csv(ROOT / "data/interim/iati_fulltext_NP.csv").drop_duplicates("aid").set_index("aid").fillna("")
CAP = {"kathmandu", "lalitpur", "bhaktapur"}
rows = []
for r in I.itertuples():
    aid = r.uid[5:] if r.uid.startswith("IATI-") else None
    core = f"{r.title} {r.desc if isinstance(r.desc, str) else ''}"
    h = geocode_text(core + (" " + ft.text.get(aid, "") if aid else ""))
    cd = {x["district"] for x in geocode_text(core)}
    h = [x for x in h if x["district"] not in CAP or x["district"] in cd]
    b = [x for x in best_by_district(h) if x["precision"] <= 4]
    nd = len({x["district"] for x in b})
    for d in {x["district"] for x in b}:
        rows.append({"k": d, "year": int(r.start_year) - 1, "w": 1 / nd, "grp": "infra" if str(r.cat).startswith("ACCESS") else ("hum" if r.cat == "HUMAN" else "other")})
Q = pd.DataFrame(rows)
out.append(f"placebo projects located: {Q.groupby('grp').size().to_dict()} | categories in non-dev: {I['cat'].value_counts().head(6).to_dict()}")
for g in ["infra", "hum"]:
    y = Q[Q.grp == g].groupby(["k", "year"]).w.sum().rename("Np").reset_index()
    pp = P.merge(y, on=["k", "year"], how="left").fillna({"Np": 0})
    if pp.Np.sum() > 20:
        fit(f"D2 placebo {g} (N={pp.Np.sum():.0f})", M1.replace("N ~", "Np ~"), pp)

# D3 text salience: mentions of a district in projects coded national (precision 8) per planning year
ST = pd.read_csv(ROOT / "data/processed/nepal/np2_project_status.csv")
PL = pd.read_csv(ROOT / "data/processed/nepal/np2_planning.csv")
nat = PL.merge(ST, on="uid"); nat = nat[(nat.best_precision == 8) & nat.planning_year.between(2019, 2024)]
INV = pd.read_csv(ROOT / "data/interim/np2_inventory.csv").set_index("uid")
from gazetteer_nepal import GEO  # noqa: E402
mm = []
for r in nat.itertuples():
    t = f"{INV.title.get(r.uid, '')} {INV.desc.get(r.uid, '')}"
    for d in GEO.parse(t)[0]:
        mm.append({"k": d, "year": r.planning_year})
out.append(f"D3 district mentions inside national-coded projects: {len(mm)} (if small, text salience cannot drive M1)")

with open(ROOT / "outputs/results_np2_diagnostics.txt", "w", encoding="utf-8") as fh:
    for s in out:
        print(s); fh.write(s + "\n")
