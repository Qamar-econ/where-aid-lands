"""Automated consistency checks, from the district sample to the numbers on the site. Exit code 1 if any check fails.

A  sample      both sample builders agree after the spec 9-26 reclassification; weights are conserved; every flagged
               location has a class
B  dataset     kh_analysis.csv matches the sample; standardised regressors have mean 0 and SD 1
C  estimates   headline models re-fitted from kh_analysis.csv equal the published effects; reported p = max of its parts;
               no randomisation/bootstrap p below 1/(B+1); Holm adjustment recomputed; results.json = published tables
D  code        no randomisation p computed without the observed draw
E  site        every headline number on the site equals its source table (claims list below)
Run after kh_run_all.sh:  python kh_check.py
"""
import os, re, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
PROCD, T, SITE = ROOT / "data/processed/cambodia", ROOT / "outputs/tables", ROOT / "site"
fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def load(script, stop):
    ns = {"__file__": str(HERE / script)}
    exec((HERE / script).read_text(encoding="utf-8").split(stop)[0], ns)
    return ns


# ---------------- A. sample ----------------
s06 = load("kh_s06_ladder.py", "LADDER = [")
cur, DF = s06["cur"], s06["DF"]
s04 = load("kh_s04_models.py", "\nD = np.hypot")
A4 = s04["A"]; A4 = A4[A4.level == "adm2"]
n06, n04 = cur[cur.unit.isin(DF.unit)].groupby("unit").w.sum(), A4[A4.unit.isin(DF.unit)].groupby("unit").w.sum()   # 141 analysis districts
J = pd.concat([n06.rename("a"), n04.rename("b")], axis=1).fillna(0)
check("A1 sample builders agree (kh_s04 vs kh_s06), per district", np.allclose(J.a, J.b, atol=1e-9), f"max diff {np.abs(J.a - J.b).max():.2e}")

os.environ["KH_GEOCODE"] = "original"
orig = load("kh_s06_ladder.py", "LADDER = [")["cur"]
os.environ.pop("KH_GEOCODE")
cl = pd.read_csv(PROCD / "kh_geocode_classified.csv"); fu = pd.read_csv(PROCD / "kh_geocode_flagged_units.csv").merge(cl[["project_id", "class"]], on="project_id", how="left")
nat = orig.merge(fu[fu["class"] == "NATIONAL"][["project_id", "unit"]], on=["project_id", "unit"]).w.sum()
check("A2 weight conserved (main = original - NATIONAL)", abs(cur.w.sum() - (orig.w.sum() - nat)) < 1e-6, f"{cur.w.sum():.4f} vs {orig.w.sum():.4f} - {nat:.4f}")
pw = cur.groupby("project_id").w.sum()
out_w = cur[~cur.unit.isin(DF.unit)].w.sum()
check("A3 activity weights in (0, 1]; weight outside the analysis districts only in excluded units", (pw <= 1 + 1e-9).all() and (cur.w > 0).all()
      and cur[~cur.unit.isin(DF.unit)].unit.isin([116088088]).all(), f"max {pw.max():.4f}; {out_w:.2f} projects in Tonle Sap Lake unit (excluded)")
check("A4 every flagged location has a class", fu["class"].notna().all() and set(cl["class"]) <= {"TOWN", "PROVINCE", "UNCLEAR", "NATIONAL"},
      f"{fu['class'].isna().sum()} unclassified")
flag_orig = orig.merge(fu[["project_id", "unit"]], on=["project_id", "unit"])
check("A5 flagged locations all belong to the original sample", len(flag_orig) == len(fu.drop_duplicates(["project_id", "unit"])), f"{len(flag_orig)} of {len(fu)}")

# ---------------- B. analysis dataset ----------------
an = pd.read_csv(PROCD / "kh_analysis.csv")
nN = an.set_index("unit").N; ref = s06["outcome"](cur).set_index("unit").N
check("B1 kh_analysis.csv N equals the sample", np.allclose(nN.reindex(ref.index).fillna(-1), ref, atol=1e-9), f"sum {nN.sum():.3f} vs {ref.sum():.3f}")
zs = {v: (an[v].mean(), an[v].std()) for v in ["phys", "conn", "lights", "need"] if v in an}
check("B2 standardised regressors: mean 0, SD 1", all(abs(m) < 1e-6 and abs(s - 1) < 1e-6 for m, s in zs.values()), str({k: (round(m, 4), round(s, 4)) for k, (m, s) in zs.items()}))

# ---------------- C. estimates ----------------
pub = pd.read_csv(PROCD / "kh_published.csv").set_index("check")
an["prov"] = an.prov.astype(str)
for v, key in [("phys", "single:phys"), ("conn", "single:conn"), ("lights", "single:lights"), ("need", "single:need")]:
    d = an.dropna(subset=[v])
    r = smf.glm(f"N ~ {v} + lnpop + C(prov)", data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)
    pct = 100 * (np.exp(r.params[v]) - 1)
    check(f"C1 {key}: re-fit from kh_analysis.csv = published", abs(pct - pub.loc[key, "effect_pct"]) < 0.05 and abs(r.pvalues[v] - pub.loc[key, "p_cluster"]) < 1e-3,
          f"{pct:.2f}% vs {pub.loc[key, 'effect_pct']:.2f}%, p {r.pvalues[v]:.4f} vs {pub.loc[key, 'p_cluster']:.4f}")
parts = ["p_cluster", "p_conley50", "p_conley100", "p_ri"]
bad = [k for k, r in pub.iterrows() if pd.notna(r.get("p_reported")) and abs(r.p_reported - np.nanmax([r.get(c, np.nan) for c in parts])) > 1e-9]
check("C2 reported p = max(cluster, Conley 50, Conley 100, RI)", not bad, ", ".join(bad))
low = []
for f in T.glob("*.csv"):
    t = pd.read_csv(f)
    for c in [c for c in t.columns if c in ("p_ri", "p_boot", "p_wild", "p_r", "p_u")]:
        x = pd.to_numeric(t[c], errors="coerce")
        if (x == 0).any():
            low.append(f"{f.name}:{c}")
check("C3 no randomisation/bootstrap p equal to 0", not low, ", ".join(low))
mn = pd.read_csv(T / "kh_main.csv").set_index("var")
dec = mn.loc[[v for v in ["phys", "conn", "ntl"] if v in mn.index]].sort_values("p_conservative"); run, k, ok = 0, len(dec), True
for i, (ix, r) in enumerate(dec.iterrows()):
    run = max(run, min(1, (k - i) * r.p_conservative)); ok &= abs(run - r.p_holm) < 1e-9
check("C4 Holm adjustment recomputed", ok, dec[["p_conservative", "p_holm"]].round(4).to_dict())
res = json.loads((SITE / "data/results.json").read_text(encoding="utf-8"))
m = {r["var"]: r for r in res["main"]}
check("C5 results.json main = kh_main.csv", all(abs(m[v]["pct"] - mn.loc[v, "pct"]) < 0.01 and abs(m[v]["p_conservative"] - round(mn.loc[v, "p_conservative"], 3)) < 1e-3
                                                  for v in ["phys", "conn", "ntl"] if v in m), "")

# ---------------- D. code ----------------
pat = re.compile(r"np\.mean\(np\.abs\([^)]*\)\s*>=")
hits = [f.name for f in HERE.glob("kh_*.py") if f.name != "kh_check.py" and pat.search(f.read_text(encoding="utf-8"))]
check("D1 randomisation p always includes the observed draw", not hits, ", ".join(hits))

# ---------------- E. site claims ----------------
html = (SITE / "index.html").read_text(encoding="utf-8")
txt = re.sub(r"<[^>]+>", " ", html)
need = pd.read_csv(T / "kh_need_conservative.csv").iloc[0]; jt = pd.read_csv(T / "kh_joint_test.csv").iloc[0]
claims = [("physical access effect", f"+{mn.loc['phys', 'pct']:.0f}%"), ("physical access p", f"p = {mn.loc['phys', 'p_conservative']:.3f}"),
          ("poverty effect", f"{need.pct:.0f}%".replace("-", "−")), ("poverty p", f"p = {need.p_conservative:.3f}"),
          ("joint test p", f"p = {jt.p_conservative:.3f}")]
for name, s in claims:
    check(f"E1 site shows {name} ({s})", s in txt)
stale = [s for s in ["+165%", "2.6 times", "−49%", "p = 0.042"] if s in txt]
check("E2 no stale headline numbers on the site", not stale, ", ".join(stale))

print(f"\n{len(passes)} passed, {len(fails)} failed" + (": " + "; ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
