"""District census measures from the IPUMS International extract (pre-registration 사전등록_센서스분석.md, C1 and C2).

Input : data/raw/cambodia/ipums/<extract>.csv(.gz) and its DDI codebook <extract>.xml (IPUMS harmonised variables)
Output: data/processed/cambodia/kh_census.csv, one row per district (GEO2_KH) and census year:
        dep_electric, dep_water, dep_sanitation, dep_floor, dep_literacy, dep_school  (share of people in deprived households)
        deprivation (equal-weight mean of available shares, standardised within year), deprivation_pc1, phone_share, n_indicators
Codes are classified from the DDI value labels by the keyword rules below, not from numeric codes, and every label is printed
so the mapping can be checked against the codebook. A label that matches no rule stops the script.
Substitutes as pre-registered: WATSUP -> DRINKWTR, TOILET -> SEWAGE, CELL -> PHONE. Fewer than 4 indicators -> H-C1 exploratory.
Usage: python kh_census_build.py [path/to/extract.csv]
"""
import os, re, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw/cambodia/ipums", ROOT / "data/processed/cambodia"
MISSING = r"niu|not in universe|unknown|missing|not reported|blank"
# indicator: (variables in order of preference, deprived-label rule, not-deprived-label rule)
RULES = {
    "electric":   (["ELECTRIC"], r"^no\b|no access|not available", r"^yes\b|has access"),
    "water":      (["WATSUP", "DRINKWTR"], r"no piped|unprotected|surface|river|lake|pond|stream|vendor|tanker|cart|other source|^no\b",
                   r"piped|tap|borehole|tube ?well|protected|rain|bottled|^yes\b"),
    "sanitation": (["TOILET", "SEWAGE"], r"no toilet|no flush|open|bush|field|bucket|unimproved|none|not connected|pit latrine without",
                   r"flush|improved|ventilated|vip|sewer|septic|pit latrine with slab|connected"),
    "floor":      (["FLOOR"], r"earth|dirt|sand|mud|dung|unfinished|bamboo|palm|no floor", r"cement|concrete|tile|wood|parquet|vinyl|ceramic|finished|marble|brick"),
}
PERSON = {"literacy": ("LIT", r"illiterate|^no\b|cannot", r"literate|^yes\b|can read"),
          "school":   ("SCHOOL", r"no, not in school|not attending|never attended|^no\b|attended in past", r"yes, in school|attending|^yes\b")}
PHONE = (["CELL", "PHONE"], r"^no\b|not available|none", r"^yes\b|available|has")


def labels(ddi):
    """{VAR: {code: label}} from an IPUMS DDI codebook."""
    out = {}
    for v in ET.parse(ddi).getroot().iter():
        if v.tag.endswith("var") and v.get("name"):
            cats = {}
            for c in v:
                if c.tag.endswith("catgry"):
                    val = next((x.text for x in c if x.tag.endswith("catValu")), None)
                    lab = next((x.text for x in c if x.tag.endswith("labl")), "")
                    if val is not None:
                        cats[int(val)] = (lab or "").strip()
            out[v.get("name").upper()] = cats
    return out


def classify(var, cats, dep, ok):
    """code -> 1 deprived, 0 not deprived, NaN missing. Every non-missing label must match exactly one rule."""
    m = {}
    for code, lab in cats.items():
        l = lab.lower()
        if re.search(MISSING, l):
            m[code] = np.nan; continue
        d_ = bool(re.search(dep, l))
        o_ = bool(re.search(ok, re.sub(dep, " ", l) if d_ else l))   # "no piped water": the negative phrase is removed before testing "piped"
        if d_ == o_:
            raise SystemExit(f"{var} code {code} '{lab}': matches {'both' if d_ else 'no'} rule(s); add it to RULES after checking the codebook")
        m[code] = 1.0 if d_ else 0.0
        print(f"   {var:<9s} {code:>5d}  {'DEPRIVED ' if d_ else 'ok       '} {lab}")
    return m


def build(csv):
    ddi = csv.with_suffix("").with_suffix(".xml") if csv.suffix == ".gz" else csv.with_suffix(".xml")
    L = labels(ddi)
    P = pd.read_csv(csv, low_memory=False); P.columns = P.columns.str.upper()
    P["unit"] = P.GEO2_KH.astype("int64"); P["hh"] = P.YEAR.astype(str) + "_" + P.SERIAL.astype(str)
    rows, used = [], {}
    for name, (vars_, dep, ok) in RULES.items():
        v = next((x for x in vars_ if x in P and x in L), None)
        if v is None:
            continue
        used[name] = v; P[f"d_{name}"] = P[v].map(classify(v, L[v], dep, ok))
    for name, (v, dep, ok) in PERSON.items():
        if v not in P or v not in L:
            continue
        used[name] = v; x = P[v].map(classify(v, L[v], dep, ok))
        if name == "literacy":       # household deprived if no member aged 15+ is literate
            lit = x.where(P.AGE >= 15)
            P["d_literacy"] = P.hh.map(lit.groupby(P.hh).apply(lambda s: np.nan if s.notna().sum() == 0 else float((s == 0).sum() == 0)))
        else:                        # household deprived if any child aged 6-14 is not in school
            sch = x.where(P.AGE.between(6, 14))
            P["d_school"] = P.hh.map(sch.groupby(P.hh).apply(lambda s: 0.0 if s.notna().sum() == 0 else float((s == 1).any())))
    pv = next((x for x in PHONE[0] if x in P and x in L), None)
    if pv:
        P["has_phone"] = 1 - P[pv].map(classify(pv, L[pv], PHONE[1], PHONE[2]))
    ind = [f"d_{k}" for k in used]
    print("indicators used:", used, "| phone:", pv)
    for (yr, u), g in P.groupby(["YEAR", "unit"]):
        r = {"year": yr, "unit": u, "persons": len(g)}
        for c in ind:
            ok_ = g[c].notna(); r["dep_" + c[2:]] = np.average(g.loc[ok_, c], weights=g.loc[ok_, "PERWT"]) if ok_.any() else np.nan
        if pv:
            h = g.drop_duplicates("hh"); ok_ = h.has_phone.notna()
            r["phone_share"] = np.average(h.loc[ok_, "has_phone"], weights=h.loc[ok_, "HHWT"]) if ok_.any() else np.nan
        rows.append(r)
    C = pd.DataFrame(rows); dep = [c for c in C if c.startswith("dep_")]
    C["n_indicators"] = len(dep)
    for yr, g in C.groupby("year"):
        mean = g[dep].mean(axis=1); C.loc[g.index, "deprivation"] = (mean - mean.mean()) / mean.std()
        Z = (g[dep] - g[dep].mean()) / g[dep].std(); w_, V_ = np.linalg.eigh(np.cov(Z.fillna(0).T)); pc = V_[:, -1] * np.sign(V_[:, -1].sum())
        s = Z.fillna(0).values @ pc; C.loc[g.index, "deprivation_pc1"] = (s - s.mean()) / s.std()
    if len(dep) < 4:
        print(f"WARNING: only {len(dep)} deprivation indicators; H-C1 is exploratory under the pre-registration")
    C.to_csv(os.environ.get("KH_CENSUS_OUT", PROC / "kh_census.csv"), index=False)   # override only for tests
    print(C.groupby("year")[dep + ["phone_share"] * bool(pv)].mean().round(3).to_string())
    return C


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else next(iter(sorted(RAW.glob("*.csv*"))), None)
    if src is None:
        raise SystemExit(f"no IPUMS extract in {RAW}")
    build(src)
