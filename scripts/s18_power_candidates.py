"""(1) Power diagnostics for the time-aligned designs (minimum detectable effect, MDE = 2.8 x SE).
(2) Candidate countries with richer observed connectivity data + geocoded aid:
    - AidData country AIMS geocoded releases (GitHub listing)
    - DHS rounds with household mobile-phone ownership (HC_HEFF_H_MPH) and GPS datasets
    - Ookla mobile tests per 100k (from ookla_country_density.csv)
Outputs: outputs/results_power_candidates.txt, outputs/tables/candidate_countries.csv
"""
import re, warnings, requests
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
LOG = open(ROOT / "outputs/results_power_candidates.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()

T = pd.read_csv(ROOT / "data/processed/windows_TL.csv"); N = pd.read_csv(ROOT / "data/processed/nepal/windows_NP.csv")
say("POWER (MDE at 80% power, alpha .05, per 1 SD of network)")
for nm, d, fe, net, road in [("TL W3 posto 61", T[(T.window == "W3") & (T.unit != "TL06U")], "mun_pcode", "net_w", "access_phys"),
                              ("TL W2 posto 61 (DHS muni, no FE)", T[(T.window == "W2") & (T.unit != "TL06U")], None, "net_w", "access_phys"),
                              ("NP W1 district 72", N[(N.window == "W1") & (N.capital == 0)], "province", "net_w", "phys_idx")]:
    d = d.assign(Net=z(d[net]), Road=z(d[road]), Dep=z(d.deprivation), lpop=np.log(d.pop_w))
    f = "active ~ Net + Road + Dep + lpop" + (f" + C({fe})" if fe else "")
    m = smf.glm(f, d, family=sm.families.Poisson()).fit(cov_type="HC1")
    se = m.bse["Net"]
    say(f"  {nm:<34} mean projects/unit {d.active.mean():.1f}; SE(Net) {se:.3f}; MDE {2.8 * se:.3f} = {(np.exp(2.8 * se) - 1) * 100:.0f}% per SD")

# ---------- candidate countries ----------
gh = requests.get("https://api.github.com/repos/AidData-WM/public_datasets/contents/geocoded", timeout=60).json()
aims = sorted({re.sub(r"(_?AIMS|_?AMP|_Geocoded.*|_geocoded.*|\.zip)", "", f["name"]).replace("_", " ").strip() for f in gh
               if re.search(r"AIMS|AMP|geocoded_v", f["name"], re.I)})
say("\nAidData country AIMS/AMP geocoded releases: " + ", ".join(aims))
cand = {"Malawi": "MW", "Nepal": "NP", "Uganda": "UG", "Senegal": "SN", "Nigeria": "NG", "Timor-Leste": "TL", "Honduras": "HN", "DRC": "CD",
        "Haiti": "HT", "Colombia": "CO", "Sierra Leone": "SL", "Burundi": "BU", "Bolivia": "BO", "Cambodia": "KH", "Iraq": "IQ", "Afghanistan": "AF"}
iso3 = {"MW": "MWI", "NP": "NPL", "UG": "UGA", "SN": "SEN", "NG": "NGA", "TL": "TLS", "HN": "HND", "CD": "COD", "HT": "HTI", "CO": "COL", "SL": "SLE",
        "BU": "BDI", "BO": "BOL", "KH": "KHM", "IQ": "IRQ", "AF": "AFG"}
dens = pd.read_csv(ROOT / "outputs/tables/ookla_country_density.csv")
rows = []
for name, c in cand.items():
    has_aims = any(name.split()[0].lower() in a.lower() for a in aims) or (name == "DRC" and any("DRC" in a for a in aims))
    surv = requests.get("https://api.dhsprogram.com/rest/dhs/data", params={"f": "json", "indicatorIds": "HC_HEFF_H_MPH", "countryIds": c,
                        "breakdown": "national", "perpage": 100, "returnFields": "SurveyId,SurveyYear,SurveyType,Value"}, timeout=120).json().get("Data", [])
    dhs = sorted({(s["SurveyYear"], s["Value"]) for s in surv if s["SurveyType"] == "DHS"})
    gps = requests.get("https://api.dhsprogram.com/rest/dhs/surveys", params={"f": "json", "countryIds": c, "surveyCharacteristicIds": "26",
                       "returnFields": "SurveyId,SurveyYear"}, timeout=120).json().get("Data", [])   # 26 = GPS collected
    oo = dens[(dens.ADM0_A3 == iso3[c]) & (dens.year == 2019)].tests_per_100k
    rows.append({"country": name, "aiddata_aims_geocoded": has_aims, "dhs_rounds_mobile": "; ".join(f"{y}:{v:.0f}%" for y, v in dhs),
                 "n_dhs_mobile": len(dhs), "dhs_gps_rounds": "; ".join(sorted({str(g["SurveyYear"]) for g in gps})), "ookla_2019_per100k": oo.iloc[0] if len(oo) else np.nan})
C = pd.DataFrame(rows).sort_values(["aiddata_aims_geocoded", "n_dhs_mobile"], ascending=False)
C.to_csv(ROOT / "outputs/tables/candidate_countries.csv", index=False)
say("\n" + C.to_string(index=False))
LOG.close()
