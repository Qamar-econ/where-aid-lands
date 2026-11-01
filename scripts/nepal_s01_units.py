"""Nepal analysis units: 75 districts on 2011 boundaries (Nawalparasi East/West and Rukum East/West merged),
province of the larger part, census 2011 covariates (population, household ICT, water, sanitation, literacy).

Output: data/processed/nepal/districts.gpkg, districts.csv
"""
import re
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw" / "nepal", ROOT / "data" / "processed" / "nepal"
OUT.mkdir(parents=True, exist_ok=True)
UTM = 32645

a2 = gpd.read_file(RAW / "cod-ab-npl/npl_admin_boundaries.shp/npl_admin2.shp")[["adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode", "geometry"]]


def key(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


MERGE = {"nawalparasieast": "nawalparasi", "nawalparasiwest": "nawalparasi", "nawalpur": "nawalparasi", "parasi": "nawalparasi",
         "rukumeast": "rukum", "rukumwest": "rukum", "nawalparasibardaghatsusteeast": "nawalparasi", "nawalparasibardaghatsustewest": "nawalparasi"}
ALIAS = {"kavrepalanchok": "kavre", "kavrepalanchowk": "kavre", "kabhrepalanchok": "kavre", "chitawan": "chitwan", "sindhupalchowk": "sindhupalchok",
         "tanahu": "tanahun", "makawanpur": "makwanpur", "dhanusha": "dhanusa", "kapilbastu": "kapilvastu", "terhathum": "tehrathum",
         "sankhuwasabha": "sankhuwasabha", "sangkhuwasabha": "sankhuwasabha", "darchula": "darchula", "dailekh": "dailekh", "kailali": "kailali", "udayapur": "udaypur",
         "rautahat": "rautahat", "parsa": "parsa", "sindhuli": "sindhuli", "okhaldhunga": "okhaldhunga", "solukhumbu": "solukhumbu",
         "mustang": "mustang", "myagdi": "myagdi", "arghakhanchi": "arghakhanchi", "rupandehi": "rupandehi", "nuwakot": "nuwakot"}
a2["k"] = a2.adm2_name.map(key).map(lambda k: MERGE.get(k, k)).map(lambda k: ALIAS.get(k, k))
a2u = a2.to_crs(UTM); a2["area"] = a2u.area
prov = a2.sort_values("area", ascending=False).drop_duplicates("k").set_index("k")[["adm1_name", "adm1_pcode"]]
D = a2.dissolve("k", aggfunc={"adm2_name": "first", "adm2_pcode": "first"}).reset_index()
D = D.merge(prov, left_on="k", right_index=True)
D.loc[D.k == "nawalparasi", "adm2_name"] = "Nawalparasi"
D.loc[D.k == "rukum", "adm2_name"] = "Rukum"
D = D.rename(columns={"adm2_name": "district", "adm1_name": "province", "adm1_pcode": "prov_pcode", "adm2_pcode": "pcode"})
D["area_km2"] = D.to_crs(UTM).area / 1e6
print("districts:", len(D))


def sheet(name, header_row, name_col=2):
    df = pd.read_excel(RAW / "census2011/2011 Household Facilities.xlsx", name, header=None)
    hdr = [str(x).strip() for x in df.iloc[header_row]]
    body = df.iloc[header_row + 1:].copy(); body.columns = hdr
    body = body[pd.to_numeric(body.iloc[:, 0], errors="coerce").notna()]
    body["k"] = body.iloc[:, name_col].map(key).map(lambda k: ALIAS.get(k, k))
    return body


fac = sheet("Hhld_Facilities", 4)
fac = fac.rename(columns={"Total": "hh"})
for c in ["Radio", "Television", "Cable Television", "Computer", "Internet", "Telephone", "Mobile  Phone", "Mobile Phone", "Motorcycle", "Refrigerator"]:
    if c in fac.columns:
        fac[c] = pd.to_numeric(fac[c], errors="coerce")
mob = "Mobile  Phone" if "Mobile  Phone" in fac.columns else "Mobile Phone"
C = pd.DataFrame({"k": fac.k, "hh2011": pd.to_numeric(fac.hh, errors="coerce"),
                  "mobile_pct": fac[mob] / pd.to_numeric(fac.hh) * 100, "internet_pct": fac["Internet"] / pd.to_numeric(fac.hh) * 100,
                  "phone_pct": fac["Telephone"] / pd.to_numeric(fac.hh) * 100, "computer_pct": fac["Computer"] / pd.to_numeric(fac.hh) * 100,
                  "tv_pct": fac["Television"] / pd.to_numeric(fac.hh) * 100})
w = sheet("Drinking water", 3)
C = C.merge(pd.DataFrame({"k": w.k, "water_piped": pd.to_numeric(w["Tap/Piped"]) / pd.to_numeric(w["Total"]) * 100,
                          "water_improved": (pd.to_numeric(w["Tap/Piped"]) + pd.to_numeric(w["Tubewell/Hand Pump"]) + pd.to_numeric(w["Covered well/Kuwa"])) / pd.to_numeric(w["Total"]) * 100}), on="k")
t = sheet("household using toilet", 3)
C = C.merge(pd.DataFrame({"k": t.k, "no_toilet": pd.to_numeric(t["Without Toilet"]) / pd.to_numeric(t["Total"]) * 100}), on="k")
lit = pd.read_excel(RAW / "census2011/2011 Household Facilities.xlsx", "Literacy_status_abv5Yrs", header=None).iloc[4:]
lit = lit[pd.to_numeric(lit.iloc[:, 0], errors="coerce").notna()]
C = C.merge(pd.DataFrame({"k": lit.iloc[:, 1].map(key).map(lambda k: ALIAS.get(k, k)),
                          "literacy": pd.to_numeric(lit.iloc[:, 7]) / pd.to_numeric(lit.iloc[:, 4]) * 100}), on="k")
pop = pd.read_excel(RAW / "census2011/DistricWise Population 2011_v1.xlsx", header=None).iloc[4:]
pop = pop[pd.to_numeric(pop.iloc[:, 0], errors="coerce").notna()]
C = C.merge(pd.DataFrame({"k": pop.iloc[:, 2].map(key).map(lambda k: ALIAS.get(k, k)), "pop2011": pd.to_numeric(pop.iloc[:, 4])}), on="k")
miss = set(D.k) - set(C.k)
print("census rows", len(C), "unmatched districts:", miss, "| unmatched census:", set(C.k) - set(D.k))
D = D.merge(C, on="k", how="left")
# 2021-based population (COD-PS 2023 projection from census 2021) for growth
ps = pd.read_excel(RAW / "cod-ps-npl/npl_admpop_2023.xlsx", sheet_name=None)
adm2 = next(v for k, v in ps.items() if "adm2" in k.lower())
adm2["k"] = adm2.filter(like="ADM2_EN").iloc[:, 0].map(key).map(lambda k: MERGE.get(k, k)).map(lambda k: ALIAS.get(k, k))
tcol = [c for c in adm2.columns if c.upper() in ("T_TL", "TOTAL", "T_TOTAL")][0]
D = D.merge(adm2.groupby("k")[tcol].sum().rename("pop2023"), on="k", how="left")
D["g_11_23"] = (D.pop2023 / D.pop2011) ** (1 / 12) - 1
D["capital"] = D.k.isin(["kathmandu", "lalitpur", "bhaktapur"]).astype(int)
D["density"] = D.pop2011 / D.area_km2
dep = pd.DataFrame({"no_water": 100 - D.water_improved, "no_toilet": D.no_toilet, "illit": 100 - D.literacy})
Zd = (dep - dep.mean()) / dep.std()
D["deprivation"] = Zd.mean(axis=1) / Zd.mean(axis=1).std()
D["net_census"] = (((D.mobile_pct - D.mobile_pct.mean()) / D.mobile_pct.std()) + ((np.log1p(D.internet_pct) - np.log1p(D.internet_pct).mean()) / np.log1p(D.internet_pct).std())) / 2
D.to_file(OUT / "districts.gpkg", driver="GPKG")
D.drop(columns="geometry").to_csv(OUT / "districts.csv", index=False, encoding="utf-8-sig")
print(D[["district", "province", "pop2011", "mobile_pct", "internet_pct", "water_piped", "no_toilet", "literacy", "deprivation"]]
      .sort_values("internet_pct").round(1).to_string())
