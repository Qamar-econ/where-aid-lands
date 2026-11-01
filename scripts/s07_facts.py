"""Descriptive facts for the report: regional shares vs population, Dili/national concentration, KOICA footprint."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data" / "processed", ROOT / "outputs" / "tables"
U = pd.read_csv(PROC / "units.csv")
PR = pd.read_csv(PROC / "oda_projects.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
X = pd.read_csv(PROC / "unit_project_links.csv")
PR = PR[(PR.start_year >= 2000) & (PR.start_year <= 2026)]
lv = AS.groupby("uid")["level"].agg(lambda s: "posto" if (s == "posto").any() else ("municipio" if (s == "municipio").any() else "national"))
meth = AS.groupby("uid").method.first()
PR["lvl"] = PR.uid.map(lv); PR["method"] = PR.uid.map(meth)
out = []
out.append(f"Unique projects 2000-2026: {len(PR):,}")
out.append("By location level:\n" + PR.lvl.value_counts().to_string())
out.append("National/HQ split: hq_default (IATI Dili office coords) = %d" % (PR.method == "hq_default").sum())
dev = PR[PR.cat.isin(["SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"])]
out.append(f"Development (non-infra) projects: {len(dev):,}; share without sub-national location: {(dev.lvl == 'national').mean():.1%}")

# regions
SOUTH = {"Covalima", "Manufahi"}
south_postos_vq = {"Uato-Lari", "Uatucarbau", "Lacluta", "Viqueque", "Barique/Natarbora", "Alas", "Fatuberliu", "Same", "Suai", "Tilomar", "Zumalai", "Fatululic", "Fatumean", "Forohem", "Maukatar", "Iliomar", "Luro"}
U["region"] = np.select([U.unit == "TL06U", U.mun_pcode == "TL12", U.unit_name.isin(south_postos_vq),
                         U.unit_name.isin(["Baucau", "Laga", "Vemasse", "Manatuto", "Laleia", "Lautém", "Lospalos", "Tutuala", "Metinaro", "Bazartete", "Liquiçá", "Maubara", "Atabae"])],
                        ["Dili (urban)", "Oecusse", "South coast", "North-coast highway"], "Interior/other")
R = U.groupby("region").agg(units=("unit", "size"), pop=("pop2022", "sum"), precise=("np_dev", "sum"), all_=("n_dev", "sum"),
                            access_proj=("n_access", "sum"), tt_wet_h=("tt_wet_min", lambda s: s.mean() / 60))
R["pop_%"] = R["pop"] / R["pop"].sum() * 100
R["precise_%"] = R.precise / R.precise.sum() * 100
R["all_%"] = R.all_ / R.all_.sum() * 100
R["precise_per10k"] = R.precise / R["pop"] * 1e4
R.to_csv(TAB / "facts_regions.csv")
out.append("Regions:\n" + R.round(1).to_string())

# KOICA footprint
k = PR[PR.donor_group == "Korea"]
out.append(f"\nKorea projects: {len(k)}; by source {k.source.value_counts().to_dict()}; by level {k.lvl.value_counts().to_dict()}")
kx = X[X.uid.isin(k.uid) & (X.precise == 1)].merge(U[["unit", "unit_name", "tt_wet_min"]], on="unit")
kk = kx.groupby(["unit_name"]).agg(n=("uid", "nunique"), tt_h=("tt_wet_min", lambda s: s.iloc[0] / 60)).sort_values("n", ascending=False)
out.append("Korea precise locations by unit:\n" + kk.round(1).to_string())
out.append(f"Korea precise: pop-weighted mean wet travel time of project units = {np.average(kk.tt_h, weights=kk.n):.1f} h vs all-unit pop mean {np.average(U.tt_wet_min/60, weights=U.pop2022):.1f} h")
# all donors comparison of travel time of precise locations
xx = X[(X.precise == 1) & (X.grp == "dev")].merge(U[["unit", "tt_wet_min"]], on="unit")
g = xx.groupby("donor_group").agg(n=("uid", "nunique"), tt_h=("tt_wet_min", lambda s: s.mean() / 60)).sort_values("tt_h")
g.to_csv(TAB / "facts_donor_traveltime.csv")
out.append("Mean wet-season travel time (h) of precise dev-project locations by donor group:\n" + g.round(2).to_string())
txt = "\n".join(out)
(ROOT / "outputs" / "facts.txt").write_text(txt, encoding="utf-8")
print(txt)
