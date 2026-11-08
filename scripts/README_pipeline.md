# Where Aid Lands: how to reproduce the Cambodia results

Python 3.11+. Packages: numpy, pandas, scipy, statsmodels, geopandas, rasterio, osmium, pyarrow, requests, matplotlib.

There are two ways in.

## A. Check the models (10 minutes, two files)

1. Download `kh_analysis.csv`, `kh_access.csv`, `kh_published.csv`, `kh_replicate.py`. Put them in one folder.
2. Run `python kh_replicate.py`.

The script rebuilds the physical-access index from its ten components and re-estimates every headline model from the data.
It then sets each of your numbers beside the published one. Effects and clustered or Conley p-values match to rounding.
Randomisation-inference p-values come from random draws, so they differ within the Monte Carlo error the script prints.

## B. Rebuild everything from the raw sources

Folder layout the scripts expect:

```
research/
  scripts/                  all kh_*.py files
  data/raw/cambodia/        raw downloads (step 1)
  data/processed/cambodia/  built tables (steps 2-6)
  outputs/tables/           result tables (step 7)
```

Run the scripts from `research/scripts`, in this order:

| Step | Script | Builds | Compare with |
|---|---|---|---|
| 1 | `kh_fetch.py` | WorldPop, night lights, Copernicus DEM, OSM 2017 and current, GeoNames, Ookla | (raw files) |
| 1b | download by hand | GODAD Cambodia extract → `data/raw/cambodia/godad_khm.csv` (or use the published copy) | `godad_khm.csv` |
| 2 | `kh_s00_boundaries.py` | district units (IPUMS GEO2_KH) → `kh_units.gpkg` | `kh_units.geojson` |
| 3 | `kh_s01_projects.py` | aid activities and their districts → `kh_projects.csv`, `kh_assign.csv` | same names |
| 4 | `kh_s02_covariates.py` | population, night lights, Ookla per district → `kh_covariates.csv` | same name |
| 5 | `kh_s03_access.py` | travel times and the physical-access index → `kh_access.csv` (slow: about 20 minutes) | same name |
| 6 | `kh_s10_rwi.py` | Meta wealth index per district → `kh_rwi.csv` | same name |
| 7 | `kh_s04_models.py` | main estimates with all inference → `outputs/tables/kh_main.csv` | `kh_published.csv` |
| 8 | `kh_s26_analysis_dataset.py` | one table with every model input → `kh_analysis.csv` | same name |
| 9 | `kh_s24_conservative_p.py`, `kh_s25_joint_test.py`, `kh_s27_need_conservative.py`, `kh_s07_shapley.py`, `kh_s18_wild_bootstrap.py` | joint models, joint test, poverty, decomposition, wild bootstrap | site tables |
| 10 | `kh_s08_extensions.py`, `kh_s09_geobias.py`, `kh_s16_access_sensitivity.py`, `kh_s19`–`kh_s23` | event study, geocoding bounds, index sensitivity and audits | site tables |

### The main analysis (spec 9-27, 9-28)

| Script | What it does | Output |
|---|---|---|
| `kh_s38_geocode_extract.py` | lists projects whose location is only a province or town name, for hand classification | `kh_geocode_flagged.csv`, `kh_geocode_flagged_units.csv` |
| (by hand) | each listed project read and classed TOWN / PROVINCE / UNCLEAR / NATIONAL, with a one-line reason | `kh_geocode_classified.csv` |
| `kh_main.py [draws]` | the main analysis: travel time and poverty, within provinces, randomisation inference; breakdown by district type; robustness; donor groups; census tests if `kh_census.csv` exists | `outputs/tables/kh_main_simple.csv`, `.txt` |
| `kh_s40_hc3_panel.py` | pre-registered H-C3: 2008-2019 change in phone ownership vs change in projects, district fixed effects | `kh_hc3.csv`, `.txt` |
| `kh_s41_bridges.py` | spec 9-32: with/without-bridge travel times on the 2017 network (bridge = ferry of F minutes) and a district event study around eight 2008-2015 bridges | `kh_bridges*.csv`, `.txt` |
| `kh_s42_capital.py [draws]` | spec 9-33: travel time to Phnom Penh vs to the nearest town, zone fixed effects, cross-validated comparison | `kh_capital.csv`, `kh_capital_compare.csv` |
| `kh_s43_province_infra.py` | spec 9-34: province-level event study around bridge and road works (counterfactual travel time, exact and sampled randomisation inference); cell-level minutes saved for the map | `kh_pinfra*.csv`, `.txt` |
| `kh_s44_verify_cases.py` | independent checks of the case studies: engine reproduces kh_access, ferry substitution, outcome by hand, leave-one-out and window | `kh_cases_verify.csv`, `.txt` |
| `kh_s45_persistence.py [draws]` | spec 9-35: do districts with 2005-2012 projects get more in 2017-2020, holding travel time fixed | `kh_persistence.csv`, `.txt` |
| `kh_s46_designs.py [draws]` | spec 9-36: neighbour pairs, straight-line distance vs road detour, field vs institutional projects, EU/US vs China | `kh_designs.csv`, `.txt` |
| `kh_s47_verify_designs.py` | independent checks of kh_s46: fast PPML = statsmodels GLM, pair adjacency, distance identity, label totals | `kh_designs_verify.csv` |
| `kh_s48_need_match.py [draws]` | spec 9-37: do projects within a district go to the sector with the larger census need | `kh_needmatch.csv`, `.txt` |
| `kh_figures.py` | figures F1-F5 in English and Korean, from the same data step and results table | `outputs/figures/cambodia/F*.pdf/.png` |
| `kh_census_build.py extract.csv` | once the IPUMS extract arrives: deprivation index and phone ownership by district, classifying answer codes from the codebook labels | `kh_census.csv` |
| `kh_check.py` | 104 automated checks from the sample to every number on the site; exit code 1 on any failure | (console) |
| `kh_run_all.sh` | reruns every step downstream of the sample, in dependency order | `outputs/logs/` |

**The census data are not redistributed.** IPUMS terms forbid sharing the microdata, and the district tables built from it (`kh_census.csv`) are kept local as well. To reproduce the census results, request your own IPUMS International extract (Cambodia 2008 and 2019; variables listed in the pre-registration, amendment 2) and run `kh_census_build.py` on it.

When the census extract arrives: put it and its DDI codebook in `data/raw/cambodia/ipums/`, run `kh_census_build.py`, then `kh_run_all.sh` and `kh_check.py`.

After any step, compare your table with the published one:

```
python kh_compare.py data/processed/cambodia/kh_access.csv path/to/downloaded/kh_access.csv
```

Raw files are not redistributed; `kh_fetch.py` downloads them from their publishers.
The IPUMS boundary file and the GODAD extract need the publishers' terms to be accepted.
OpenStreetMap snapshots change over time, so use the dated file `cambodia-170101.osm.pbf` for the main index.
