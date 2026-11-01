# Where Aid Lands: Roads, Connectivity, or Need? Evidence from Cambodia

Tae Yoon Moon (University of Seoul) · Claude Opus 5.5 (Anthropic)
Working results, November 2026. Census (IPUMS) indicators are pending.

Do development projects go where people are poorest, or where they are easiest to reach? We place geocoded aid projects
starting 2017–2020 (GODAD) onto Cambodia's 141 harmonised districts and compare road access, mobile connectivity,
economic activity (night lights) and need (Meta Relative Wealth Index), one construct per model, with PPML and province
fixed effects.

**Website:** the project page is published from `site/` with GitHub Pages (see *Settings → Pages* for the address).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/kh_*.py` | Cambodia pipeline, run in order `kh_fetch` → `kh_s00` … `kh_s15` |
| `scripts/` (other) | Earlier Timor-Leste and Nepal analyses that shaped the design |
| `data/processed/` | District tables and project assignments built by the scripts |
| `outputs/tables/` | Every result table (`kh_main.csv`, `kh_dedup_main.csv`, `kh_access2017_main.csv`, …) |
| `site/` | The project website (static; deployed by `.github/workflows/pages.yml`) |
| `모형명세서_캄보디아.md` | Model specification, written before estimation; later amendments are labelled (9-1 … 9-6) |
| `보고서_*.md`, `paper/` | Results reports, literature review, geocoding-bias review (Korean) |

## Reproduce

Raw downloads are not stored here (3.7 GB). `scripts/kh_fetch.py` downloads WorldPop, night lights, Copernicus DEM,
OpenStreetMap (latest and the 1 January 2017 extract), GeoNames and Ookla. Two inputs must be obtained by hand:

- GODAD project-level data (Bomprezzi et al. 2025): save the Cambodia extract as `data/raw/cambodia/godad_khm.csv`
- IPUMS GEO2_KH boundaries (IPUMS International, free registration)

Then run the scripts in order with Python 3.11+ (pandas, geopandas, statsmodels, rasterio, osmium, scipy, matplotlib).
`kh_s03_access.py` builds the road-access index; set `KH_OSM_SNAPSHOT=170101` to build it on the pre-period network.

## Use of AI

Research question, country choice, hypotheses and design rules are the author's. Data scripts, estimation, figures and
the website were produced with Claude Opus 5.5 in Claude Code at the author's direction; outside reviews from Gemini and
ChatGPT were checked against the data before adoption. Under COPE and ICMJE guidance an AI system cannot be an author of
a submitted paper; there, Claude's role will be described in the methods, and the human author is responsible for the content.

## Sources

GODAD (Bomprezzi et al. 2025), IPUMS International, Ookla Open Data (CC BY-NC-SA 4.0; only district aggregates are
redistributed), OpenStreetMap contributors (ODbL), Copernicus GLO-90, WorldPop, Meta Relative Wealth Index (Chi et al. 2022),
harmonised DMSP–VIIRS night lights (Li et al. 2020), OECD DAC.
