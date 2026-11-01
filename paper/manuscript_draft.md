# Aid Follows Districts, Not People: Time-Aligned Evidence on the Subnational Allocation of Development Projects in Nepal and Timor-Leste

*Working draft v0.2 (2026-11-01). Results are provisional until project locations have been double-coded by humans (validation protocol in Appendix C).*

## Abstract
Do development projects go where roads are, where connectivity is, or where need is? We assemble every reported development project in Nepal (8,394 projects, US$37.6 billion, 2010–2025) and Timor-Leste (3,244 projects, 2010–2025). We geocode them with one transparent, precision-coded rule set. Each project is linked to its district's characteristics in the year it was planned. Accessibility is measured with a network-based index built from travel times to towns, district headquarters and hospitals, monsoon travel times, walking access, road density and terrain. Connectivity and need are measured separately from census household data. With one construct per model and province-by-year fixed effects, neither physical access nor need predicts where new projects are planned in Nepal. In Timor-Leste, connectivity predicts allocation modestly (+16% per SD) and access and need do not. The most robust regularity in both countries is different. Projects scale sub-proportionally with population, with elasticities of 0.77 in Nepal and 0.55 in Timor-Leste. The Timor-Leste estimate is robust across specifications; the Nepal estimate is less so. Small districts therefore receive more projects per person. The pattern is stable across 2010–2025 and driven by small, bilateral projects. In Timor-Leste it persists conditional on remoteness. We argue that subnational aid is organised around administrative units rather than people, a within-country analogue of the small-country bias. We also show that attaching single-year covariates to cumulative project stocks, a common design, produces spurious "connectivity effects".

## 1. Introduction

Where do development projects go within a country? Two intuitions shape both practitioner critique and academic work.
- **Spatial convenience.** Projects gravitate to places that are easy to reach. This is Chambers' (1983) "tarmac bias", in which outsiders see, and serve, the roadside rather than the remote.
- **Need.** A large literature asks whether aid reaches the poorest places and mostly finds that it does not (Briggs 2017, 2018, 2021; Öhler et al. 2019).
- A more recent possibility is that **digital connectivity** has become a new axis of convenience. Connected places are easier to monitor, report from and recruit partners in.

We test these against a more mundane alternative: projects are spread across administrative units more evenly than population is. Cross-nationally this is the well-known tendency of small countries to receive more aid per capita (Alesina and Dollar 2000). Subnationally, it implies that people in small districts receive more projects per head. That pattern is distinct from targeting either access or need.

The paper makes three contributions.

1. **Data.** We build a project-level database of every development project reported for Nepal and Timor-Leste over 2010–2025.
   - It combines AidData's geocoded aid management system data, IATI activity records, World Bank and Asian Development Bank project documents and Timor-Leste's Budget Book.
   - Every IATI and World Bank record is geocoded with the same UCDP/AidData-style rules (Strandow et al. 2011). Each rule for headquarters defaults, village co-mention and name collisions was written in response to observed false positives.
   - For the largest projects we code the date planning began, from ADB concept clearances and World Bank concept-stage Project Information Documents. The observed concept-to-start lag has a median of 0.86 years.
2. **Measurement.** Accessibility is a network-based index. Connectivity and need come from census household data and share no components with it or with each other. To avoid the instability caused by correlated regressors, each enters a separate model.
3. **Design.** Covariates are aligned with the planning year. In earlier exploratory work on the same countries, a design that attached a single-year connectivity measure to cumulative project stocks produced positive effects: +0.17 (Timor-Leste) and +0.40 (Nepal) log points per SD. Both disappeared once timing was aligned.

**Results in brief.**
- Conditional on population, neither physical access nor need predicts the planning location of new development projects in Nepal. In Timor-Leste, only connectivity does, and modestly.
- What does hold in both countries is sub-proportional scaling with population: θ = 0.77 (Nepal; p = 0.03 against θ = 1) and θ = 0.55 (Timor-Leste; p < 0.001).
- The elasticity is stable across five-year periods. It is lowest for small projects, bilateral donors, and agriculture and water. It is near one for multilateral-bank and large projects.

## 2. Related literature
**Need.**
- Briggs (2017) finds that aid within African countries does not favour poorer regions. Briggs (2018) uses gridded data to show that aid flows to richer areas.
- Briggs (2021) attributes this to donors' preference for implementability.
- Öhler et al. (2019) find limited geographic targeting of World Bank projects to the bottom 40 percent. Nunnenkamp, Öhler and Sosa Andrés (2017) find mixed evidence for need-based allocation across Indian districts.

**Politics.** Aid follows electoral incentives (Jablonski 2014) and leaders' birth regions (Dreher et al. 2019). In our framework such factors are unobserved; province-by-year fixed effects absorb their aggregate component.

**Access.** Chambers (1983) describes tarmac, roadside and dry-season biases. Quantitative tests are rare because accessibility is usually proxied by distance to the capital.

**Scale.** The small-country bias in cross-national aid is long documented (Alesina and Dollar 2000). We are not aware of evidence on its subnational analogue.

**Methods.**
- We follow the AidData/UCDP geocoding conventions (Strandow et al. 2011).
- We estimate by PPML, the standard estimator for counts with many zeros and fixed effects (Santos Silva and Tenreyro 2006). With unit-level choices it is equivalent to a conditional-logit location-choice model (Guimarães, Figueiredo and Woodward 2003).

## 3. Data

### 3.1 Projects
- **Nepal.** 8,394 projects starting 2010–2025, worth US$37.6 billion. Sources: AidData-AIMS (2010–2014), AidData's earthquake-response data (2015), IATI and the World Bank.
- **Timor-Leste.** 3,244 projects from AidData-AMP, the Budget Book's development-partner section and IATI.
- Sectors form seven development categories. Humanitarian and infrastructure projects serve as placebo families.

### 3.2 Locations
Projects are assigned to Nepal's 75 districts (2011 boundaries) or Timor-Leste's 61 postos (Dili's urban postos merged). Projects that reach districts only through central institutions are classified separately and not attributed to the capital. Multi-district projects are split equally.

In Nepal:
- 21.2% of projects (33.8% of USD) are located at district precision or better.
- 18.7% (33.5% of USD) are central-institution projects.
- 60.1% are national or unresolved (Figure 2).
- Located shares are similar across start years, and a single-source series (IATI and World Bank only) gives the same results (Table 3).

### 3.3 Planning dates
Concept clearance or concept PID dates were found for 61 ADB and World Bank projects. The concept-to-start lag has a median of 0.86 years (interquartile range 0.52–1.42). We set planning year = start year − 1 for all projects, and use documented dates where available as a robustness check.

### 3.4 Constructs (one variable each)
- **Physical access** (Nepal): the first principal component (73% of variance) of ten signed, standardised, population-weighted components:
  - travel time to the nearest town of ≥20,000 people (dry season and monsoon)
  - travel time to the district headquarters and to the nearest hospital
  - walking time to the road network
  - the share of people within 30 minutes' walk of a road
  - road density, paved share, the monsoon penalty and steep-terrain share
  - Travel times come from an OSM network with surface- and gradient-dependent speeds and WorldPop 2020. Loadings are 0.30–0.36 for all components except paved share (0.09).
- **Physical access** (Timor-Leste): PC1 of the analogous components (52% of variance).
- **Connectivity:** household internet share, from the 2011 and 2021 censuses, interpolated logit-linearly. The 2021 district values were parsed from NSO's *Housing and Households* report, Annex 11.1, and sum to the official national 37.8%. For Timor-Leste we use the posto network index from the original study.
- **Need:** a census deprivation index of no improved water, no toilet and illiteracy. It contains no communication item.
- **Correlations** in Nepal: access–population 0.85, connectivity–need −0.66, access–connectivity 0.55. Entered jointly, the access VIF is 5.9.

## 4. Empirical strategy
$$E[N_{d,t}] = \exp\big(\beta_X\,X_{d,(t)} + \theta \ln Pop_{d,t} + \gamma_{p(d),t}\big),\qquad X\in\{Access,\ Conn,\ Need\}$$

- $N_{d,t}$ is the number of new development projects planned in district *d* in year *t* (2009–2024). There are 1,200 district-years in Nepal and 976 posto-years in Timor-Leste.
- $\gamma_{p,t}$ are province (municipality)-by-year fixed effects. They absorb aggregate shocks such as the 2015 earthquake, the 2017 federal transition, changes in IATI reporting and COVID-19, so identification comes from comparisons within a province and year.
- Each construct enters its own model.
- θ is the population elasticity. θ = 1 means allocation in proportion to population, and θ < 1 means more projects per person in small units. It is estimated from the population-only model.
- Standard errors are clustered by district (posto).
- A joint model is reported in Appendix Table A1. It is not our preferred specification, because access and population are collinear in Nepal (r = 0.85).

## 5. Results

**Table 2. Main estimates** (development projects; % change per +1 SD; θ)

| | Nepal | Timor-Leste |
|---|---|---|
| Population only: θ | **0.77** (p[θ=1] = 0.034) | **0.55** (p[θ=1] < 0.001) |
| Physical access | −14% (p = 0.43) | +13% (p = 0.06) |
| Connectivity | +12% (p = 0.42) | **+16% (p = 0.002)** |
| Need (deprivation) | −10% (p = 0.34) | −9% (p = 0.11) |
| Projects (located) / unit-years | 1,279 / 1,200 | 465 / 976 |

**Access and need.**
- In Nepal, neither construct predicts allocation. The 95% confidence interval for access excludes increases larger than about +25% per SD.
- In Timor-Leste, access is borderline positive and need is negative but not significant.
- No access component alone shows a positive association in Nepal. Road density and the share within 30 minutes' walk are negative conditional on population.
- **Tarmac bias, in the sense of more projects per person in accessible places, is not present in either country.**

**Connectivity.**
- In Timor-Leste, more connected postos receive more projects (+16% per SD). The estimate is unchanged when Dili is excluded.
- In Nepal the point estimate is similar (+12%) but imprecise. It becomes significant when the Kathmandu valley is excluded (+43%, p = 0.017).
- Within districts, increases in connectivity are not followed by more projects (district and year fixed effects: −23%, p = 0.07). The cross-sectional association therefore reflects which places are connected, not responses to connectivity growth.

**Scale.** The most robust result is θ < 1 (Figure 4).
- **Nepal.** θ is 0.77 in the population-only model (SE 0.11). Across specifications it ranges from 0.50 (binary counting) to 0.90 (single-district projects; not significantly different from one, see Section 6).
- **Timor-Leste.** θ is 0.55 (0.45–0.51 conditional on each construct).
- Per person, small districts and postos receive substantially more projects.
- **Is this remoteness rather than smallness?** In Nepal, small districts are also remote (r = 0.85), and conditioning on access raises θ toward one. In Timor-Leste, where population and access are only moderately correlated (r = 0.51), θ stays at 0.51 conditional on access. **There, smallness itself drives the pattern.**

**Heterogeneity** (Figure 5).
- In Nepal, θ is lowest for agriculture (0.56), water and sanitation (0.64), bilateral donors (0.74) and projects under US$1 million (0.74). It is close to or above one for multilateral development banks (1.31) and projects over US$10 million (1.00).
- Need is never positively associated with allocation in Nepal at conventional significance. Environment and multisector projects go to less deprived districts (−49%, p = 0.003).
- Agriculture and water projects go disproportionately to less accessible districts (−46% and −33% per SD).
- In Timor-Leste, governance and large projects follow both access and connectivity.

**Stability over time** (Figure 6). θ is 0.78, 0.77 and 0.78 in 2010–14, 2015–19 and 2020–25. Access and need coefficients are near zero in every period. The federal transition did not change the basic pattern.

## 6. Robustness
**Nepal (Table 3).** The null results for access, connectivity and need hold in every specification below, except where noted:
- an IATI/World Bank-only series, removing the AidData-AIMS break in 2014
- excluding earthquake and COVID-19 projects
- restricting to single-district projects, or to projects in at most ten districts
- documented planning years
- binary counting (access becomes negative: −19%, p = 0.047)

**θ in Nepal.** The point estimate stays below one throughout (0.50–0.90), but its significance varies:
- θ < 1 is significant in the main model, the single-source series (0.78, p = 0.039), binary counting (0.50, p < 0.001) and excluding the Kathmandu valley (0.64, p = 0.004).
- It is not significant when earthquake/COVID projects are excluded (0.79, p = 0.059), in projects in ≤10 districts (0.80, p = 0.062), or in single-district projects (0.90, p = 0.52).
- Single-district projects thus scale nearly proportionally. Unit-spreading is partly a feature of multi-district programmes. The Timor-Leste estimate (0.55, p < 0.001) is more robust.

Under USD weighting, access is negative (−44%, p = 0.047) and θ ≈ 1.06. Money, as opposed to project counts, is allocated roughly in proportion to population.

**Placebos.** Humanitarian projects show θ = 0.56 and are, as expected, less concentrated in accessible districts.

**Timor-Leste.** Results are unchanged when Dili is excluded or all sectors are included.

## 7. Discussion
The evidence fits a simple account. Subnational project allocation is organised around administrative units.
- Many programmes, especially small bilateral and NGO projects, are designed to work "in N districts" or with selected local governments.
- Implementation partners and permissions are organised by district.
- Visibility and reporting reward coverage of units rather than of people.

This produces a within-country version of the small-country bias. People in sparsely populated districts receive more projects per head, while large, mostly multilateral programmes scale roughly with population.

The implications for the targeting debate are twofold.
- The absence of tarmac bias does not mean aid is pro-poor: need is unrelated to allocation in both countries.
- Per-capita comparisons that ignore unit size will mistake unit-spreading for targeting of remote areas.

## 8. Limitations
1. **Location coverage.** Only about a fifth of projects (a third of USD) can be located below the national level. Results describe located projects. A human double-coding exercise on a stratified sample of 200 projects is under way.
2. **Constant planning lag.** Planning years are start year − 1 except for 61 projects with documented dates. Results are insensitive to ±1 year shifts in earlier analyses.
3. **Connectivity interpolation.** District connectivity between census years is interpolated, so within-district changes are smooth by construction.
4. **District headquarters.** HQ coordinates were available in GeoNames for only 21 of 75 districts; the remainder use district centroids. The HQ travel-time component should be re-estimated with official locations.
5. **Unobserved factors.** Political and personal factors are unobserved and enter the error term.
6. **Scope.** Two countries.

## Figures
- Figure 1. Development projects per 100,000 people (2010–2025) and the physical-access index, Nepal ([F1_map.png](../outputs/figures/paper/F1_map.png))
- Figure 2. Location coverage by start year ([F2_coverage.png](../outputs/figures/paper/F2_coverage.png))
- Figure 3. Main estimates, each construct alone vs jointly ([F3_main.png](../outputs/figures/paper/F3_main.png))
- Figure 4. Population scaling ([F4_population_scaling.png](../outputs/figures/paper/F4_population_scaling.png))
- Figure 5. Heterogeneity, Nepal ([F5_heterogeneity.png](../outputs/figures/paper/F5_heterogeneity.png))
- Figure 6. Year-by-year coefficients, Nepal ([F6_yearly.png](../outputs/figures/paper/F6_yearly.png))

## Appendix
- **A1.** Joint model: [np3_table2_main.csv](../outputs/tables/np3_table2_main.csv) (row 4), [tl3_table_main.csv](../outputs/tables/tl3_table_main.csv)
- **A2.** Access components: [np3_access_pca.csv](../outputs/tables/np3_access_pca.csv), [np3_access_components.csv](../outputs/tables/np3_access_components.csv)
- **B.** Geocoding rules V1–V5 and planning-date codes O1/O1a/O2/O3/S1/E1: [np2_geo.py](../scripts/np2_geo.py), [np2_s04_planning.py](../scripts/np2_s04_planning.py)
- **C.** Validation sheet: [np2_validation_sheet.xlsx](../data/manual/np2_validation_sheet.xlsx); one-page project summaries: [np2_project_summaries.pdf](../outputs/project_summaries/np2_project_summaries.pdf)
- **D.** Exploratory findings not replicated (size gradient in road effects): [사전등록_규모기울기_복제.md](../사전등록_규모기울기_복제.md)

## References
- Alesina, A., and D. Dollar (2000). Who gives foreign aid to whom and why? *Journal of Economic Growth* 5(1): 33–63.
- Briggs, R. C. (2017). Does foreign aid target the poorest? *International Organization* 71(1): 187–206.
- Briggs, R. C. (2018). Poor targeting: A gridded spatial analysis of the degree to which aid reaches the poor in Africa. *World Development* 103: 133–148.
- Briggs, R. C. (2021). Why does aid not target the poorest? *International Studies Quarterly* 65(3): 739–752.
- Chambers, R. (1983). *Rural Development: Putting the Last First.* London: Longman.
- Dreher, A., A. Fuchs, R. Hodler, B. C. Parks, P. A. Raschky, and M. J. Tierney (2019). African leaders and the geography of China's foreign assistance. *Journal of Development Economics* 140: 44–71.
- Guimarães, P., O. Figueiredo, and D. Woodward (2003). A tractable approach to the firm location decision problem. *Review of Economics and Statistics* 85(1): 201–204.
- Jablonski, R. S. (2014). How aid targets votes. *World Politics* 66(2): 293–330.
- Nunnenkamp, P., H. Öhler, and M. Sosa Andrés (2017). Need, merit and politics in multilateral aid allocation: A district-level analysis of World Bank projects in India. *Review of Development Economics* 21(1): 126–156.
- Öhler, H., M. Negre, L. Smets, R. Massari, and Ž. Bogetić (2019). Putting your money where your mouth is: Geographic targeting of World Bank projects to the bottom 40 percent. *PLoS ONE* 14(6): e0218671.
- Santos Silva, J. M. C., and S. Tenreyro (2006). The log of gravity. *Review of Economics and Statistics* 88(4): 641–658.
- Strandow, D., M. Findley, D. Nielson, and J. Powell (2011). *The UCDP and AidData Codebook on Geo-referencing Aid, Version 1.1.* Uppsala University.

*(Bibliographic details to be verified against published versions before submission.)*
