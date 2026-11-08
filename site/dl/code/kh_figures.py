"""Publication figures for the main analysis (spec 9-27, 9-28), in English and Korean. Reads only kh_main.py's data step and
its results table, so every number drawn is a number reported. Each figure is saved as PDF (vector) and PNG; Korean versions
carry the suffix _ko.

F1  map: travel time to the nearest town by district, with projects per district
F2  projects per 100,000 people by within-province travel-time decile, 95% CI from a province bootstrap, with the PPML fit
F3  where the gradient comes from: projects per 100,000 people in Phnom Penh, provincial towns and rural districts
    (rural split by travel-time tercile), 95% CI from a province bootstrap
F4  the travel-time estimate across robustness checks and donor groups, 95% CI, randomisation-inference p
F5  remote districts are also the poorest: travel time against poverty, circle area = projects
Output: outputs/figures/cambodia/F1-F5[_ko].pdf/.png
"""
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
src = (HERE / "kh_main.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_main.py")}
exec(src[:src.index("# ---------------- estimation")], ns)               # same districts, projects and reclassification as kh_main.py
DF, MAIN, counts, PROC, FIG, ROOT = ns["DF"], ns["MAIN"], ns["counts"], ns["PROC"], ns["FIG"], ns["ROOT"]
R = pd.read_csv(ROOT / "outputs/tables/kh_main_simple.csv")
INK, INK2, MUTED = "#24211b", "#47423a", "#8a8478"
TEAL, RUST, SAND = "#0b6b6b", "#a8461f", "#e9e5dc"
BOOT = 2000
MINUS = "−"

d = counts(MAIN)
CEN = PROC / "kh_census.csv"
if CEN.exists():                                   # census measures, standardised over the same 141 districts as kh_main.py
    c19 = pd.read_csv(CEN).query("year == 2019").set_index("unit")
    d["phone_share"] = d.unit.map(c19.phone_share)
    dep = d.unit.map(c19.deprivation); d["deprivation"] = (dep - dep.mean()) / dep.std()
d["type"] = np.where(d.capital, "capital", np.where(d.ADMIN_NAME.str.contains("Krong"), "town", "rural"))
d["rel"] = d.reach - d.groupby("prov").reach.transform("mean")
provs = d.prov.unique()


def boot_rates(groupcol, order, seed, within=False):
    """Projects per 100,000 people by group, with a 95% interval from resampling whole provinces, or (within=True)
    districts within each group, for groups that sit inside a single province (Phnom Penh)."""
    rng = np.random.default_rng(seed)
    rate = lambda x: (1e5 * x.groupby(groupcol).N.sum() / x.groupby(groupcol).pop_2019.sum()).reindex(order)
    obs = rate(d); B = []
    by = {p: d[d.prov == p] for p in provs}; grp = {g: d[d[groupcol] == g] for g in order}
    for _ in range(BOOT):
        if within:
            B.append(rate(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in grp.values()])).values)
        else:
            B.append(rate(pd.concat([by[p] for p in rng.choice(provs, len(provs), replace=True)])).values)
    lo, hi = np.nanpercentile(np.array(B, dtype=float), [2.5, 97.5], axis=0)
    return obs.values, lo, hi


# deciles of within-province travel time (F2) and district groups (F3), computed once for both languages
K = 10; d["bin"] = pd.qcut(d.rel, K, labels=False)
f2_obs, f2_lo, f2_hi = boot_rates("bin", range(K), 7)
f2_x = d.groupby("bin").rel.mean().values
fit2 = smf.glm("N ~ rel + lnpop + C(prov)", data=d, family=sm.families.Poisson()).fit(maxiter=300)
grid = np.linspace(np.percentile(d.rel, 2), np.percentile(d.rel, 98), 100)   # no extrapolation beyond the data
base = np.mean(fit2.predict(d.assign(rel=0)) / d.pop_2019)
f2_fit = 1e5 * base * np.exp(fit2.params["rel"] * grid)
rural = d.type == "rural"
G7 = {}
if "phone_share" in d:
    d["lights_z"] = (lambda v: (v - v.mean()) / v.std())(np.log1p(d.ntl_pc_2015 * 1000))
    d["phone_z"] = (d.phone_share - d.phone_share.mean()) / d.phone_share.std()
    for col in ["phone_z", "deprivation", "lights_z"]:
        rel = d[col] - d.groupby("prov")[col].transform("mean"); d["bin"] = pd.qcut(rel.rank(method="first"), K, labels=False)
        o_, l_, h_ = boot_rates("bin", range(K), 7)
        m_ = smf.glm(f"N ~ rel_ + lnpop + C(prov)", data=d.assign(rel_=rel), family=sm.families.Poisson()).fit(maxiter=300)
        g_ = np.linspace(np.percentile(rel, 2), np.percentile(rel, 98), 100)
        b_ = np.mean(m_.predict(d.assign(rel_=0)) / d.pop_2019)
        G7[col] = (rel.groupby(d["bin"]).mean().values, o_, l_, h_, g_, 1e5 * b_ * np.exp(m_.params["rel_"] * g_), 100 * (np.exp(m_.params["rel_"]) - 1))
IMP = ROOT / "outputs/tables/kh_main_importance.csv"
d["grp"] = d.type
d.loc[rural, "grp"] = "rural" + pd.qcut(d.loc[rural, "tt_town_dry"], 3, labels=["1", "2", "3"]).astype(str)
G3 = ["capital", "town", "rural1", "rural2", "rural3"]
f3_obs, f3_lo, f3_hi = boot_rates("grp", G3, 11, within=True)
f3_tt = d.groupby("grp").tt_town_dry.median().reindex(G3).values
f3_n = d.groupby("grp").size().reindex(G3).values

TX = {
    "en": dict(font="DejaVu Sans", sfx="",
               f1t="Projects and travel time to the nearest town, by district", f1cb="Travel time to the nearest town of 20,000+ (minutes, log scale)",
               f1leg="Projects 2017–2020", proj=lambda n: f"{n} project{'s' if n > 1 else ''}",
               f2t="Districts farther from a town than others in their province receive fewer projects",
               f2x="Travel time to the nearest town, relative to the province average", f2y="Projects per 100,000 people",
               f2a="Decile of districts (95% CI, province bootstrap)", f2b=lambda p: f"PPML fit: {p:+.0f}% per doubling of travel time".replace("-", MINUS),
               f2ticks=["¼×", "½×", "province\naverage", "2×", "4×"],
               f3t="Projects concentrate in Phnom Penh and provincial towns", f3y="Projects per 100,000 people",
               f3lab=["Phnom Penh\n(4 khans)", "Provincial\ntowns", "Rural:\nclosest third", "Rural:\nmiddle third", "Rural:\nfarthest third"],
               f3sub=lambda n, t: f"{n} districts\n{t:.0f} min",               f4t="Physical access: the travel-time estimate across checks", f4x="% change in projects (95% CI, province-clustered); teal: CI excludes 0 and randomisation-inference p < 0.05",
               f4g=["Main estimate", "Robustness", "Donor groups", "Subsample"],
               f4lab={"physical access alone": "Main estimate (all 141 districts)", "R1a geocodes as published (no reclassification)": "Geocodes as published",
                      "R1b worst case: province-level geocodes to remotest district": "Reclassified projects in remotest district",
                      "R2a province-only projects added at remotest district": "+ province-only projects, remotest district",
                      "R2b province-only projects added at best-reached district": "+ province-only projects, best-reached district",
                      "R3 Phnom Penh excluded": "Phnom Penh excluded", "R4 Weiss et al. (2018) travel time to cities": "Weiss et al. (2018) travel-time map",
                      "rural districts: physical access alone": "Rural districts only (112)",
                      "donors: EU/US": "EU and US donors", "donors: World Bank": "World Bank", "donors: China": "China"},
               f5t="Remote districts are also the poorest", f5x="Travel time to the nearest town (minutes, log scale)", f5y="Poverty (−Meta wealth index, SD)",
               f5leg="Projects",
               f4c_t="Three constructs: each alone, with checks, and in rural districts only",
               f4c_x="% change in projects (95% CI, province-clustered); teal: CI excludes 0 and randomisation p < 0.05",
               f4c_g=["Physical access (travel time ×2)", "Digital access (+1 SD)", "Poverty and economic activity (+1 SD)"],
               f4c={"physical access alone": "Main estimate", "pair: physical access + digital access (r = -0.62)": "Holding digital access fixed",
                    "R4 Weiss et al. (2018) travel time to cities": "Weiss et al. (2018) map", "rural districts: physical access alone": "Rural districts only",
                    "digital access alone": "Main estimate: phone ownership 2019", "digital check: phone ownership 2008": "Phone ownership 2008 (before projects)",
                    "digital check: Ookla mobile speed": "Ookla mobile speed", "pair:phone": "Holding physical access fixed",
                    "rural districts: digital access alone": "Rural districts only",
                    "poverty alone": "Main estimate: census deprivation 2019", "poverty check: 2008 index": "Census deprivation 2008",
                    "poverty check: without water (piped only)": "Deprivation without water", "poverty check: Meta RWI": "Meta wealth index (negative)", "need check: night lights per person 2015": "Night lights per person 2015 (richer = more)",
                    "rural districts: poverty alone": "Rural districts only"},
               f5c_t="Remote districts are less connected and poorer", f5c_y1="Households with a mobile phone, 2019 (%)",
               f5c_y2="Census deprivation, 2019 (SD)",
               f7_t="Projects by decile of each measure, within province",
               f7_p=["Digital access: phone ownership 2019", "Poverty: census deprivation 2019", "Economic activity: night lights 2015"],
               f7_x=["phone ownership vs province mean (SD)", "deprivation vs province mean (SD)", "night lights vs province mean (SD)"],
               f7_y="Projects per 100,000 people",
               f8_t="Which matters more?", f8_a="Share of explained variation (Shapley)", f8_b="Out-of-sample gain in fit (%, 5-fold × 20)",
               f8_lab={"physical access": "Physical access", "digital access": "Digital access", "poverty": "Poverty", "night lights": "Night lights"}),
    "ko": dict(font="Malgun Gothic", sfx="_ko",
               f1t="군별 원조 사업과 가장 가까운 도시까지의 이동시간", f1cb="인구 2만 명 이상 도시까지 이동시간 (분, 로그 눈금)",
               f1leg="사업 2017–2020", proj=lambda n: f"사업 {n}개",
               f2t="같은 주의 다른 군보다 도시에서 먼 군일수록 사업이 적다",
               f2x="가장 가까운 도시까지 이동시간 (주 평균 대비)", f2y="인구 10만 명당 사업 수",
               f2a="군 10분위 (95% 신뢰구간, 주 단위 부트스트랩)", f2b=lambda p: f"PPML 추정: 이동시간 두 배일 때 {p:+.0f}%".replace("-", MINUS),
               f2ticks=["¼배", "½배", "주\n평균", "2배", "4배"],
               f3t="사업은 프놈펜과 도청 소재지에 몰린다", f3y="인구 10만 명당 사업 수",
               f3lab=["프놈펜\n(4개 칸)", "도청\n소재지", "농촌:\n가까운 1/3", "농촌:\n중간 1/3", "농촌:\n먼 1/3"],
               f3sub=lambda n, t: f"{n}개 군\n{t:.0f}분",               f4t="물리적 접근성: 이동시간 추정치의 점검별 비교", f4x="사업 수 변화 % (95% 신뢰구간, 주 군집); 청록: 구간이 0을 포함하지 않고 무작위화 추론 p < 0.05",
               f4g=["주 추정치", "견고성 점검", "공여자 그룹", "하위 표본"],
               f4lab={"physical access alone": "주 추정치 (141개 군 전체)", "R1a geocodes as published (no reclassification)": "지오코딩 원자료 그대로",
                      "R1b worst case: province-level geocodes to remotest district": "재분류 사업을 가장 외진 군에",
                      "R2a province-only projects added at remotest district": "+ 주 단위 사업, 가장 외진 군에",
                      "R2b province-only projects added at best-reached district": "+ 주 단위 사업, 가장 가까운 군에",
                      "R3 Phnom Penh excluded": "프놈펜 제외", "R4 Weiss et al. (2018) travel time to cities": "Weiss et al. (2018) 이동시간 지도",
                      "rural districts: physical access alone": "농촌 군만 (112개)",
                      "donors: EU/US": "유럽·미국", "donors: World Bank": "세계은행", "donors: China": "중국"},
               f5t="외진 군이 가장 가난한 군이기도 하다", f5x="가장 가까운 도시까지 이동시간 (분, 로그 눈금)", f5y="빈곤 (−Meta 부 지수, 표준편차)",
               f5leg="사업 수",
               f4c_t="세 구성개념: 단독 추정, 점검 측정치, 농촌 군만",
               f4c_x="사업 수 변화 % (95% 신뢰구간, 주 군집); 청록: 구간이 0을 포함하지 않고 무작위화 p < 0.05",
               f4c_g=["물리적 접근성 (이동시간 ×2)", "디지털 접근성 (+1SD)", "빈곤과 경제활동 (+1SD)"],
               f4c={"physical access alone": "주 추정치", "pair: physical access + digital access (r = -0.62)": "디지털 접근성 고정",
                    "R4 Weiss et al. (2018) travel time to cities": "Weiss et al. (2018) 지도", "rural districts: physical access alone": "농촌 군만",
                    "digital access alone": "주 추정치: 2019 휴대전화 보유율", "digital check: phone ownership 2008": "2008 보유율 (사업 이전)",
                    "digital check: Ookla mobile speed": "Ookla 모바일 속도", "pair:phone": "물리적 접근성 고정",
                    "rural districts: digital access alone": "농촌 군만",
                    "poverty alone": "주 추정치: 2019 센서스 박탈", "poverty check: 2008 index": "2008 센서스 박탈",
                    "poverty check: without water (piped only)": "식수 제외 박탈 지수", "poverty check: Meta RWI": "Meta 부 지수 (음수)", "need check: night lights per person 2015": "2015 1인당 야간 불빛 (밝을수록 부유)",
                    "rural districts: poverty alone": "농촌 군만"},
               f5c_t="외진 군일수록 연결이 약하고 더 가난하다", f5c_y1="휴대전화 보유 가구, 2019 (%)",
               f5c_y2="센서스 박탈 지수, 2019 (표준편차)",
               f7_t="측정치별 10분위에 따른 사업 수 (주 안 비교)",
               f7_p=["디지털 접근성: 2019 휴대전화 보유율", "빈곤: 2019 센서스 박탈", "경제활동: 2015 야간 불빛"],
               f7_x=["주 평균 대비 휴대전화 보유율 (SD)", "주 평균 대비 박탈 (SD)", "주 평균 대비 야간 불빛 (SD)"],
               f7_y="인구 10만 명당 사업 수",
               f8_t="무엇이 더 중요한가?", f8_a="설명력 배분 (Shapley)", f8_b="표본 밖 적합도 개선 (%, 5겹 × 20회)",
               f8_lab={"physical access": "물리적 접근성", "digital access": "디지털 접근성", "poverty": "빈곤", "night lights": "야간 불빛"}),
}

U = gpd.read_file(PROC / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"}); U["unit"] = U.unit.astype("int64")
U = U.merge(d[["unit", "tt_town_dry", "N"]], on="unit", how="left")
cen = U.to_crs(32648).geometry.centroid.to_crs(4326); U["px"], U["py"] = cen.x.values, cen.y.values   # not "cx": GeoDataFrame.cx is the coordinate indexer
PROV = U.dissolve("PARENT")

for lang, t in TX.items():
    plt.rcParams.update({"font.family": [t["font"], "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300, "axes.unicode_minus": True})

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"{name}{t['sfx']}.{ext}")
        plt.close(fig)

    # F1 map
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    norm = LogNorm(vmin=10, vmax=400)
    U[U.tt_town_dry.notna()].plot(column="tt_town_dry", ax=ax, cmap="YlOrBr", norm=norm, edgecolor="white", linewidth=.25)
    U[U.tt_town_dry.isna()].plot(ax=ax, color=SAND, edgecolor="white", linewidth=.25)
    PROV.boundary.plot(ax=ax, color=INK2, linewidth=.6)
    m = U[U.N > 0]
    ax.scatter(m["px"], m["py"], s=m.N * 9, facecolor=TEAL, edgecolor="white", linewidth=.5, alpha=.85, zorder=3)
    for n in (1, 5, 15):
        ax.scatter([], [], s=n * 9, facecolor=TEAL, edgecolor="white", label=t["proj"](n))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, .02), ncol=3, title=t["f1leg"], title_fontsize=8, fontsize=8, columnspacing=2.2, handletextpad=.8)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="YlOrBr", norm=norm), ax=ax, shrink=.55, pad=.01)
    cb.set_label(t["f1cb"], fontsize=8); cb.set_ticks([10, 30, 100, 300]); cb.set_ticklabels(["10", "30", "100", "300"])
    ax.set_axis_off(); ax.set_title(t["f1t"])
    save(fig, "F1_map")

    # F2 gradient
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.vlines(f2_x, f2_lo, f2_hi, color=MUTED, lw=1.2)
    ax.plot(f2_x, f2_obs, "o", color=TEAL, ms=5.5, zorder=3, label=t["f2a"])
    ax.plot(grid, f2_fit, color=RUST, lw=1.6, label=t["f2b"](100 * (np.exp(fit2.params["rel"]) - 1)))
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_xticklabels(t["f2ticks"])
    ax.set_xlabel(t["f2x"]); ax.set_ylabel(t["f2y"]); ax.set_ylim(bottom=0); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(t["f2t"])
    save(fig, "F2_reach_gradient")

    # F3 district groups
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(G3)); cols = [RUST, RUST, TEAL, TEAL, TEAL]
    ax.bar(x, f3_obs, color=cols, width=.62, alpha=.9)
    ax.vlines(x, f3_lo, f3_hi, color=INK2, lw=1.1)
    for i, v in enumerate(f3_obs):
        ax.text(i + .33, v, f"{v:.2f}", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{a}\n\n{t['f3sub'](n, tt)}" for a, n, tt in zip(t["f3lab"], f3_n, f3_tt)], fontsize=8)
    ax.set_ylabel(t["f3y"]); ax.set_title(t["f3t"])
    save(fig, "F3_district_types")

    # F4 coefficients
    keys = list(t["f4lab"])
    F = R[R.model.isin(keys) & R["var"].isin(["reach", "reach_weiss"])].copy()
    F["o"] = F.model.map({k: i for i, k in enumerate(keys)}); F = F.sort_values("o")
    breaks = {1: t["f4g"][1], keys.index("rural districts: physical access alone"): t["f4g"][3], keys.index("donors: EU/US"): t["f4g"][2]}
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    y, ypos, heads = 0, [], []
    for row in F.itertuples():
        if row.o in breaks:
            y += .6; heads.append((y, breaks[row.o])); y += .8
        ypos.append(y); y += 1
    ax.axvline(0, color=INK2, lw=.8)
    for yy, row in zip(ypos, F.itertuples()):
        c = TEAL if (row.p_ri < .05 and (row.ci_hi < 0 or row.ci_lo > 0)) else MUTED   # interval and RI must agree
        ax.plot([row.ci_lo, row.ci_hi], [yy, yy], color=c, lw=1.7, solid_capstyle="butt")
        ax.plot(row.pct, yy, "D" if row.o == 0 else "o", color=c, ms=6 if row.o == 0 else 5)
        ax.text(1.01, yy, f"{row.pct:+.0f}%  p {row.p_ri:.3f}".replace("-", MINUS), transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    for yy, h in heads:
        ax.text(0, yy, h, transform=ax.get_yaxis_transform(), fontsize=8.5, fontweight="bold", color=INK, va="center", ha="right")
    ax.set_yticks(ypos); ax.set_yticklabels([t["f4lab"][k] for k in F.model]); ax.invert_yaxis()
    ax.set_xlabel(t["f4x"]); ax.set_title(t["f4t"])
    save(fig, "F6_robustness")


    # F4 three constructs
    pair = "pair: physical access + digital access (r = -0.62)"
    spec = [(0, [("physical access alone", "reach"), (pair, "reach"), ("R4 Weiss et al. (2018) travel time to cities", "reach_weiss"), ("rural districts: physical access alone", "reach")]),
            (1, [("digital access alone", "phone"), ("digital check: phone ownership 2008", "phone_2008"), ("digital check: Ookla mobile speed", "speed"),
                 (pair, "phone"), ("rural districts: digital access alone", "phone")]),
            (2, [("poverty alone", "deprivation"), ("poverty check: 2008 index", "dep_2008"), ("poverty check: without water (piped only)", "dep_nowater"),
                 ("poverty check: Meta RWI", "poverty"), ("need check: night lights per person 2015", "lights"), ("rural districts: poverty alone", "deprivation")])]
    if all(((R.model == m) & (R["var"] == v)).any() for _, rows_ in spec for m, v in rows_):
        fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.8), gridspec_kw={"height_ratios": [4, 5, 6], "hspace": .55})
        for (g, rows_), ax in zip(spec, axes):
            sub = [R[(R.model == m) & (R["var"] == v)].iloc[0] for m, v in rows_]
            ax.axvline(0, color=INK2, lw=.8)
            for i, r_ in enumerate(sub):
                c = TEAL if (r_.p_ri < .05 and (r_.ci_hi < 0 or r_.ci_lo > 0)) else MUTED
                ax.plot([r_.ci_lo, r_.ci_hi], [i, i], color=c, lw=1.7, solid_capstyle="butt")
                ax.plot(r_.pct, i, "D" if i == 0 else "o", color=c, ms=6 if i == 0 else 5)
                ax.text(1.01, i, f"{r_.pct:+.0f}%  p {r_.p_ri:.3f}".replace("-", MINUS), transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
            labs = [t["f4c"]["pair:phone" if (m == pair and v == "phone") else m] for m, v in rows_]
            ax.set_yticks(range(len(rows_))); ax.set_yticklabels(labs); ax.set_ylim(len(rows_) - .5, -.7)
            ax.set_title(t["f4c_g"][g], fontsize=9.5)
        axes[-1].set_xlabel(t["f4c_x"]); fig.suptitle(t["f4c_t"], x=.01, ha="left", fontsize=10, fontweight="bold")
        save(fig, "F4_three_constructs")

    # F5 overlap with the census measures
    if "phone_share" in d:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharex=True)
        for ax, col, ylab, scale in [(axes[0], "phone_share", t["f5c_y1"], 100), (axes[1], "deprivation", t["f5c_y2"], 1)]:
            ax.scatter(d.tt_town_dry, d[col] * scale, s=8 + d.N * 12, facecolor=TEAL, alpha=.35, edgecolor=TEAL, linewidth=.6)
            ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"])
            ax.set_xlabel(t["f5x"], fontsize=8); ax.set_ylabel(ylab, fontsize=8.5)
            ax.text(.03, .97 if col == "deprivation" else .1, f"r = {np.corrcoef(np.log(d.tt_town_dry), d[col])[0, 1]:.2f}".replace("-", MINUS),
                    transform=ax.transAxes, va="top", color=RUST)
        fig.suptitle(t["f5c_t"], x=.01, ha="left", fontsize=10, fontweight="bold"); fig.tight_layout()
        save(fig, "F5_reach_poverty")

    # F7 gradients for digital access, poverty and night lights
    if G7:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
        for ax, col, ttl, xl in zip(axes, ["phone_z", "deprivation", "lights_z"], t["f7_p"], t["f7_x"]):
            xm, o_, l_, h_, g_, f_, pct_ = G7[col]
            ax.vlines(xm, l_, h_, color=MUTED, lw=1.1); ax.plot(xm, o_, "o", color=TEAL, ms=4.5, zorder=3)
            ax.plot(g_, f_, color=RUST, lw=1.5); ax.set_ylim(bottom=0)
            ax.set_title(ttl, fontsize=9); ax.set_xlabel(xl, fontsize=8)
            ax.text(.97, .95, f"{pct_:+.0f}% / SD".replace("-", MINUS), transform=ax.transAxes, ha="right", va="top", color=RUST, fontsize=8.5)
        axes[0].set_ylabel(t["f7_y"]); fig.suptitle(t["f7_t"], x=.01, ha="left", fontsize=10, fontweight="bold"); fig.tight_layout()
        save(fig, "F7_gradients")
        for name, cols_ in [("F7_digital", ["phone_z"]), ("F7_need", ["deprivation", "lights_z"])]:   # one figure per chapter
            idx = [["phone_z", "deprivation", "lights_z"].index(c) for c in cols_]
            fig, axes = plt.subplots(1, len(cols_), figsize=(3.6 * len(cols_) + .6, 3.4), sharey=True, squeeze=False)
            for ax, col, k in zip(axes[0], cols_, idx):
                xm, o_, l_, h_, g_, f_, pct_ = G7[col]
                ax.vlines(xm, l_, h_, color=MUTED, lw=1.1); ax.plot(xm, o_, "o", color=TEAL, ms=4.5, zorder=3)
                ax.plot(g_, f_, color=RUST, lw=1.5); ax.set_ylim(bottom=0)
                ax.set_title(t["f7_p"][k], fontsize=9); ax.set_xlabel(t["f7_x"][k], fontsize=8)
                ax.text(.97, .95, f"{pct_:+.0f}% / SD".replace("-", MINUS), transform=ax.transAxes, ha="right", va="top", color=RUST, fontsize=8.5)
            axes[0][0].set_ylabel(t["f7_y"]); fig.tight_layout()
            save(fig, name)

    # F8 which matters more
    if IMP.exists():
        I = pd.read_csv(IMP).sort_values("shapley_share", ascending=True)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
        yl = [t["f8_lab"][m] for m in I.measure]; cols = [TEAL if m == "physical access" else MUTED for m in I.measure]
        a1.barh(yl, I.shapley_share, color=cols, height=.6)
        for i, v in enumerate(I.shapley_share):
            a1.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8.5)
        a1.set_xlim(0, 100); a1.set_title(t["f8_a"], fontsize=9)
        a2.barh(yl, I.cv_gain, xerr=I.cv_sd, color=cols, height=.6, error_kw={"ecolor": INK2, "lw": .9})
        for i, v in enumerate(I.cv_gain):
            a2.text(max(v, 0) + I.cv_sd.iloc[i] + .6, i, f"{v:+.1f}%".replace("-", MINUS), va="center", fontsize=8.5)
        a2.axvline(0, color=INK2, lw=.8); a2.set_title(t["f8_b"], fontsize=9)
        fig.suptitle(t["f8_t"], x=.01, ha="left", fontsize=10, fontweight="bold"); fig.tight_layout()
        save(fig, "F8_importance")

    # F5 with the Meta wealth index (kept for comparison)
    q = d.dropna(subset=["poverty"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(q.tt_town_dry, q.poverty, s=8 + q.N * 14, facecolor=TEAL, alpha=.35, edgecolor=TEAL, linewidth=.6)
    ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"])
    ax.set_xlabel(t["f5x"]); ax.set_ylabel(t["f5y"])
    ax.text(.02, .97, f"r = {np.corrcoef(np.log(q.tt_town_dry), q.poverty)[0, 1]:.2f}".replace("-", MINUS), transform=ax.transAxes, va="top", color=RUST)
    for n in (1, 5, 15):
        ax.scatter([], [], s=8 + n * 14, facecolor=TEAL, alpha=.35, edgecolor=TEAL, label=f"{n}")
    ax.legend(title=t["f5leg"], loc="lower right", fontsize=8, title_fontsize=8, labelspacing=1.0)
    ax.set_title(t["f5t"])
    save(fig, "F5_reach_poverty_rwi")
print("figures written to", FIG)

# ---------------- data for the site's interactive charts (same numbers as the figures above) ----------------
import json
r2 = lambda a: [None if (v is None or not np.isfinite(v)) else round(float(v), 4) for v in np.asarray(a, dtype=float)]
CH = {"grad": {"travel": {"x": r2(f2_x), "rate": r2(f2_obs), "lo": r2(f2_lo), "hi": r2(f2_hi), "fx": r2(grid), "fy": r2(f2_fit),
                          "pct": round(100 * (np.exp(fit2.params["rel"]) - 1), 1)}},
      "types": [{"key": g, "rate": round(float(o), 3), "lo": round(float(l), 3), "hi": round(float(h), 3), "n": int(n), "tt": round(float(tt), 0)}
                for g, o, l, h, n, tt in zip(G3, f3_obs, f3_lo, f3_hi, f3_n, f3_tt)],
      "coef": R[["model", "var", "pct", "ci_lo", "ci_hi", "p_ri", "p_cluster"]].round(6).to_dict("records")}   # 6 decimals: no double rounding of p
for col, key in [("phone_z", "phone"), ("deprivation", "deprivation"), ("lights_z", "lights")]:
    if col in G7:
        xm, o_, l_, h_, g_, f_, pct_ = G7[col]
        CH["grad"][key] = {"x": r2(xm), "rate": r2(o_), "lo": r2(l_), "hi": r2(h_), "fx": r2(g_), "fy": r2(f_), "pct": round(float(pct_), 1)}
if IMP.exists():
    CH["importance"] = pd.read_csv(IMP).round(2).to_dict("records")
# capital vs nearest town (spec 9-33) and infrastructure events (spec 9-32, 9-34)
T = ROOT / "outputs/tables"
CAPF, PIN = T / "kh_capital_compare.csv", T / "kh_pinfra_panel.csv"
if (T / "kh_designs.csv").exists():   # spec 9-36
    CH["designs"] = pd.read_csv(T / "kh_designs.csv")[["design", "model", "pct", "ci_lo", "ci_hi", "p_ri", "p_cluster"]].round(6).to_dict("records")
if (T / "kh_hub_consistency.csv").exists():   # spec 9-38: each design on all districts and on the countryside alone
    CH["hubs"] = pd.read_csv(T / "kh_hub_consistency.csv")[["test", "sample", "pct", "ci_lo", "ci_hi", "p_ri", "p_cluster", "n"]].round(6).to_dict("records")
if (T / "kh_river.csv").exists():   # spec 9-39
    CH["river"] = pd.read_csv(T / "kh_river.csv").round(6).to_dict("records")
if CAPF.exists():
    CH["capital"] = pd.read_csv(CAPF).round(4).to_dict("records")   # one rounding only, in the chart
    CH["capital_coef"] = pd.read_csv(T / "kh_capital.csv")[["model", "var", "fe", "pct", "ci_lo", "ci_hi", "p_ri", "p_cluster"]].round(6).to_dict("records")
if PIN.exists():
    pn = pd.read_csv(PIN)
    ctl = pn[pn.control].groupby("yr").activities.mean()
    CH["wpanel"] = {"years": [int(y) for y in ctl.index], "control": r2(ctl.values),
                    "treated": [{"province": p, "work": g.treated_by.iloc[0], "year": int(g.completed.iloc[0]), "y": r2(g.sort_values("yr").activities.values)}
                                for p, g in pn[pn.treated_by.notna()].groupby("province")]}
    CH["wpanel"]["treated"].sort(key=lambda d: d["year"])
    ev = pd.read_csv(T / "kh_pinfra_events.csv"); pr = pd.read_csv(T / "kh_pinfra.csv")
    CH["wcoef"] = pd.concat([pr, ev.drop(columns="event")])[["model", "pct", "ci_lo", "ci_hi", "p_ri", "treated_pre", "treated_post"]].drop_duplicates("model").round(4).to_dict("records")
    cl = pd.read_csv(T / "kh_pinfra_cells.csv"); works = [c for c in cl.columns if c not in ("lon", "lat", "pop", "k")]
    cl["save"] = cl[works].max(axis=1); cl["work"] = cl[works].idxmax(axis=1)
    cl["gx"], cl["gy"] = (cl.lon / .04).round() * .04, (cl.lat / .04).round() * .04
    gcell = cl.sort_values("save").groupby(["gx", "gy"]).tail(1)
    gcell = gcell[gcell.save >= 5]
    CH["wgrid"] = {"step": .04, "works": works, "cells": [[round(x, 2), round(y, 2), round(float(v), 0), works.index(w)] for x, y, v, w in zip(gcell.gx, gcell.gy, gcell.save, gcell.work)]}
    CH["wsites"] = [{"work": "NR48 Koh Kong bridges", "lon": 103.47, "lat": 11.19, "year": 2008}, {"work": "Sekong bridge", "lon": 105.99, "lat": 13.54, "year": 2008},
                    {"work": "Prek Tamak bridge", "lon": 105.00, "lat": 11.75, "year": 2010}, {"work": "NR78 Ratanakiri paving", "lon": 106.55, "lat": 13.72, "year": 2012},
                    {"work": "Stung Treng bridge", "lon": 105.94, "lat": 13.53, "year": 2015}, {"work": "Neak Loeung bridge", "lon": 105.27, "lat": 11.27, "year": 2015}]
def _clean(o):   # browsers reject NaN in JSON: write missing values as null
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return None if isinstance(o, float) and not np.isfinite(o) else o


(ROOT / "site/data/charts.json").write_text(json.dumps(_clean(CH), ensure_ascii=False, allow_nan=False), encoding="utf-8")
print("site/data/charts.json written")
