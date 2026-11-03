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
               f3sub=lambda n, t: f"{n} districts\n{t:.0f} min",               f4t="Change in projects when travel time doubles, across checks", f4x="% change in projects (95% CI, province-clustered); teal: CI excludes 0 and randomisation-inference p < 0.05",
               f4g=["Main estimate", "Robustness", "Donor groups", "Subsample"],
               f4lab={"reach alone": "Main estimate (all 141 districts)", "R1a geocodes as published (no reclassification)": "Geocodes as published",
                      "R1b worst case: province-level geocodes to remotest district": "Reclassified projects in remotest district",
                      "R2a province-only projects added at remotest district": "+ province-only projects, remotest district",
                      "R2b province-only projects added at best-reached district": "+ province-only projects, best-reached district",
                      "R3 Phnom Penh excluded": "Phnom Penh excluded", "R4 Weiss et al. (2018) travel time to cities": "Weiss et al. (2018) travel-time map",
                      "rural districts: reach alone": "Rural districts only (112)",
                      "donors: EU/US": "EU and US donors", "donors: World Bank": "World Bank", "donors: China": "China"},
               f5t="Remote districts are also the poorest", f5x="Travel time to the nearest town (minutes, log scale)", f5y="Poverty (−Meta wealth index, SD)",
               f5leg="Projects"),
    "ko": dict(font="Malgun Gothic", sfx="_ko",
               f1t="군별 원조 사업과 가장 가까운 도시까지의 이동시간", f1cb="인구 2만 명 이상 도시까지 이동시간 (분, 로그 눈금)",
               f1leg="사업 2017–2020", proj=lambda n: f"사업 {n}개",
               f2t="같은 주의 다른 군보다 도시에서 먼 군일수록 사업이 적다",
               f2x="가장 가까운 도시까지 이동시간 (주 평균 대비)", f2y="인구 10만 명당 사업 수",
               f2a="군 10분위 (95% 신뢰구간, 주 단위 부트스트랩)", f2b=lambda p: f"PPML 추정: 이동시간 두 배일 때 {p:+.0f}%".replace("-", MINUS),
               f2ticks=["¼배", "½배", "주\n평균", "2배", "4배"],
               f3t="사업은 프놈펜과 도청 소재지에 몰린다", f3y="인구 10만 명당 사업 수",
               f3lab=["프놈펜\n(4개 칸)", "도청\n소재지", "농촌:\n가까운 1/3", "농촌:\n중간 1/3", "농촌:\n먼 1/3"],
               f3sub=lambda n, t: f"{n}개 군\n{t:.0f}분",               f4t="이동시간이 두 배일 때 사업 수 변화: 점검별 비교", f4x="사업 수 변화 % (95% 신뢰구간, 주 군집); 청록: 구간이 0을 포함하지 않고 무작위화 추론 p < 0.05",
               f4g=["주 추정치", "견고성 점검", "공여자 그룹", "하위 표본"],
               f4lab={"reach alone": "주 추정치 (141개 군 전체)", "R1a geocodes as published (no reclassification)": "지오코딩 원자료 그대로",
                      "R1b worst case: province-level geocodes to remotest district": "재분류 사업을 가장 외진 군에",
                      "R2a province-only projects added at remotest district": "+ 주 단위 사업, 가장 외진 군에",
                      "R2b province-only projects added at best-reached district": "+ 주 단위 사업, 가장 가까운 군에",
                      "R3 Phnom Penh excluded": "프놈펜 제외", "R4 Weiss et al. (2018) travel time to cities": "Weiss et al. (2018) 이동시간 지도",
                      "rural districts: reach alone": "농촌 군만 (112개)",
                      "donors: EU/US": "유럽·미국", "donors: World Bank": "세계은행", "donors: China": "중국"},
               f5t="외진 군이 가장 가난한 군이기도 하다", f5x="가장 가까운 도시까지 이동시간 (분, 로그 눈금)", f5y="빈곤 (−Meta 부 지수, 표준편차)",
               f5leg="사업 수"),
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
    breaks = {1: t["f4g"][1], keys.index("rural districts: reach alone"): t["f4g"][3], keys.index("donors: EU/US"): t["f4g"][2]}
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
    save(fig, "F4_coefficients")

    # F5 overlap
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
    save(fig, "F5_reach_poverty")
print("figures written to", FIG)
