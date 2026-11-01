"""Figures (static PNG, 200 dpi) for the Timor-Leste accessibility-bias study.

Palette = reference dataviz palette (sequential blue ramp, second sequential orange,
blue<->red diverging with gray midpoint, categorical slots 1-3 only on scatter/maps).
"""
import json
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, TAB = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fde3d6", "#f9c3a8", "#f4a07c", "#eb6834", "#c9501f", "#9e3d16", "#702b0f"]
seq_blue = LinearSegmentedColormap.from_list("b", BLUE)
seq_orange = LinearSegmentedColormap.from_list("o", ORANGE)
div = LinearSegmentedColormap.from_list("d", ["#b8302f", "#e34948", "#f2a3a2", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])

mpl.rcParams.update({
    "font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

U = gpd.read_file(PROC / "units.gpkg")
P = gpd.read_file(PROC / "posto_covariates.gpkg")
corr = gpd.read_file(PROC / "corridors.gpkg")
roads = gpd.read_file(PROC / "roads.gpkg")
major = roads[roads.highway.isin(["trunk", "primary", "secondary", "trunk_link", "primary_link", "secondary_link"])]
mun = P.dissolve("municipio")
DILI = (125.5789, -8.5569)
D = U[U.unit != "TL06U"].copy()


def base_map(ax, title, sub=None):
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=13, pad=22 if sub else 6)
    if sub:
        ax.text(0, 1.01, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    ax.set_xlim(124.0, 127.4); ax.set_ylim(-9.55, -8.1)


def overlay(ax, roads_=True, corridors=False, dili=True):
    mun.boundary.plot(ax=ax, color="#ffffff", linewidth=1.4)
    mun.boundary.plot(ax=ax, color=INK2, linewidth=0.5)
    if roads_:
        major.plot(ax=ax, color=INK, linewidth=0.5, alpha=0.55)
    if corridors:
        corr.plot(ax=ax, color=S2, linewidth=2.2)
    if dili:
        ax.plot(*DILI, marker="*", ms=13, color=INK, mec="#ffffff", mew=1.2, zorder=10)
        ax.annotate("Dili", DILI, xytext=(4, 6), textcoords="offset points", fontsize=9, weight="bold")


def cbar(fig, ax, mappable, label):
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, labelsize=8); cb.set_label(label, color=INK2, fontsize=9)


# ---------- F1: wet-season travel time + corridors ----------
fig, ax = plt.subplots(figsize=(11, 5.6))
v = (U.tt_wet_min / 60).clip(upper=14)
U.assign(v=v).plot(column="v", cmap=seq_blue, ax=ax, edgecolor="#ffffff", linewidth=0.3)
overlay(ax, corridors=True)
base_map(ax, "그림 1. 우기 기준 딜리까지 실질 이동시간 (인구가중, 시간)",
         "OSM 도로망 x 노면·경사·우기 감속·산사태 위험 지형, 오에쿠시·아타우로는 해상 구간 포함 | 주황선 = 공여기관 도로 개량 구간")
sm_ = plt.cm.ScalarMappable(cmap=seq_blue, norm=plt.Normalize(0, 14)); cbar(fig, ax, sm_, "시간 (14시간 이상은 14로 표시)")
for _, r in corr.iterrows():
    c = r.geometry.centroid
    ax.annotate(f"{r.corridor}\n{r.financier} {r.year}", (c.x, c.y), fontsize=7, color=INK2, ha="center",
                xytext={"Tibar-Gleno": (-45, -10), "Dili-Tibar-Liquica": (-30, 14), "Manatuto-Natarbora": (0, -16),
                        "Dili-Aileu-Maubisse-Ainaro": (-70, 0)}.get(r.corridor, (0, 10)), textcoords="offset points")
fig.savefig(FIG / "fig01_travel_time_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F2: composite accessibility ----------
fig, ax = plt.subplots(figsize=(11, 5.6))
U.plot(column="access_pc1", cmap=div, norm=TwoSlopeNorm(0, -2, 3.2), ax=ax, edgecolor="#ffffff", linewidth=0.3)
overlay(ax, roads_=False)
base_map(ax, "그림 2. 종합 접근성 지수 (주성분 1, 표준화)", "구성: 우기 이동시간, 포장률, 도로밀도, 급경사 비율, 우기 지연, Ookla 측정 타일·측정 수 | 파랑 = 접근성 높음")
cbar(fig, ax, plt.cm.ScalarMappable(cmap=div, norm=TwoSlopeNorm(0, -2, 3.2)), "표준편차 단위")
fig.savefig(FIG / "fig02_access_index_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F3: precise project locations + counts ----------
ad = RAW / "aiddata_tl/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
L = pd.read_csv(ad / "locations.csv", encoding="latin1")
L = L[L.precision_code.isin([1, 2])]
locs = pd.DataFrame(json.loads((RAW / "iati/dportal_loc_TL.json").read_text(encoding="utf-8")))
locs = locs[(locs.reporting != "AidData") & (locs.location_precision == 1)]
locs["xy"] = locs.location_longitude.round(2).astype(str) + locs.location_latitude.round(2).astype(str)
locs = locs[locs.xy.map(locs.groupby("xy").aid.nunique()) < 40]
pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pd.concat([L.longitude, locs.location_longitude]),
                                                   pd.concat([L.latitude, locs.location_latitude])), crs=4326)
pts = gpd.sjoin(pts, P[["geometry"]], predicate="within")
rng = np.random.default_rng(1)
pts["geometry"] = gpd.points_from_xy(pts.geometry.x + rng.normal(0, 0.008, len(pts)), pts.geometry.y + rng.normal(0, 0.008, len(pts)))
fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
ax = axs[0]
P.plot(ax=ax, color="#f0efec", edgecolor="#ffffff", linewidth=0.3)
overlay(ax, dili=True)
pts.plot(ax=ax, color=S1, markersize=4, alpha=0.45, linewidth=0)
base_map(ax, "그림 3a. 정밀 좌표가 있는 ODA 사업 위치", f"AidData AMP(정밀도 1-2) + IATI(정밀도 1, 기본값 좌표 제외), n = {len(pts):,} 지점")
ax = axs[1]
U.assign(v=U.np_dev / U.pop2022 * 1e4).plot(column="v", cmap=seq_blue, ax=ax, edgecolor="#ffffff", linewidth=0.3, vmin=0, vmax=60)
overlay(ax, roads_=False)
base_map(ax, "그림 3b. 인구 1만 명당 비인프라 개발사업 수 (정밀 위치)", "2000-2026 시작 사업, 딜리 도심 4개 포스토 통합")
cbar(fig, ax, plt.cm.ScalarMappable(cmap=seq_blue, norm=plt.Normalize(0, 60)), "사업 수 / 인구 1만")
fig.savefig(FIG / "fig03_projects_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F4: need (poverty) and allocation gap ----------
fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
ax = axs[0]
U.assign(v=-U.rwi).plot(column="v", cmap=seq_orange, ax=ax, edgecolor="#ffffff", linewidth=0.3)
overlay(ax, roads_=False)
base_map(ax, "그림 4a. 필요: 빈곤도 (-상대적 부 지수, Meta RWI)", "인구가중 평균 | 진할수록 가난")
cbar(fig, ax, plt.cm.ScalarMappable(cmap=seq_orange, norm=plt.Normalize((-U.rwi).min(), (-U.rwi).max())), "-RWI")
ax = axs[1]
m = smf.negativebinomial("np_dev ~ log_pop + I(-rwi)", U.assign(np_dev=U.np_dev.round().astype(int))).fit(disp=0)
U["gap"] = np.log((U.np_dev + 1) / (m.predict() + 1))
U.plot(column="gap", cmap=div, norm=TwoSlopeNorm(0, -1.2, 1.2), ax=ax, edgecolor="#ffffff", linewidth=0.3)
overlay(ax, roads_=False, corridors=True)
base_map(ax, "그림 4b. 필요 대비 배분 격차", "log(실제/기대 사업 수), 기대값 = 인구·빈곤만으로 예측 | 파랑 = 필요 대비 과다, 빨강 = 과소")
cbar(fig, ax, plt.cm.ScalarMappable(cmap=div, norm=TwoSlopeNorm(0, -1.2, 1.2)), "log 비율")
fig.savefig(FIG / "fig04_need_and_gap_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F5: scatter count vs access ----------
LABEL = {"Dili (urban)", "Baucau", "Lospalos", "Maliana", "Suai", "Pante Macassar", "Same", "Viqueque", "Lacluta",
         "Uatucarbau", "Fatululic", "Atauro", "Metinaro", "Passabe", "Aileu Vila", "Laclubar", "Hatolia"}
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, (x, xl) in zip(axs, [("access_pc1", "종합 접근성 지수 (표준화)"), ("tt_wet_min", "우기 이동시간 (시간, 로그축)")]):
    xv = U[x] / (60 if x == "tt_wet_min" else 1)
    sz = np.sqrt(U.pop2022) / 6
    oe = U.mun_pcode == "TL12"
    ax.scatter(xv[~oe], U.np_dev[~oe], s=sz[~oe], color=S1, alpha=0.75, edgecolor=SURF, linewidth=1.5, label="본토 포스토")
    ax.scatter(xv[oe], U.np_dev[oe], s=sz[oe], color=S2, alpha=0.85, edgecolor=SURF, linewidth=1.5, label="오에쿠시")
    for _, r in U.iterrows():
        if r.unit_name in LABEL:
            ax.annotate(r.unit_name, (r[x] / (60 if x == "tt_wet_min" else 1), r.np_dev), fontsize=7.5, color=INK2,
                        xytext=(5, 3), textcoords="offset points")
    dd = D.assign(xx=np.log(D.tt_wet_min / 60) if x == "tt_wet_min" else D[x])
    fit = smf.ols("np_dev ~ xx", dd).fit()
    xs = np.linspace(dd.xx.min(), dd.xx.max(), 50)
    ax.plot(np.exp(xs) if x == "tt_wet_min" else xs, fit.params.iloc[0] + fit.params.iloc[1] * xs, color=INK2, lw=1.5, ls="--",
            label=f"OLS (딜리 제외) r²={fit.rsquared:.2f}")
    if x == "tt_wet_min":
        ax.set_xscale("log"); ax.set_xticks([0.25, 1, 2, 4, 8, 14]); ax.set_xticklabels(["0.25", "1", "2", "4", "8", "14"])
    ax.set_xlabel(xl); ax.set_ylabel("정밀 위치 비인프라 개발사업 수")
    ax.legend(loc="upper left" if x == "access_pc1" else "upper right", fontsize=8)
axs[0].set_title("그림 5. 접근성과 개발사업 수 (원 크기 = 인구)")
fig.tight_layout(); fig.savefig(FIG / "fig05_scatter_access.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F6: per-capita vs access and poverty ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (x, xl) in zip(axs, [("access_pc1", "종합 접근성 지수"), ("rwi", "상대적 부 지수 (RWI, 높을수록 부유)"), ("log_pop", "log 인구")]):
    yv = D.np_dev / D.pop2022 * 1e4
    ax.scatter(D[x], yv, s=26, color=S1, alpha=0.75, edgecolor=SURF, linewidth=1)
    fit = smf.ols("y ~ x", pd.DataFrame({"y": yv, "x": D[x]})).fit()
    xs = np.linspace(D[x].min(), D[x].max(), 50)
    ax.plot(xs, fit.params.iloc[0] + fit.params.iloc[1] * xs, color=INK2, ls="--", lw=1.4)
    ax.set_xlabel(xl); ax.set_ylabel("1만 명당 사업 수" if x == "access_pc1" else "")
    ax.text(0.98, 0.95, f"r = {np.corrcoef(D[x], yv)[0,1]:+.2f}", transform=ax.transAxes, ha="right", va="top", color=INK2)
axs[0].set_title("그림 6. 인구 1인당 사업 수는 접근성·부와 무관, 인구 규모와 반비례 (딜리 제외 61개)")
fig.tight_layout(); fig.savefig(FIG / "fig06_percapita_scatter.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F7: coefficient plot ----------
allm = pd.read_csv(TAB / "h1_all_models.csv")
het = pd.read_csv(TAB / "h1_heterogeneity.csv")
comp = pd.read_csv(TAB / "h1_components.csv")
fig, axs = plt.subplots(1, 3, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
ax = axs[0]
mods = ["M1 all units, main outcome", "M3 excl. Dili urban", "M4 precise-only outcome", "M5 precise-only, excl. Dili",
        "M6 precise, + density"]
lab = ["M1 전체·주 지표", "M3 딜리 제외", "M4 정밀 위치만", "M5 정밀·딜리 제외", "M6 정밀·밀도 통제"]
for i, (mname, l) in enumerate(zip(mods, lab)):
    for j, (term, col, nm) in enumerate([("access_pc1", S1, "접근성 (1 SD)"), ("poor_z", S2, "빈곤도 (1 SD)")]):
        r = allm[(allm.model == mname) & (allm.term == term)].iloc[0]
        y = i + (0.15 if j == 0 else -0.15)
        ax.errorbar(r.coef, y, xerr=1.96 * r.se, fmt="o", color=col, ms=7, capsize=0, lw=2, label=nm if i == 0 else None)
ax.axvline(0, color=AXIS, lw=1); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab); ax.invert_yaxis()
ax.set_xlabel("음이항 계수 (95% CI), log 인구 통제"); ax.set_title("그림 7a. 주 모형"); ax.legend(loc="upper right", fontsize=8)
ax = axs[1]
cl = {"log wet-season travel time (+ = farther)": "우기 이동시간(log, +는 멂)", "paved share": "포장률", "log road density": "도로밀도(log)",
      "steep/landslide terrain share": "급경사·산사태 지형", "physical sub-index": "물리 접근성 하위지수", "network sub-index (Ookla)": "네트워크 하위지수"}
for i, r in comp.reset_index(drop=True).iterrows():
    c = S3 if "network" in r.label else S1
    ax.errorbar(r.coef, i, xerr=1.96 * r.se, fmt="o", color=c, ms=7, lw=2)
ax.axvline(0, color=AXIS, lw=1); ax.set_yticks(range(len(comp))); ax.set_yticklabels([cl[x] for x in comp.label]); ax.invert_yaxis()
ax.set_xlabel("계수 (1 SD, 95% CI)"); ax.set_title("그림 7b. 접근성 구성요소별 (각각 투입)")
ax = axs[2]
hk = {"Era 1: start 2000-2015": "시기1: 2000-2015", "Era 2: start 2016-2026": "시기2: 2016-2026", "Bilateral donors": "양자 공여",
      "Multilaterals & UN": "다자·UN", "Korea + Japan": "한국+일본", "Education": "교육", "Health": "보건", "Water & sanitation": "식수·위생",
      "Governance": "거버넌스", "Productive sectors": "생산부문", "ACCESS infrastructure projects": "접근성 인프라 사업", "Humanitarian": "인도적 지원"}
for i, r in het.reset_index(drop=True).iterrows():
    ax.errorbar(r.b_access, i, xerr=1.96 * r.se_access, fmt="o", color=S1, ms=6, lw=2)
ax.axvline(0, color=AXIS, lw=1); ax.set_yticks(range(len(het))); ax.set_yticklabels([hk[g] for g in het.group]); ax.invert_yaxis()
ax.set_xlabel("접근성 계수 (95% CI)"); ax.set_title("그림 7c. 시기·공여기관·부문별 접근성 효과")
fig.tight_layout(); fig.savefig(FIG / "fig07_coefficients.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F8: quintile shares ----------
Q = pd.read_csv(TAB / "simple_quintiles.csv", index_col=0)
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(Q)); w = 0.38
ax.bar(x - w / 2 - 0.01, Q["pop_share_%"], w, color=MUTED, label="인구 점유율 %")
ax.bar(x + w / 2 + 0.01, Q["precise_share_%"], w, color=S1, label="정밀 위치 사업 점유율 %")
for i, (a, b) in enumerate(zip(Q["pop_share_%"], Q["precise_share_%"])):
    ax.text(i + w / 2, b + 0.6, f"{b:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(["Q1\n최저", "Q2", "Q3", "Q4", "Q5\n최고", "딜리\n(도심)"])
ax.set_ylabel("%"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
ax.set_title("그림 8. 접근성 5분위별 인구 대비 사업 점유율 (딜리 도심 별도)")
fig.savefig(FIG / "fig08_quintile_shares.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F9: concentration curves ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, df, t in [(axs[0], U, "딜리 포함 (62개)"), (axs[1], D, "딜리 제외 (61개)")]:
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    for rk, col, nm in [("access_pc1", S1, "접근성 낮은→높은 순"), ("rwi", S2, "가난한→부유한 순")]:
        d = df.sort_values(rk)
        cp = np.r_[0, d.pop2022.cumsum() / d.pop2022.sum()]; cy = np.r_[0, d.np_dev.cumsum() / d.np_dev.sum()]
        ax.plot(cp, cy, color=col, lw=2, label=nm)
    ax.set_xlabel("누적 인구 비율"); ax.set_ylabel("누적 사업 비율"); ax.set_title(t, fontsize=11); ax.legend(fontsize=8)
fig.suptitle("그림 9. 집중곡선: 대각선 위 = 하위(덜 접근/더 가난) 지역에 인구보다 많은 사업", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "fig09_concentration.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F10: event study ----------
EV = pd.read_csv(TAB / "h2_event_study.csv")
fig, ax = plt.subplots(figsize=(10, 4.6))
for (y, col, nm, off) in [("n_dev_new", S1, "전체 (지자체 분할 포함)", -0.1), ("np_dev_new", S2, "정밀 위치만", 0.1)]:
    e = EV[EV.outcome == y].sort_values("rel_year")
    ax.errorbar(e.rel_year + off, e.coef, yerr=1.96 * e.se, fmt="-o", color=col, ms=6, lw=2, capsize=0, label=nm)
ax.axhline(0, color=AXIS); ax.axvline(-0.5, color=INK2, ls=":", lw=1)
ax.set_xlabel("도로 개량 완공 기준 상대 연도 (-1 = 기준, 양끝 구간 묶음)"); ax.set_ylabel("PPML 계수 (단위·연도 고정효과)")
ax.set_title("그림 10. 도로 개량 전후 신규 개발사업 (이벤트 스터디, 딜리 제외)"); ax.legend(fontsize=8)
ax.text(0.01, 0.02, "주의: 2015/16년 데이터 소스 전환(AMP→IATI·예산서)과 처치 시점(2018·2020)이 겹쳐 해석 제한", transform=ax.transAxes, fontsize=8, color=INK2)
fig.savefig(FIG / "fig10_event_study.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F11: projects per year by source & location precision ----------
PR = pd.read_csv(PROC / "oda_projects.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
lv = AS.groupby("uid")["level"].agg(lambda s: "포스토" if (s == "posto").any() else ("지자체" if (s == "municipio").any() else "전국·본부"))
PR["lvl"] = PR.uid.map(lv)
PR = PR[(PR.start_year >= 2000) & (PR.start_year <= 2026)]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
ct = PR.pivot_table(index="start_year", columns="source", values="uid", aggfunc="count").fillna(0)
cols = {"AidData-AMP": S1, "IATI": S2, "BudgetBook5": S3, "WorldBank-API": MUTED}
bottom = np.zeros(len(ct))
for s in ["AidData-AMP", "IATI", "BudgetBook5", "WorldBank-API"]:
    axs[0].bar(ct.index, ct[s], bottom=bottom, color=cols[s], label=s, width=0.8, edgecolor=SURF, linewidth=1)
    bottom += ct[s].values
axs[0].set_title("그림 11a. 시작 연도별 사업 수 (소스별)"); axs[0].legend(fontsize=8); axs[0].grid(axis="x", visible=False)
sh = PR.pivot_table(index="start_year", columns="lvl", values="uid", aggfunc="count").fillna(0)
sh = sh.div(sh.sum(1), axis=0) * 100
for l, c in [("포스토", S1), ("지자체", S3), ("전국·본부", MUTED)]:
    axs[1].plot(sh.index, sh[l], color=c, lw=2, label=l)
axs[1].set_ylabel("%"); axs[1].set_title("그림 11b. 위치 특정 수준별 비중"); axs[1].legend(fontsize=8)
axs[1].axvspan(2015.5, 2026.5, color=GRID, alpha=0.5, lw=0); axs[1].text(2016, 92, "AMP 종료 이후", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(FIG / "fig11_projects_by_year.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F12: network small multiples ----------
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for ax, y in zip(axs, [2019, 2022, 2025]):
    o = pd.read_csv(RAW / f"ookla/ookla_mobile_{y}Q1_TL.csv", dtype={"quadkey": str})
    z = 16
    def c(qk):
        x = yy = 0
        for i, ch in enumerate(qk):
            m = 1 << (z - i - 1); d = int(ch)
            if d & 1: x |= m
            if d & 2: yy |= m
        n = 2 ** z
        return x / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (yy + .5) / n))))
    xy = np.array([c(q) for q in o.quadkey])
    P.plot(ax=ax, color="#f0efec", edgecolor="#ffffff", linewidth=0.3)
    overlay(ax, roads_=False)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.log10(o.avg_d_kbps / 1000 + 0.1), cmap=seq_blue, s=6, vmin=-1, vmax=2)
    base_map(ax, f"{y}년 1분기", f"측정 타일 {len(o)}개, 측정 {o.tests.sum():,}회, 중위 다운로드 {o.avg_d_kbps.median()/1000:.1f} Mbps")
fig.suptitle("그림 12. Ookla 모바일 측정 타일 (색 = 다운로드 속도, log Mbps)", x=0.01, ha="left", fontweight="bold")
cb = fig.colorbar(sc, ax=axs, orientation="horizontal", fraction=0.04, pad=0.02, aspect=60)
cb.set_ticks([-1, 0, 1, 2]); cb.set_ticklabels(["0.1", "1", "10", "100 Mbps"]); cb.outline.set_visible(False)
fig.savefig(FIG / "fig12_network_maps.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F13: Shapley ----------
S = pd.read_csv(TAB / "shapley_r2.csv")
kor = {"Accessibility (PC1)": "접근성 (PC1)", "Poverty (-RWI)": "빈곤도", "Population size": "인구 규모", "Density": "인구밀도"}
fig, ax = plt.subplots(figsize=(8, 3.4))
S = S.sort_values("share")
ax.barh([kor[p] for p in S.predictor], S.share * 100, color=[S1 if "Access" in p else MUTED for p in S.predictor], height=0.55)
for i, v in enumerate(S.share * 100):
    ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=9, color=INK2)
ax.set_xlabel("설명된 분산(R²) 중 기여 비율 (Shapley)"); ax.grid(axis="y", visible=False)
ax.set_title("그림 13. 1인당 사업 수 설명력 분해 (딜리 제외)")
fig.savefig(FIG / "fig13_shapley.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
