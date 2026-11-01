"""Figures for the extended analyses (fig14-fig21). Same palette/rcParams as s06."""
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB, FIG = ROOT / "data" / "processed", ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
div = LinearSegmentedColormap.from_list("d", ["#b8302f", "#e34948", "#f2a3a2", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
mpl.rcParams.update({
    "font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False})
U = pd.read_csv(PROC / "units.csv")
D = U[U.unit != "TL06U"].copy()
PR = pd.read_csv(PROC / "oda_projects.csv")
PR = PR[PR.start_year.between(2000, 2026)]

# ---------- F14 target-type composition ----------
TT = {"located": "지역 특정", "central_institution": "중앙기관(딜리)", "nationwide": "전국 보편", "unresolved": "미확인", "regional_multicountry": "지역 외·다국가"}
COL = {"located": S1, "central_institution": S2, "nationwide": S3, "unresolved": AXIS, "regional_multicountry": "#d8d6cf"}
fig, ax = plt.subplots(figsize=(11, 2.9))
for i, (lab, w) in enumerate([("사업 수", PR.target_type.value_counts(normalize=True)),
                              ("금액", PR.groupby("target_type").usd.sum() / PR.usd.sum())]):
    left = 0
    for t in TT:
        v = w.get(t, 0) * 100
        ax.barh(i, v, left=left, color=COL[t], edgecolor=SURF, linewidth=2, height=0.6, label=TT[t] if i == 0 else None)
        if v > 4:
            ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="#ffffff" if t in ("located", "central_institution") else INK, fontsize=9)
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels(["사업 수", "금액"]); ax.set_xlim(0, 100); ax.grid(False); ax.invert_yaxis()
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9); ax.set_xticks([])
ax.set_title("그림 14. 대상 지역 유형별 구성 (2000-2026 시작 사업 3,982건, 상위 110건 수작업 검토 반영)")
fig.savefig(FIG / "fig14_target_types.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F15 correlation heatmap ----------
C = pd.read_csv(TAB / "ext_B_corr_spearman.csv", index_col=0)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(C.values, cmap=div, vmin=-1, vmax=1)
ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=60, ha="right", fontsize=8.5)
ax.set_yticks(range(len(C))); ax.set_yticklabels(C.index, fontsize=8.5); ax.grid(False)
for i in range(len(C)):
    for j in range(len(C)):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="#ffffff" if abs(v) > 0.6 else INK)
for b in [6.5, 9.5, 16.5]:
    ax.axhline(b, color=INK2, lw=1); ax.axvline(b, color=INK2, lw=1)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
ax.set_title("그림 15. 접근성 구성요소·필요·사업 지표 간 Spearman 상관 (딜리 제외 61개)\n(구획: 접근성 구성요소 | 하위·종합지수 | 인구·필요 | 사업 지표)", fontsize=11)
fig.savefig(FIG / "fig15_correlation_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F16 roads vs network ----------
B = pd.read_csv(TAB / "ext_B_roads_vs_network.csv")
B = B[B.outcome == "precise"]
G = pd.read_csv(TAB / "ext_B_2x2.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axs[0]
labs = {"B1 physical only": "도로·물리만", "B2 network only": "네트워크만", "B3 both": "둘 다", "B4 interaction": "둘 다 + 상호작용"}
for i, (mname, l) in enumerate(labs.items()):
    r = B[B.model == mname].iloc[0]
    for j, (v, col, nm) in enumerate([("phys_z", S1, "도로·물리 (1 SD)"), ("net_z", S3, "네트워크 (1 SD)"), ("phys_z:net_z", S2, "상호작용")]):
        if pd.notna(r.get(f"b_{v}")):
            ax.errorbar(r[f"b_{v}"], i + (j - 1) * 0.2, xerr=1.96 * r[f"se_{v}"], fmt="o", color=col, ms=7, lw=2, label=nm if i == 3 else None)
    ax.text(0.37, i, f"AIC {r.aic:.0f}", va="center", fontsize=8.5, color=INK2)
ax.axvline(0, color=AXIS); ax.set_yticks(range(4)); ax.set_yticklabels(labs.values()); ax.invert_yaxis(); ax.set_xlim(-0.35, 0.45)
ax.set_xlabel("PPML 계수 (95% CI), 정밀 위치 사업, log 인구·센서스 박탈 통제"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("그림 16a. 도로 vs 네트워크: 모형 비교")
ax = axs[1]
order = ["both low", "road high only", "network high only", "both high"]
kor = {"both low": "둘 다 낮음", "road high only": "도로만 높음", "network high only": "네트워크만 높음", "both high": "둘 다 높음"}
cols = [AXIS, S1, S3, S2]
ax.bar([kor[o] for o in order], G.loc[order, "precise_per10k"], color=cols, width=0.6)
for i, o in enumerate(order):
    ax.text(i, G.loc[o, "precise_per10k"] + 0.3, f"{G.loc[o, 'precise_per10k']:.1f}\n({int(G.loc[o, 'units'])}개)", ha="center", fontsize=8.5, color=INK2)
ax.set_ylabel("인구 1만 명당 정밀 위치 사업"); ax.grid(axis="x", visible=False)
ax.set_title("그림 16b. 중위수 기준 2×2 집단")
fig.tight_layout(); fig.savefig(FIG / "fig16_roads_vs_network.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F17 reverse causality ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
x = D.n_dev_0610 / D.cpop2010 * 1e4; y = D.g_10_15 * 100
ax = axs[0]; ax.scatter(x, y, s=np.sqrt(D.cpop2010) / 5, color=S1, alpha=0.75, edgecolor=SURF, linewidth=1)
fit = smf.ols("y ~ np.log1p(x)", pd.DataFrame({"x": x, "y": y})).fit()
xs = np.linspace(x.min(), x.max(), 60); ax.plot(xs, fit.params.iloc[0] + fit.params.iloc[1] * np.log1p(xs), color=INK2, ls="--")
ax.set_xlabel("2006-10년 시작 개발사업 / 2010년 인구 1만"); ax.set_ylabel("2010→2015 센서스 연평균 인구증가율 (%)")
ax.text(0.98, 0.95, f"Spearman ρ = {stats.spearmanr(x, y)[0]:+.2f} (p={stats.spearmanr(x, y)[1]:.2f})", transform=ax.transAxes, ha="right", va="top", color=INK2)
ax.set_title("그림 17a. 사업이 인구를 끌어들였나? (사업 → 이후 인구증가)")
ax = axs[1]
x2 = D.g_10_15 * 100; y2 = D.n_dev_1625 / D.cpop2015 * 1e4
ax.scatter(x2, y2, s=np.sqrt(D.cpop2015) / 5, color=S2, alpha=0.75, edgecolor=SURF, linewidth=1)
ax.set_xlabel("2010→2015 연평균 인구증가율 (%)"); ax.set_ylabel("2016-25년 시작 개발사업 / 2015년 인구 1만")
ax.text(0.98, 0.95, f"Spearman ρ = {stats.spearmanr(x2, y2)[0]:+.2f} (p={stats.spearmanr(x2, y2)[1]:.2f})", transform=ax.transAxes, ha="right", va="top", color=INK2)
ax.set_title("그림 17b. 인구가 사업을 끌어들였나? (인구증가 → 이후 사업)")
fig.tight_layout(); fig.savefig(FIG / "fig17_reverse_causality.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F18 sensitivity by category / ICT / size / duration ----------
S = pd.read_csv(TAB / "ext_D_sensitivity.csv")
fam_lab = {"category": "부문", "ict": "ICT", "size": "규모", "duration": "기간"}
fig, axs = plt.subplots(1, 2, figsize=(14, 8.4), sharey=True)
yl = list(S.group)
ypos = np.arange(len(S))
for ax, (b, se, col, t) in zip(axs, [("b_phys", "se_phys", S1, "도로·물리 접근성 (1 SD)"), ("b_net", "se_net", S3, "네트워크 접근성 (1 SD)")]):
    sig = S[b.replace("b_", "p_")] < 0.05
    ax.errorbar(S[b], ypos, xerr=1.96 * S[se], fmt="none", ecolor=col, lw=2, alpha=0.6)
    ax.scatter(S[b], ypos, color=np.where(sig, col, SURF), edgecolor=col, s=55, zorder=3, linewidth=1.8)
    ax.axvline(0, color=AXIS); ax.set_title(t); ax.set_xlim(-1.4, 1.1)
    prev = None
    for i, f in enumerate(S.family):
        if f != prev and i > 0:
            ax.axhline(i - 0.5, color=GRID, lw=1.2)
        prev = f
axs[0].set_yticks(ypos); axs[0].set_yticklabels([f"{g}  (n={int(n)})" for g, n in zip(S.group, S.total)]); axs[0].invert_yaxis()
fig.suptitle("그림 18. 사업 유형별 접근성 민감도 (PPML, 딜리 제외, 도로·네트워크 동시 투입, 채운 점 = p<0.05)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "fig18_sensitivity_types.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F19 ICT concentration in Dili ----------
dili = U[U.unit == "TL06U"].iloc[0]
vals = {"인구": dili.pop2022 / U.pop2022.sum(), "비ICT 개발사업": dili.n_dev_nonict / U.n_dev_nonict.sum(), "ICT 요소 개발사업": dili.n_dev_ict / U.n_dev_ict.sum()}
dev = PR[PR.cat.isin(["SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"])]
cen = dev.groupby("ict").target_type.apply(lambda s: (s == "central_institution").mean())
a, b = U.n_dev_ict.sum(), U.n_dev_nonict.sum()
tab = np.array([[dili.n_dev_ict, a - dili.n_dev_ict], [dili.n_dev_nonict, b - dili.n_dev_nonict]])
chi = stats.chi2_contingency(np.round(tab))
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(list(vals), [v * 100 for v in vals.values()], color=[MUTED, S1, S3], width=0.55)
for i, v in enumerate(vals.values()):
    ax.text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center", color=INK2)
ax.set_ylabel("딜리(도심) 점유율 %"); ax.grid(axis="x", visible=False)
ax.set_title("그림 19. 지역 특정 개발사업 중 딜리 점유율: ICT 요소 사업이 수도에 더 집중")
ax.text(0.99, 0.95, f"χ² p = {chi[1]:.3f}", transform=ax.transAxes, ha="right", va="top", color=INK2)
fig.savefig(FIG / "fig19_ict_dili.png", dpi=200, bbox_inches="tight"); plt.close(fig)
pd.DataFrame({"share": vals, "chi2_p": chi[1]}).to_csv(TAB / "ext_D_ict_dili.csv", encoding="utf-8-sig")

# ---------- F20 road upgrades ----------
E = pd.read_csv(TAB / "ext_C_event_bb5.csv")
T = pd.read_csv(TAB / "ext_C_db_periods.csv", header=[0, 1], index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axs[0]
ax.errorbar(E.rel, E.coef, yerr=1.96 * E.se, fmt="-o", color=S1, lw=2, ms=6)
ax.axhline(0, color=AXIS); ax.axvline(-0.5, color=INK2, ls=":")
ax.set_xlabel("도로 개량 완공 기준 예산서 판(edition) 상대 연도"); ax.set_ylabel("PPML 계수 (지자체·판 고정효과)")
ax.set_title("그림 20a. 예산서 일관 패널: 개량 지자체의 비인프라 사업 언급")
ax = axs[1]
proj = T["projects"]; share = proj.div(proj.sum(0), axis=1) * 100
usd = T["usd_m"]; ushare = usd.div(usd.sum(0), axis=1) * 100
cols = {"East (Baucau+Lautém postos)": S2, "Other upgraded postos": S1, "Not upgraded postos": MUTED}
kor = {"East (Baucau+Lautém postos)": "동부(바우카우·라우템)", "Other upgraded postos": "기타 개량 지역", "Not upgraded postos": "비개량 지역"}
for g in share.index:
    ax.plot(share.columns, share.loc[g], "-o", color=cols[g], lw=2, label=f"{kor[g]} 사업 수")
    ax.plot(ushare.columns, ushare.loc[g], "--", color=cols[g], lw=1.4, label=f"{kor[g]} 금액")
ax.axvspan(2.5, 3.5, color=GRID, alpha=0.6, lw=0); ax.text(2.55, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 50, "동부 회랑 완공\n(2020-21)", fontsize=8, color=INK2, va="top")
ax.set_ylabel("전체(딜리 제외) 대비 점유율 %"); ax.legend(fontsize=7.5, ncol=2, loc="upper left")
ax.set_title("그림 20b. 시기별 사업 수·금액 점유율 (우리 DB)")
fig.tight_layout(); fig.savefig(FIG / "fig20_road_upgrades.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- F21 crowd-in ----------
Ecr = pd.read_csv(TAB / "ext_E_crowd_in.csv")
Ecr = Ecr[Ecr.model.str.startswith(("E3", "E4"))]
fig, ax = plt.subplots(figsize=(9, 3.6))
lab = {"any_access_l1_3": "인프라 사업 시작 후 1-3년 (lag)", "any_access_f1_3": "인프라 사업 시작 1-3년 전 (lead, 위약)"}
for i, r in Ecr.reset_index(drop=True).iterrows():
    col = S1 if "l1" in r.term else MUTED
    ax.errorbar(r.coef, i, xerr=1.96 * r.se, fmt="o", color=col, ms=7, lw=2)
ax.set_yticks(range(len(Ecr))); ax.set_yticklabels([f"{lab[t]} — {'2006-25' if m.startswith('E3') else 'AMP 2006-15'}" for t, m in zip(Ecr.term, Ecr.model)], fontsize=8.5)
ax.axvline(0, color=AXIS); ax.invert_yaxis(); ax.set_xlabel("PPML 계수 (포스토·연도 고정효과), 신규 비인프라 개발사업")
ax.set_title("그림 21. 인프라 사업 이후 다른 사업이 따라오는가?")
fig.savefig(FIG / "fig21_crowd_in.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("chi2 ICT Dili p =", chi[1], vals, "central share by ict:", cen.to_dict())
