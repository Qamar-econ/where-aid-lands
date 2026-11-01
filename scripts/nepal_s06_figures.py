"""Nepal figures (N1-N4) and Timor-Leste vs Nepal comparison (C1). Same palette as s06."""
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, TAB, FIG = ROOT / "data/processed/nepal", ROOT / "outputs/tables", ROOT / "outputs/figures"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("o", ["#fde3d6", "#f9c3a8", "#f4a07c", "#eb6834", "#c9501f", "#9e3d16", "#702b0f"])
GREEN = LinearSegmentedColormap.from_list("g", ["#d4f2e6", "#9fe0c6", "#5fc9a0", "#1baf7a", "#138a5f", "#0c6545"])
mpl.rcParams.update({"font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False})
G = gpd.read_file(OUT / "districts.gpkg")[["k", "geometry"]].merge(pd.read_csv(OUT / "district_panel.csv"), on="k")
G["proj_pc"] = G.y_all / G.pop2011 * 1e5
prov = G.dissolve("province")


def mp(ax, col, cmap, title, sub, lab, vmin=None, vmax=None):
    G.plot(column=col, cmap=cmap, ax=ax, edgecolor="#ffffff", linewidth=0.4, vmin=vmin, vmax=vmax)
    prov.boundary.plot(ax=ax, color=INK2, linewidth=0.8)
    G[G.capital == 1].plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=1.6, hatch="///")
    ax.set_axis_off(); ax.set_title(title, pad=20); ax.text(0, 1.01, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin if vmin is not None else G[col].min(), vmax if vmax is not None else G[col].max()))
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40); cb.outline.set_visible(False); cb.set_label(lab, color=INK2, fontsize=8.5)


fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
mp(axs[0, 0], G.assign(h=G.tt_wet_min / 60).h.name if False else "tt_wet_min", BLUE, "그림 N1a. 우기 기준 카트만두까지 실질 이동시간",
   "OSM 도로·산길 1,553만 구간, 노면·경사·몬순 감속 | 빗금 = 카트만두 분지(주 모형 제외)", "분 (우기)")
mp(axs[0, 1], "net_census", GREEN, "그림 N1b. 네트워크 접근성 (2011 센서스 가구 휴대폰·인터넷)", "정부 전수조사 기반 지수 (표준화)", "지수")
mp(axs[1, 0], "deprivation", ORANGE, "그림 N1c. 박탈지수 (2011 센서스)", "개선 식수 미접근, 화장실 없음, 문맹 (표준화 평균)", "지수")
mp(axs[1, 1], "proj_pc", BLUE, "그림 N1d. 인구 10만 명당 지역 특정 개발사업", "2000-2026 시작, 비인프라 개발사업", "사업 / 10만", 0, np.nanpercentile(G.proj_pc, 95))
fig.tight_layout(); fig.savefig(FIG / "nepal_fig01_maps.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# N2 scatter: network vs projects; deprivation vs network
S = G[G.capital == 0]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (x, xl) in zip(axs, [("net_census", "네트워크 지수 (센서스)"), ("phys_idx", "물리(도로) 접근성 지수"), ("deprivation", "박탈지수")]):
    ax.scatter(S[x], S.proj_pc, s=np.sqrt(S.pop2011) / 12, color=S1, alpha=0.7, edgecolor=SURF)
    b = np.polyfit(S[x], S.proj_pc, 1); xs = np.linspace(S[x].min(), S[x].max(), 50); ax.plot(xs, b[0] * xs + b[1], ls="--", color=INK2)
    ax.set_xlabel(xl); ax.set_ylabel("사업 / 인구 10만" if x == "net_census" else "")
    ax.text(0.98, 0.95, f"ρ = {S[[x, 'proj_pc']].corr('spearman').iloc[0, 1]:+.2f}", transform=ax.transAxes, ha="right", va="top", color=INK2)
axs[0].set_title("그림 N2. 네팔 군별 상관 (카트만두 분지 제외 72개, 원 크기 = 인구)")
fig.tight_layout(); fig.savefig(FIG / "nepal_fig02_scatter.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# C1 comparison of primary + robustness: Timor vs Nepal
T = pd.read_csv(TAB / "final_robustness.csv"); N = pd.read_csv(TAB / "nepal_robustness.csv")
pick_t = {"P  PRIMARY": "주 모형", "R1 tier A (exact points) only": "정밀 좌표만", "R3 + pre-period hub: log night lights 2010": "+ 야간 불빛",
          "R5 municipality-clustered SE (12 clusters)": "상위단위 군집 SE", "R10 no municipality FE": "고정효과 없음"}
pick_n = {"P  PRIMARY (census network, province FE)": "주 모형", "R1 tier A only": "정밀 좌표만", "R3 + night lights 2010": "+ 야간 불빛",
          "R5 province-clustered SE (7)": "상위단위 군집 SE", "R10 no FE": "고정효과 없음"}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
for ax, (v, t) in zip(axs, [("net", "네트워크 (1 SD)"), ("phys", "도로·물리 (1 SD)"), ("dep", "박탈 (1 SD)")]):
    for j, (df, pick, col, nm) in enumerate([(T, pick_t, S2, "동티모르 (포스토 61)"), (N, pick_n, S1, "네팔 (군 72)")]):
        sub = df[df.spec.isin(pick)].set_index("spec").loc[list(pick)]
        y = np.arange(len(pick)) + (j - 0.5) * 0.3
        ax.errorbar(sub[f"b_{v}"], y, xerr=1.96 * sub[f"se_{v}"], fmt="o", color=col, ms=6, lw=2, label=nm)
    ax.axvline(0, color=AXIS); ax.set_title(t); ax.set_yticks(range(len(pick_t))); ax.set_yticklabels(list(pick_t.values()))
axs[0].invert_yaxis(); axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("그림 C1. 두 나라 주 모형·강건성 비교 (PPML, 95% CI)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "compare_fig01_coefficients.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
