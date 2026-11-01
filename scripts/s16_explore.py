"""Exploratory time-aligned models (plan v2 §4.1) + Table 1 + Fairbairn-style figures.

E[Y_iw] = exp(b_N Net + b_R Road + b_L Lights + b_D Dep + theta ln Pop + gamma_g)   (PPML, cross-section per window)
  Nepal W1 (2009-13): Net = census 2011 household mobile/internet index, gamma = province FE, 72 districts (excl. valley)
  Timor-Leste W3 (2019-25): Net = Ookla index (secondary measure), gamma = municipality FE, 61 postos (excl. Dili urban)
  Timor-Leste W1/W2: Net = DHS municipality mobile ownership (no FE possible; SE clustered by municipality) - descriptive
Outputs: outputs/tables/explore_models.csv, table1_summary.csv/.md, figures fig_E1_smallmultiples.png, fig_E2_effect_matrix.png
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
LOG = open(ROOT / "outputs/results_explore.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()

T = pd.read_csv(ROOT / "data/processed/windows_TL.csv"); N = pd.read_csv(ROOT / "data/processed/nepal/windows_NP.csv")
nw1 = N[(N.window == "W1") & (N.capital == 0)].copy()
nw1 = nw1.assign(Net=z(nw1.net_w), Road=z(nw1.phys_idx), Lights=z(np.log1p(nw1.ntl2010 / nw1.area_km2)), Dep=z(nw1.deprivation), lpop=np.log(nw1.pop_w), fe=nw1.province)
tw3 = T[(T.window == "W3") & (T.unit != "TL06U")].copy()
tw3 = tw3.assign(Net=z(tw3.net_w), Road=z(tw3.access_phys), Lights=z(np.log1p(tw3.ntl_w / tw3.area_km2)), Dep=z(tw3.deprivation), lpop=np.log(tw3.pop_w), fe=tw3.mun_pcode)
SETS = {"네팔 W1 (2009–13, 센서스)": nw1, "동티모르 W3 (2019–25, Ookla)": tw3}
OUTS = {"active": "개발사업 전체", "active_edu": "교육", "active_health": "보건", "active_water": "식수·위생", "active_gov": "거버넌스",
        "active_prod": "생산부문", "active_ict": "ICT 요소"}
TERMS = ["Net", "Road", "Lights", "Dep", "lpop"]
TLAB = {"Net": "통신 접근성", "Road": "도로 접근성", "Lights": "야간 불빛", "Dep": "박탈", "lpop": "인구 (log)"}


def fit(y, d):
    if d[y].sum() < 10:
        return None
    return smf.glm(f"{y} ~ Net + Road + Lights + Dep + lpop + C(fe)", d, family=sm.families.Poisson()).fit(cov_type="HC1")


rows = []
for sname, d in SETS.items():
    say("=" * 90 + f"\n{sname}: N={len(d)}")
    for y, ylab in OUTS.items():
        m = fit(y, d)
        if m is None:
            continue
        for t in TERMS:
            rows.append({"set": sname, "outcome": ylab, "y": y, "term": t, "coef": m.params[t], "se": m.bse[t], "p": m.pvalues[t], "n_proj": d[y].sum()})
        if y == "active":
            say("  primary (active): " + "  ".join(f"{t}={m.params[t]:+.3f}(p={m.pvalues[t]:.3f})" for t in TERMS[:4]))
    for alt in ["new", "active_noimp", "active_all"]:
        m = fit(alt, d)
        say(f"  {alt:<13}: " + "  ".join(f"{t}={m.params[t]:+.3f}(p={m.pvalues[t]:.3f})" for t in TERMS[:4]))
E = pd.DataFrame(rows); E.to_csv(TAB / "explore_models.csv", index=False)

# TL W1/W2 with municipality-level DHS network (descriptive; 13 clusters)
say("=" * 90 + "\nTimor-Leste W1/W2 with DHS municipality mobile ownership (no FE; SE clustered by municipality, 13 clusters)")
for w in ["W1", "W2"]:
    d = T[(T.window == w) & (T.unit != "TL06U")].copy()
    d = d.assign(Net=z(d.net_w), Road=z(d.access_phys), Dep=z(d.deprivation), lpop=np.log(d.pop_w))
    for y in ["active", "new"]:
        m = smf.glm(f"{y} ~ Net + Road + Dep + lpop", d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.mun_pcode)[0]})
        say(f"  {w} {y:<7}: " + "  ".join(f"{t}={m.params[t]:+.3f}(p={m.pvalues[t]:.3f})" for t in ["Net", "Road", "Dep"]))

# ---------------- Table 1 (Fairbairn style) ----------------
VARS = [("active", "진행 중 개발사업 (정밀 위치)", "건"), ("new", "신규 개발사업 (정밀 위치)", "건"), ("pop_w", "인구 (사업 이전 센서스)", "명"),
        ("net_w", "통신 지표 (원값)", ""), ("tt_wet_min", "우기 이동시간 (수도까지)", "분"), ("deprivation", "센서스 박탈지수", "SD"), ("rwi", "상대적 부 지수 (Meta RWI)", "")]
t1 = []
for sname, d, net_desc in [("네팔 W1 · 군 72", nw1, "센서스 2011 휴대폰·인터넷 지수"), ("동티모르 W3 · 포스토 61", tw3, "Ookla 측정 지수")]:
    for v, lab, unit in VARS:
        s = d[v]
        t1.append({"표본": sname, "변수": lab + (f" — {net_desc}" if v == "net_w" else ""), "단위": unit, "평균": s.mean(), "표준편차": s.std(), "최소": s.min(), "최대": s.max()})
T1 = pd.DataFrame(t1); T1.to_csv(TAB / "table1_summary.csv", index=False, encoding="utf-8-sig")
with open(TAB / "table1_summary.md", "w", encoding="utf-8") as f:
    f.write("| 표본 | 변수 | 단위 | 평균 ± 표준편차 | 최소 | 최대 |\n|---|---|---|---|---|---|\n")
    for r in T1.itertuples():
        fm = (lambda x: f"{x:,.0f}") if abs(r.평균) > 100 else (lambda x: f"{x:.2f}")
        f.write(f"| {r.표본} | {r.변수} | {r.단위} | {fm(r.평균)} ± {fm(r.표준편차)} | {fm(r.최소)} | {fm(r.최대)} |\n")

# ---------------- figures ----------------
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SIG, NS = "#1baf7a", "#a9a8a2"
mpl.rcParams.update({"font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
                     "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold"})

# E1: small multiples - network (x) vs projects (y); colour = significant network effect; PPML prediction + 95% CI
fig, axs = plt.subplots(len(SETS), len(OUTS), figsize=(17, 5.6), sharex="row")
for i, (sname, d) in enumerate(SETS.items()):
    for j, (y, ylab) in enumerate(OUTS.items()):
        ax = axs[i, j]; m = fit(y, d)
        if m is None:
            ax.set_axis_off(); continue
        sig = m.pvalues["Net"] < 0.05; col = SIG if sig else NS
        ax.scatter(d.Net, d[y], s=14, color=col, alpha=0.55, edgecolor="none")
        xs = np.linspace(d.Net.min(), d.Net.max(), 60)
        grid = pd.DataFrame({"Net": xs, "Road": 0.0, "Lights": 0.0, "Dep": 0.0, "lpop": d.lpop.mean(), "fe": d.fe.mode()[0]})
        pr = m.get_prediction(grid).summary_frame(alpha=0.05)
        scale = d[y].mean() / m.predict(d.assign(Net=0.0, Road=0.0, Lights=0.0, Dep=0.0, lpop=d.lpop.mean())).mean()   # average over FE
        ax.fill_between(xs, pr.mean_ci_lower * scale, pr.mean_ci_upper * scale, color=col, alpha=0.18, lw=0)
        ax.plot(xs, pr["mean"] * scale, color=col if sig else INK2, lw=1.8, ls="-" if sig else "--")
        ax.set_ylim(bottom=0)
        ax.text(0.03, 0.96, f"b={m.params['Net']:+.2f}\np={m.pvalues['Net']:.3f}", transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
        if i == 0: ax.set_title(ylab)
        if j == 0: ax.set_ylabel(sname.replace(" (", "\n("), fontsize=8.5)
        if i == len(SETS) - 1: ax.set_xlabel("통신 접근성 (SD)")
        ax.tick_params(labelsize=7.5)
fig.suptitle("그림 E1. 통신 접근성과 같은 시점 진행 중 개발사업 수 (점 = 행정단위, 선 = PPML 예측과 95% CI; 초록 = 통신 효과 p<0.05, 회색 = 비유의)",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig_E1_smallmultiples.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# E2: effect matrix (rows = determinants, columns = outcome x country); colour bins of % change per SD; arrows = direction; grey = n.s.
cols = [(s, o) for s in SETS for o in OUTS.values()]
fig, ax = plt.subplots(figsize=(15, 3.9))
bins = [-100, -30, -10, 10, 30, 1000]; cmap = ListedColormap(["#b8302f", "#f2a3a2", "#f0efec", "#9fe0c6", "#138a5f"])
for j, (s, o) in enumerate(cols):
    for i, t in enumerate(TERMS):
        r = E[(E.set == s) & (E.outcome == o) & (E.term == t)]
        if r.empty:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#ffffff", ec=GRID)); continue
        r = r.iloc[0]; pct = (np.exp(r.coef) - 1) * 100
        if r.p < 0.05:
            k = np.digitize([pct], bins)[0] - 1
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(k), ec=SURF, lw=2))
            ax.text(j + 0.5, i + 0.5, "↑" if pct > 0 else "↓", ha="center", va="center", fontsize=12, color=INK if abs(pct) < 30 else "#ffffff")
        else:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#e6e5e0", ec=SURF, lw=2))
ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(TERMS)); ax.invert_yaxis()
ax.set_yticks(np.arange(len(TERMS)) + 0.5); ax.set_yticklabels([TLAB[t] for t in TERMS])
ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels([o for s, o in cols], rotation=40, ha="right")
ax.axvline(len(OUTS), color=INK2, lw=1.2)
for k, s in enumerate(SETS):
    ax.text(k * len(OUTS) + len(OUTS) / 2, -0.35, s, ha="center", fontsize=9.5, fontweight="bold")
ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(5)] + [plt.Rectangle((0, 0), 1, 1, color="#e6e5e0")]
ax.legend(handles, ["≤ −30%", "−30~−10%", "−10~+10%", "+10~+30%", "≥ +30%", "비유의 (p≥0.05)"], title="1 SD당 사업 수 변화", loc="center left",
          bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, title_fontsize=8.5)
fig.suptitle("그림 E2. 요인별 효과 행렬 (PPML, 모든 요인 동시 투입 + 상위 단위 고정효과; 화살표 = 방향)", x=0.01, ha="left", y=1.08, fontsize=10.5, fontweight="bold")
fig.savefig(FIG / "fig_E2_effect_matrix.png", dpi=220, bbox_inches="tight"); plt.close(fig)
LOG.close()
