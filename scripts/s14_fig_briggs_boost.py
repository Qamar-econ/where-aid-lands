"""Figure B1: Briggs replication + network extension (Africa), and lights/network/road across the four settings."""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
mpl.rcParams.update({"font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False})
B = pd.read_csv(TAB / "briggs_models.csv").set_index("spec")
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axs[0]
rows = [("B1 Briggs replication: poverty", "poor", "빈곤 (Briggs 재현)", S2), ("B2 + mobile phone ownership", "poor", "빈곤 | + 휴대폰", S2),
        ("B4 night lights only", "ntl", "야간 불빛", MUTED), ("B5 night lights + mobile", "ntl", "야간 불빛 | + 휴대폰", MUTED),
        ("B2 + mobile phone ownership", "mob", "휴대폰 보유 (빈곤 통제)", S3), ("B5 night lights + mobile", "mob", "휴대폰 보유 (불빛 통제)", S3)]
for i, (spec, v, lab, col) in enumerate(rows):
    r = B.loc[spec]
    ax.errorbar(r[f"b_{v}"], i, xerr=1.96 * r[f"se_{v}"], fmt="o", color=col, ms=8, lw=2.2, mfc=col if "|" not in lab else SURF, mew=2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows]); ax.invert_yaxis(); ax.axvline(0, color=AXIS)
ax.set_xlabel("PPML 계수 (1 SD, 국가 고정효과, 국가 군집 95% CI)")
ax.set_title("그림 B1a. Briggs(2017) 재현: 아프리카 25개국 DHS 지역 251개\n속 빈 점 = 휴대폰 보유율을 추가했을 때")
ax = axs[1]
lb = pd.read_csv(TAB / "boost_lights.csv"); pk = pd.read_csv(TAB / "nepal_palika_models.csv").set_index("model")
items = [("아프리카\n(Briggs, DHS 지역)", B.loc["B5 night lights + mobile", "b_ntl"], B.loc["B5 night lights + mobile", "b_mob"]),
         ("동티모르\n(포스토, 2015 불빛)", lb.query("country=='Timor-Leste' and lights_year==2015 and model=='lights + network'").b_lights.iloc[0],
          lb.query("country=='Timor-Leste' and lights_year==2015 and model=='lights + network'").b_net.iloc[0]),
         ("네팔 군\n(2015 불빛)", lb.query("country=='Nepal' and lights_year==2015 and model=='lights + network'").b_lights.iloc[0],
          lb.query("country=='Nepal' and lights_year==2015 and model=='lights + network'").b_net.iloc[0]),
         ("네팔 palika\n(군 고정효과)", pk.loc["lights + network", "b_ntl"], pk.loc["lights + network", "b_net"])]
x = np.arange(len(items)); w = 0.36
ax.bar(x - w / 2 - 0.01, [i[1] for i in items], w, color=MUTED, label="야간 불빛")
ax.bar(x + w / 2 + 0.01, [i[2] for i in items], w, color=S3, label="통신망")
for xi, it in zip(x, items):
    ax.text(xi + w / 2, it[2] + 0.01, f"{it[2]:+.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.text(xi - w / 2, (it[1] + 0.01) if it[1] >= 0 else (it[1] - 0.04), f"{it[1]:+.2f}", ha="center", fontsize=8.5, color=INK2)
ax.axhline(0, color=AXIS); ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items]); ax.grid(axis="x", visible=False)
ax.set_ylabel("계수 (불빛과 통신망 동시 투입, 1 SD)"); ax.legend(loc="lower left")
ax.set_title("그림 B1b. 네 가지 설정에서 불빛 vs 통신망 경쟁")
fig.tight_layout(); fig.savefig(FIG / "fig_B1_briggs_boost.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
