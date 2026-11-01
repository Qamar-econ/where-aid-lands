"""Paper figures (Nepal 2010-2025 + Timor-Leste comparison).
F1 map: development projects per 100k people and physical-access index (Nepal districts)
F2 location coverage by start year (share of projects / USD located, central-institution, national-unresolved)
F3 main coefficients: each construct alone vs jointly, Nepal and Timor-Leste (% per SD, 95% CI)
F4 population scaling: log projects vs log population with fitted slope (theta) and the proportional line, both countries
F5 heterogeneity matrix (Nepal, joint model) with theta column
F6 year-by-year coefficients (Nepal)
"""
import numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "outputs/tables", ROOT / "outputs/figures/paper"; F.mkdir(parents=True, exist_ok=True)
INK, MUTED, BLUE, GREY, ORANGE = "#1f2328", "#6e7781", "#1f6feb", "#8c959f", "#d4600f"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
LAB = {"access": "Physical access", "conn": "Connectivity", "need": "Need (deprivation)"}


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


# ---- F1 map
P = pd.read_csv(ROOT / "data/processed/nepal/np3_projects.csv"); AS = pd.read_csv(ROOT / "data/processed/nepal/np3_assign.csv")
B = pd.read_csv(ROOT / "data/processed/nepal/np3_panel.csv")
Dg = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")
dev = AS.merge(P[["uid", "family"]], on="uid").query("family == 'development'")
n = dev.groupby("k").w.sum()
pop = B[B.year == 2016].set_index("k").lnpop.pipe(np.exp)
Dg["pc"] = Dg.k.map(n).fillna(0) / Dg.k.map(pop) * 1e5
Dg["acc"] = Dg.k.map(B.drop_duplicates("k").set_index("k").access)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=220)
for ax, col, cmap, title in [(axs[0], "pc", "Blues", "Development projects per 100,000 people, 2010–2025"), (axs[1], "acc", "Greens", "Physical-access index (SD)")]:
    Dg.plot(column=col, cmap=cmap, ax=ax, edgecolor="white", linewidth=0.3, legend=True, legend_kwds={"shrink": 0.6})
    ax.set_title(title, fontsize=7.5, color=INK, loc="left"); ax.axis("off")
fig.tight_layout(); fig.savefig(F / "F1_map.png"); plt.close(fig)

# ---- F2 coverage
S = pd.read_csv(T / "np3_portfolio_status.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), dpi=220, sharey=True)
cols = {"located": BLUE, "central_institution": ORANGE, "national_unresolved": "#d0d7de"}
for ax, v, lab in [(axs[0], "n", "Share of projects"), (axs[1], "usd", "Share of USD")]:
    pv = S.pivot(index="start_year", columns="status", values=v).fillna(0); pv = pv.div(pv.sum(1), axis=0)
    bottom = np.zeros(len(pv))
    for c in ["located", "central_institution", "national_unresolved"]:
        ax.bar(pv.index, pv[c], bottom=bottom, color=cols[c], width=0.8, label=c.replace("_", " ")); bottom += pv[c].values
    ax.set_title(lab, fontsize=7.5, loc="left", color=INK); clean(ax)
axs[0].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
fig.tight_layout(); fig.savefig(F / "F2_coverage.png"); plt.close(fig)

# ---- F3 main coefficients
NP = pd.read_csv(T / "np3_table2_main.csv"); TL = pd.read_csv(T / "tl3_table_main.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), dpi=220, sharex=True)
for ax, v in zip(axs, ["access", "conn", "need"]):
    yy = 0; ticks = []
    for cname, df, col in [("Nepal", NP, BLUE), ("Timor-Leste", TL, ORANGE)]:
        for spec, lab in [(f"({['access', 'conn', 'need'].index(v) + 1}) {v} only", "alone"), ("(4) joint", "joint")]:
            r = df[df.spec == spec].iloc[0]
            b, se = r[v + "_b"], r[v + "_se"]
            lo, hi, pt = [100 * (np.exp(x) - 1) for x in (b - 1.96 * se, b + 1.96 * se, b)]
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.6, alpha=1 if lab == "alone" else 0.45)
            ax.plot(pt, yy, "o", color=col, ms=4, mec="white", mew=0.8, alpha=1 if lab == "alone" else 0.45)
            ticks.append(f"{cname} · {lab}"); yy += 1
        yy += 0.6
    ax.axvline(0, color=MUTED, lw=0.7); ax.set_yticks([0, 1, 2.6, 3.6]); ax.set_yticklabels(ticks if v == "access" else [""] * 4, fontsize=6.3)
    ax.invert_yaxis(); ax.set_title(LAB[v], fontsize=7.5, loc="left", color=INK); clean(ax)
axs[1].set_xlabel("Change in new development projects per +1 SD (%), 95% CI")
fig.tight_layout(); fig.savefig(F / "F3_main.png"); plt.close(fig)

# ---- F4 population scaling (district/posto totals over 2010-2025, development)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=220)
tl_as = None
try:
    import importlib.util
except Exception:
    pass
for ax, (name, units, col) in zip(axs, [("Nepal (75 districts)", None, BLUE), ("Timor-Leste (61 postos)", None, ORANGE)]):
    if name.startswith("Nepal"):
        x = np.log(Dg.k.map(pop)); y = np.log1p(Dg.k.map(n).fillna(0))
        th = NP[NP.spec == "(0) population only"].lnpop_b.iloc[0]
    else:
        tl = pd.read_csv(T / "tl3_units_totals.csv")
        x = np.log(tl["pop"]); y = np.log1p(tl.N)
        th = TL[TL.spec == "(0) population only"].lnpop_b.iloc[0]
    ax.scatter(x, y, s=9, color=col, alpha=0.75, edgecolor="white", linewidth=0.3)
    xs = np.linspace(x.min(), x.max(), 50)
    b1 = np.polyfit(x, y, 1)
    ax.plot(xs, np.polyval(b1, xs), color=col, lw=1.4, label=f"fitted slope {b1[0]:.2f} (PPML θ = {th:.2f})")
    ax.plot(xs, y.mean() + 1.0 * (xs - x.mean()), color=INK, lw=0.9, ls="--", label="proportional (slope 1)")
    ax.set_xlabel("ln population"); ax.set_ylabel("ln (1 + development projects)"); ax.set_title(name, fontsize=7.5, loc="left", color=INK)
    ax.legend(fontsize=5.8, frameon=False); clean(ax)
fig.tight_layout(); fig.savefig(F / "F4_population_scaling.png"); plt.close(fig)

# ---- F5 heterogeneity matrix
Hh = pd.read_csv(T / "np3_heterogeneity.csv")
Hh["label"] = Hh.group + " (n=" + Hh.n_projects.astype(str) + ")"
vals = Hh[[f"{v}_pct" for v in LAB]].values; ps = Hh[[f"{v}_p" for v in LAB]].values
lv = np.log1p(np.clip(vals, -90, 300) / 100); lim = np.log(2.5)
fig, (ax, axt) = plt.subplots(1, 2, figsize=(6.6, 0.27 * len(Hh) + 1.2), dpi=220, gridspec_kw={"width_ratios": [3, 0.8], "wspace": 0.03})
ax.imshow(lv, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(len(Hh)):
    for j in range(3):
        if np.isnan(vals[i, j]):
            continue
        s = ps[i, j] < 0.05
        ax.text(j, i, f"{vals[i, j]:+.0f}%" + ("▲" if s and vals[i, j] > 0 else "▼" if s else ""), ha="center", va="center", fontsize=6,
                fontweight="bold" if s else "normal", color="white" if abs(lv[i, j]) > lim * 0.6 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(list(LAB.values()), fontsize=6.8); ax.xaxis.tick_top()
ax.set_yticks(range(len(Hh))); ax.set_yticklabels(Hh.label, fontsize=6.3); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
axt.set_xlim(-0.5, 0.5); axt.set_ylim(len(Hh) - 0.5, -0.5); axt.axis("off")
axt.text(0, -0.75, "θ (pop.\nelasticity)", ha="center", va="bottom", fontsize=6.5, color=INK)
for i, (t_, p_) in enumerate(zip(Hh.lnpop_b, Hh.theta_p_eq1)):
    axt.text(0, i, f"{t_:.2f}" + ("‡" if p_ < 0.05 else ""), ha="center", va="center", fontsize=6.3, color=INK)
fig.text(0.01, 0.005, "Each construct in its own PPML (+ ln pop, province×year FE); % change per +1 SD. ▲▼ p<0.05. ‡ θ differs from 1 (p<0.05).", fontsize=5.6, color=MUTED)
fig.subplots_adjust(left=0.36, right=0.98, top=0.9, bottom=0.05); fig.savefig(F / "F5_heterogeneity.png"); plt.close(fig)

# ---- F6 yearly coefficients
Y = pd.read_csv(T / "np3_yearly.csv")
fig, axs = plt.subplots(1, 4, figsize=(7.4, 1.9), dpi=220, sharex=True)
for ax, v, lab in zip(axs, ["access", "conn", "need", "lnpop"], list(LAB.values()) + ["θ (population elasticity)"]):
    b, se = Y[v + "_b"], Y[v + "_se"]
    ax.fill_between(Y.year, b - 1.96 * se, b + 1.96 * se, color=BLUE, alpha=0.15, lw=0); ax.plot(Y.year, b, color=BLUE, lw=1.3)
    ax.axhline(1 if v == "lnpop" else 0, color=INK, lw=0.7, ls="--"); ax.set_title(lab, fontsize=7, loc="left", color=INK); clean(ax)
    ax.set_ylim(-1.5, 2.5) if v != "lnpop" else ax.set_ylim(-0.5, 2)
fig.tight_layout(); fig.savefig(F / "F6_yearly.png"); plt.close(fig)
print("figures:", sorted(p.name for p in F.glob("*.png")))
