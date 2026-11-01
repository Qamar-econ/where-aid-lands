"""Nepal 2019-2024 (planning year): which district characteristics predict new development projects, by category, size,
duration and donor class?  Same formula as the first study, now time-aligned:

  E[N^c_{d,t}] = exp( b_N Net_{d,t} + b_R Road_d + b_P Paved_d + b_D Dep_d + theta lnPop_{d,t} + kappa Cap_d + g_{p(d),t} )

  c = subgroup (category / size / duration / donor class); d = district (75); t = planning year 2019-2024; g = province x year FE.
  All regressors standardised (1 SD); PPML; SE clustered by district; BH-FDR within each subgroup family.
Also: simple bivariate version (one regressor + lnPop + province x year FE) and an intuitive concentration table
(share of projects vs share of population by district quintile of each characteristic).
Outputs: outputs/tables/np2_heterogeneity.csv, np2_concentration.csv, outputs/figures/np2_fig3_effect_matrix.png, np2_fig4_concentration.png
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PL = pd.read_csv(ROOT / "data/processed/nepal/np2_planning.csv")
ST = pd.read_csv(ROOT / "data/processed/nepal/np2_project_status.csv")
G = pd.read_csv(ROOT / "data/processed/nepal/np2_geocodes.csv")
INV = pd.read_csv(ROOT / "data/interim/np2_inventory.csv").set_index("uid")
B = pd.read_csv(ROOT / "data/processed/nepal/np2_panel.csv")
D = pd.read_csv(ROOT / "data/processed/nepal/districts.csv")
B = B.merge(D[["k", "density"]], on="k")
z = lambda s: (s - s.mean()) / s.std()
B["dens"] = z(np.log(B.density))
IV = {"net_idx": "Connectivity", "road": "Road access", "paved": "Paved-road density", "dep": "Deprivation"}
# population density is not used as a regressor: given ln pop it is minus ln area (r = 0.89 with ln pop, VIF 9).
# Instead the population elasticity theta is reported: theta < 1 means projects grow less than proportionally with population.

prj = PL.merge(ST, on="uid")
prj = prj[(prj.best_precision <= 4) & prj.planning_year.between(2019, 2024)].copy()
prj["cat"] = prj.uid.map(INV["cat"])
prj["duration"] = prj.uid.map(INV["duration"])
CATN = {"SOC_education": "Education", "SOC_health": "Health", "SOC_water": "Water & sanitation", "GOV": "Governance",
        "PROD": "Agriculture & production", "ENV_MULTI": "Environment & multisector", "SOC_other": "Other social"}
prj["category"] = prj["cat"].map(CATN)
prj["size"] = pd.cut(prj.usd, [-1, 1e6, 1e7, np.inf], labels=["< $1M", "$1-10M", "> $10M"]).astype(str)
prj.loc[prj.usd.isna(), "size"] = "unknown"
prj["dur"] = pd.cut(prj.duration, [-1, 2, 5, 100], labels=["<= 2 y", "2-5 y", "> 5 y"]).astype(str)
asg = G[G.uid.isin(prj.uid)].groupby(["uid", "district"]).size().reset_index()[["uid", "district"]]
asg = asg.merge(prj[["uid", "planning_year", "n_districts", "category", "size", "dur", "dclass"]], on="uid")
asg["w"] = 1 / asg.n_districts

GROUPS = [("All", None, None)] + [("Category", "category", v) for v in CATN.values()] + \
         [("Size", "size", v) for v in ["< $1M", "$1-10M", "> $10M"]] + [("Duration", "dur", v) for v in ["<= 2 y", "2-5 y", "> 5 y"]] + \
         [("Donor", "dclass", v) for v in ["MDB", "bilateral", "UN-other"]]


def panel(a):
    y = a.groupby(["district", "planning_year"]).w.sum().rename("N").reset_index().rename(columns={"district": "k", "planning_year": "year"})
    p = B.drop(columns="N").merge(y, on=["k", "year"], how="left")
    p["N"] = p.N.fillna(0)
    return p


def fit(f, p):
    return smf.glm(f, data=p, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.k)[0]}, maxiter=300)


rows = []
for fam, col, val in GROUPS:
    a = asg if col is None else asg[asg[col] == val]
    nproj = a.uid.nunique()
    if nproj < 25:
        rows.append({"family": fam, "group": val or "All", "n_projects": nproj, "model": "multi", "iv": None}); continue
    p = panel(a)
    try:
        m = fit("N ~ " + " + ".join(IV) + " + lnpop + capital + C(pt)", p)
        for v in IV:
            rows.append({"family": fam, "group": val or "All", "n_projects": nproj, "model": "multi", "iv": v, "b": m.params[v], "se": m.bse[v], "p": m.pvalues[v]})
        th, se = m.params["lnpop"], m.bse["lnpop"]
        from scipy.stats import norm
        rows.append({"family": fam, "group": val or "All", "n_projects": nproj, "model": "multi", "iv": "theta_pop", "b": th, "se": se,
                     "p": 2 * norm.sf(abs((th - 1) / se))})  # H0: theta = 1 (proportional to population)
    except Exception as e:
        print(fam, val, "multi failed", e)
    for v in IV:
        try:
            m = fit(f"N ~ {v} + lnpop + capital + C(pt)", p)
            rows.append({"family": fam, "group": val or "All", "n_projects": nproj, "model": "simple", "iv": v, "b": m.params[v], "se": m.bse[v], "p": m.pvalues[v]})
        except Exception:
            pass
R = pd.DataFrame(rows).dropna(subset=["iv"])
R["pct"] = np.where(R.iv == "theta_pop", np.nan, 100 * (np.exp(R.b) - 1))
# BH within (family, model, iv)
def bh(p):
    p = p.values; n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q
R["q_bh"] = R.groupby(["family", "model", "iv"]).p.transform(lambda s: pd.Series(bh(s), index=s.index))
(ROOT / "outputs/tables").mkdir(exist_ok=True)
R.to_csv(ROOT / "outputs/tables/np2_heterogeneity.csv", index=False)

# ---- concentration: share of projects vs population by quintile (pooled 2019-2024, equal-split counts)
cs = B.groupby("k").agg(pop=("pop", "mean"), **{v: (v, "mean") for v in IV}).reset_index()
CONC = []
for fam, col, val in GROUPS:
    a = asg if col is None else asg[asg[col] == val]
    if a.uid.nunique() < 25:
        continue
    n = a.groupby("district").w.sum()
    t = cs.assign(N=cs.k.map(n).fillna(0))
    for v in IV:
        t["q"] = pd.qcut(t[v].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        s = t.groupby("q").agg(N=("N", "sum"), pop=("pop", "sum"))
        CONC.append({"family": fam, "group": val or "All", "iv": v, **{f"Q{q}": round(100 * s.N[q] / s.N.sum() / (s["pop"][q] / s["pop"].sum()), 0) for q in s.index}})
CONC = pd.DataFrame(CONC)
CONC.to_csv(ROOT / "outputs/tables/np2_concentration.csv", index=False)

# ---- Figure 3: effect matrix (multi model), Fairbairn-style
TH = R[(R.model == "multi") & (R.iv == "theta_pop")].copy()
M = R[(R.model == "multi") & (R.iv != "theta_pop")].copy()
M["label"] = M.family.where(M.family == "All", M.family + ": ") .where(M.family != "All", "") + M.group + " (n=" + M.n_projects.astype(str) + ")"
order = [l for l in M.label.drop_duplicates()]
piv = M.pivot_table(index="label", columns="iv", values="pct", sort=False).reindex(order)[list(IV)]
qv = M.pivot_table(index="label", columns="iv", values="q_bh", sort=False).reindex(order)[list(IV)]
pv = M.pivot_table(index="label", columns="iv", values="p", sort=False).reindex(order)[list(IV)]
val = np.log1p(piv.clip(-90, 300) / 100)  # symmetric in log space
fig, (ax, axt) = plt.subplots(1, 2, figsize=(7.6, 0.33 * len(piv) + 1.6), dpi=200, gridspec_kw={"width_ratios": [len(IV), 1.1], "wspace": 0.04})
lim = np.nanmax(np.abs(val.values)); lim = min(lim, np.log(3))
im = ax.imshow(val.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        x = piv.iat[i, j]
        if np.isnan(x):
            continue
        sig = pv.iat[i, j] < 0.05
        arrow = ("▲" if x > 0 else "▼") if sig else ""
        star = "*" if qv.iat[i, j] < 0.10 else ""
        ax.text(j, i, f"{x:+.0f}%{arrow}{star}", ha="center", va="center", fontsize=6.3,
                color="white" if abs(val.iat[i, j]) > lim * 0.6 else "#1f2328", fontweight="bold" if sig else "normal")
ax.set_xticks(range(len(IV))); ax.set_xticklabels([IV[v] for v in IV], fontsize=7, color="#1f2328")
ax.xaxis.tick_top()
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6.6, color="#1f2328")
for y in [0.5] + [i - 0.5 for i in range(1, len(piv)) if piv.index[i].split(":")[0] != piv.index[i - 1].split(":")[0]]:
    ax.axhline(y, color="white", lw=2.5)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
TH["label"] = TH.family.where(TH.family == "All", TH.family + ": ").where(TH.family != "All", "") + TH.group + " (n=" + TH.n_projects.astype(str) + ")"
th = TH.set_index("label").reindex(order)
axt.set_xlim(-0.5, 0.5); axt.set_ylim(len(piv) - 0.5, -0.5); axt.axis("off")
axt.text(0, -0.62, "Pop.\nelasticity θ", ha="center", va="bottom", fontsize=7, color="#1f2328")
for i, (b_, p_) in enumerate(zip(th.b, th.p)):
    if not np.isnan(b_):
        axt.text(0, i, f"{b_:.2f}" + ("‡" if p_ < 0.05 else ""), ha="center", va="center", fontsize=6.5, color="#1f2328")
cb = fig.colorbar(im, ax=[ax, axt], fraction=0.03, pad=0.02)
cb.set_ticks([np.log(0.5), 0, np.log(2)]); cb.set_ticklabels(["−50%", "0", "+100%"]); cb.ax.tick_params(labelsize=6)
fig.text(0.01, 0.01, "Change in new projects per +1 SD, all four characteristics jointly (PPML, province×year FE, ln pop, capital). "
         "▲▼ p<0.05; * BH q<0.10 within family. θ: ‡ differs from 1 (p<0.05).", fontsize=5.6, color="#6e7781")
fig.suptitle("Nepal 2019–2024 · what predicts where new development projects are planned", fontsize=8.5, color="#1f2328", x=0.02, ha="left", y=0.985)
fig.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.06); fig.savefig(ROOT / "outputs/figures/np2_fig3_effect_matrix.png"); plt.close(fig)

# ---- Figure 4: concentration small multiples (All projects): projects per capita relative to national, by quintile
A0 = CONC[CONC.group == "All"].set_index("iv")
fig, axs = plt.subplots(1, len(IV), figsize=(7, 2.1), dpi=200, sharey=True)
for ax, v in zip(axs, IV):
    yv = A0.loc[v, ["Q1", "Q2", "Q3", "Q4", "Q5"]].values.astype(float)
    ax.bar(range(1, 6), yv, color=["#1f6feb" if x >= 100 else "#8c959f" for x in yv], width=0.7)
    ax.axhline(100, color="#1f2328", lw=0.8, ls="--")
    ax.set_title(IV[v], fontsize=7, color="#1f2328"); ax.set_xticks(range(1, 6)); ax.set_xticklabels(["Q1\nlow", "2", "3", "4", "Q5\nhigh"], fontsize=5.8)
    ax.tick_params(axis="y", labelsize=6, colors="#6e7781")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("Projects per capita\n(national = 100)", fontsize=6.5)
fig.suptitle("Where do new projects go? District quintiles of each characteristic, 2019–2024 planning years", fontsize=7.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/np2_fig4_concentration.png"); plt.close(fig)

pd.set_option("display.width", 200)
S = M.assign(cell=lambda d: d.pct.round(0).astype(int).astype(str) + "%" + np.where(d.p < 0.05, "†", "")).pivot_table(index="label", columns="iv", values="cell", aggfunc="first", sort=False).reindex(order)[list(IV)]
print(S.join(TH.set_index("label")[["b", "p"]].round(2).rename(columns={"b": "theta", "p": "p(theta=1)"})).to_string())
print("\nconcentration (All; projects per capita, national = 100):\n", A0[["Q1", "Q2", "Q3", "Q4", "Q5"]].to_string())

# ---- formal gradient test: stacked size groups, group-specific FE and slopes; road slope linear in size rank (1, 2, 3)
st = []
for r_, s_ in enumerate(["< $1M", "$1-10M", "> $10M"], 1):
    p = panel(asg[asg["size"] == s_]); p["grp"] = r_; p["gpt"] = p.pt + f"_{r_}"
    st.append(p)
st = pd.concat(st)
f = "N ~ " + " + ".join(f"{v} + {v}:grp" for v in IV) + " + lnpop + lnpop:grp + capital + C(gpt)"
mg = smf.glm(f, data=st, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(st.k)[0]}, maxiter=300)
lines = [f"size-gradient (per size step) {v}:grp {mg.params[v + ':grp']:+.3f} (se {mg.bse[v + ':grp']:.3f}, p {mg.pvalues[v + ':grp']:.3f})" for v in list(IV) + ["lnpop"]]
st2 = []
for r_, s_ in enumerate(["<= 2 y", "2-5 y", "> 5 y"], 1):
    p = panel(asg[asg["dur"] == s_]); p["grp"] = r_; p["gpt"] = p.pt + f"_{r_}"
    st2.append(p)
st2 = pd.concat(st2)
md = smf.glm(f, data=st2, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(st2.k)[0]}, maxiter=300)
lines += [f"duration-gradient (per step) {v}:grp {md.params[v + ':grp']:+.3f} (se {md.bse[v + ':grp']:.3f}, p {md.pvalues[v + ':grp']:.3f})" for v in list(IV) + ["lnpop"]]
(ROOT / "outputs/results_np2_gradient.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
