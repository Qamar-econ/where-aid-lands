"""Major national roads (ADB, Japan; kh_s32) and later social-sector aid: checks of the kh_s31 finding (exploratory).

Outcome: new social-sector activities (health, education, water, governance; OECD 110-169) per district-year, 2000-2020,
from all district-located GODAD activities. PPML with district and year fixed effects, errors clustered by district.
  A  main: major-road start in t-1..t-5 (sum = five-year response) and t+1, t+2 (placebo), with and without GODAD's own
     transport/energy activities as a control
  B  event time: k = -5..+6 around a district's first major-road start (k = -1 omitted, ends binned), for the figure
  C  randomisation inference: the eleven projects' start years are shuffled among the projects (district sets kept),
     999 draws; p = share of draws with a five-year response at least as large as observed
  D  timing by completion or opening year instead of approval (projects with a documented end year)
  E  leave one project out
Figures: event-time plot; average social projects per district before and after the road (treated vs other districts);
case series for the NR56 districts.
Output: outputs/tables/kh_major_roads_rigor.txt, kh_major_roads_event.csv; outputs/figures/cambodia/S6_*.png
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from scipy.stats import norm
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, ROOT = ns["DF"], ns["L"], ns["ROOT"]
T, PROCD, FIG = ROOT / "outputs/tables", ROOT / "data/processed/cambodia", ROOT / "outputs/figures/cambodia"
RNG = np.random.default_rng(20261103)
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

L = L.copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce"); L = L[L.unit.isin(DF.unit) & L.yr.between(1990, 2022)]
MJ = pd.read_csv(PROCD / "kh_major_roads_units.csv")
units, YEARS = DF.unit.values, range(2000, 2021)
grid = pd.MultiIndex.from_product([units, range(1990, 2024)], names=["unit", "yr"])
soc = L[L.family == "social"].groupby(["unit", "yr"]).w.sum().reindex(grid, fill_value=0)
ginf = L[L.family == "infra"].groupby(["unit", "yr"]).size().reindex(grid, fill_value=0).clip(upper=1)


def indicator(events):
    return events.groupby(["unit", "yr"]).size().reindex(grid, fill_value=0).clip(upper=1)


def panel(mj_events):
    I = indicator(mj_events)
    P = pd.DataFrame({"y": soc}).reset_index()
    for l in range(1, 6):
        P[f"m_l{l}"] = I.groupby(level=0).shift(l).values; P[f"g_l{l}"] = ginf.groupby(level=0).shift(l).values
    for l in (1, 2):
        P[f"m_f{l}"] = I.groupby(level=0).shift(-l).values
    P = P[P.yr.isin(YEARS)].fillna(0)
    return P[P.groupby("unit").y.transform("sum") > 0].reset_index(drop=True)


ML, GL, LEAD = [f"m_l{l}" for l in range(1, 6)], [f"g_l{l}" for l in range(1, 6)], ["m_f1", "m_f2"]


def fit(P, X, se=True):
    m = smf.glm("y ~ " + " + ".join(X) + " + C(unit) + C(yr)", data=P, family=sm.families.Poisson())
    return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.unit)[0]}, maxiter=500) if se else m.fit(maxiter=300)


def summed(m, vs):
    b = m.params[vs].sum(); s = float(np.sqrt(np.ones(len(vs)) @ m.cov_params().loc[vs, vs].values @ np.ones(len(vs))))
    return b, s


fmt = lambda b, s: f"{100 * (np.exp(b) - 1):+.0f}% (95% CI {100 * (np.exp(b - 1.96 * s) - 1):+.0f}% to {100 * (np.exp(b + 1.96 * s) - 1):+.0f}%, p {2 * norm.sf(abs(b / s)):.3f})"
ev_start = MJ[["unit", "start_year"]].rename(columns={"start_year": "yr"})
P0 = panel(ev_start)
say(f"major road projects: {MJ.project.nunique()} | treated districts: {MJ.unit.nunique()} | panel {P0.unit.nunique()} districts x {P0.yr.nunique()} years")

say("== A. five-year response of new social activities")
mA = fit(P0, ML + LEAD); bA, sA = summed(mA, ML)
say("  major roads, lags 1-5:            " + fmt(bA, sA)); say("  placebo, leads 1-2:               " + fmt(*summed(mA, LEAD)))
mA2 = fit(P0, ML + LEAD + GL)
say("  + GODAD transport/energy control: " + fmt(*summed(mA2, ML)) + " | GODAD infra " + fmt(*summed(mA2, GL)))

say("== C. randomisation inference: project start years shuffled among the projects (999 draws)")
proj_year = MJ.drop_duplicates("project").set_index("project").start_year
obs = bA; draws = []
for _ in range(999):
    shuffled = dict(zip(proj_year.index, RNG.permutation(proj_year.values)))
    ev = MJ.assign(yr=MJ.project.map(shuffled))[["unit", "yr"]]
    try:
        draws.append(fit(panel(ev), ML + LEAD, se=False).params[ML].sum())
    except Exception:
        pass
draws = np.array(draws)
say(f"  observed five-year response {100 * (np.exp(obs) - 1):+.0f}% | share of {len(draws)} shuffles at least as large: "
    f"p = {(1 + np.sum(draws >= obs)) / (1 + len(draws)):.3f} (two-sided |b|: {(1 + np.sum(np.abs(draws) >= abs(obs))) / (1 + len(draws)):.3f})")

say("== D. timing by completion or opening year")
ev_end = MJ.dropna(subset=["end_year"])[["unit", "end_year"]].rename(columns={"end_year": "yr"}).astype(int)
mD = fit(panel(ev_end), ML + LEAD)
say(f"  {MJ.dropna(subset=['end_year']).project.nunique()} projects with a documented end year: lags 1-5 " + fmt(*summed(mD, ML)) + " | leads " + fmt(*summed(mD, LEAD)))

say("== E. leave one project out (lags 1-5)")
for pj in proj_year.index:
    m = fit(panel(MJ[MJ.project != pj][["unit", "start_year"]].rename(columns={"start_year": "yr"})), ML + LEAD)
    say(f"  without {pj:<50s} " + fmt(*summed(m, ML)))

# ---------- B. event time ----------
first = ev_start.groupby("unit").yr.min()
Pe = pd.DataFrame({"y": soc}).reset_index(); Pe = Pe[Pe.yr.isin(YEARS)]
Pe["k"] = Pe.yr - Pe.unit.map(first)
KS = list(range(-5, 7))
for k in KS:
    if k == -1:
        continue
    cond = (Pe.k <= -5) if k == -5 else (Pe.k >= 6) if k == 6 else (Pe.k == k)
    Pe[f"k{k + 5}"] = (cond & Pe.k.notna()).astype(int)
Pe = Pe[Pe.groupby("unit").y.transform("sum") > 0].reset_index(drop=True)
KV = [f"k{k + 5}" for k in KS if k != -1]
mB = smf.glm("y ~ " + " + ".join(KV) + " + C(unit) + C(yr)", data=Pe, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(Pe.unit)[0]}, maxiter=500)
EV = pd.DataFrame([{"k": int(v[1:]) - 5, "b": mB.params[v], "se": mB.bse[v]} for v in KV] + [{"k": -1, "b": 0.0, "se": 0.0}]).sort_values("k")
EV["pct"], EV["lo"], EV["hi"] = [100 * (np.exp(EV.b + c * EV.se) - 1) for c in (0, -1.96, 1.96)]
EV.to_csv(T / "kh_major_roads_event.csv", index=False)
say("== B. event time (percent vs the year before the first major road; ends binned)")
say("  " + " | ".join(f"k={int(r.k)}: {r.pct:+.0f}%" for r in EV.itertuples()))

# ---------- figures ----------
INK, TEAL, GREY, RUST = "#24211b", "#0b6b6b", "#9a958a", "#a8461f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.axhline(0, color=GREY, lw=.8); ax.axvline(-.5, color=GREY, lw=.8, ls="--")
ax.fill_between(EV.k, EV.lo.clip(-100, 600), EV.hi.clip(-100, 600), color=TEAL, alpha=.15, lw=0)
ax.plot(EV.k, EV.pct.clip(-100, 600), "o-", color=TEAL, ms=4)
ax.set_xticks(KS); ax.set_xticklabels(["≤−5"] + [str(k) for k in KS[1:-1]] + ["≥6"])
ax.set_xlabel("Years since the first major road project in the district (approval year = 0)")
ax.set_ylabel("New social projects, % vs year −1")
ax.set_title("Social-sector aid around the arrival of a major road (district and year fixed effects, 95% CI)", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "S6_major_roads_event.png"); plt.close(fig)

# raw averages: treated districts by event time; other districts in the same calendar years
tr = Pe[Pe.unit.isin(first.index)].copy(); tr["kk"] = tr.k.clip(-8, 10)
other = Pe[~Pe.unit.isin(first.index)].groupby("yr").y.mean()
tr["ctrl"] = tr.yr.map(other)
A = tr.groupby("kk").agg(treated=("y", "mean"), others=("ctrl", "mean"), n=("unit", "nunique"))
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
ax.plot(A.index, A.treated, "o-", color=TEAL, ms=4, label=f"districts on a major road ({first.size})")
ax.plot(A.index, A.others, "s--", color=GREY, ms=3, label="other districts, same calendar years")
ax.axvline(-.5, color=GREY, lw=.8, ls="--"); ax.set_xlabel("Years since the first major road project (approval year = 0)")
ax.set_ylabel("New social projects per district per year"); ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "S6_major_roads_raw.png"); plt.close(fig)

# case: NR56 districts
case = MJ[MJ.project == "ADB 2539 NR56"].unit.unique()
cs = soc.loc[(case, list(YEARS))].groupby(level=1).sum()
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
ax.bar(cs.index, cs.values, color=TEAL, width=.75)
for x, lab in [(2009, "ADB loan approved"), (2015, "road completed")]:
    ax.axvline(x - .5, color=RUST, lw=1, ls="--"); ax.text(x - .4, cs.max() * .95, lab, color=RUST, fontsize=8)
ax.set_title(f"Case: NR56 Sisophon-Samraong ({len(case)} districts) - new social projects per year", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "S6_case_nr56.png"); plt.close(fig)
say("  NR56 districts, new social projects per year: " + ", ".join(f"{y}:{v:.1f}" for y, v in cs.items()))
(T / "kh_major_roads_rigor.txt").write_text("\n".join(log), encoding="utf-8")
