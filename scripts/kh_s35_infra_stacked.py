"""Does new transport infrastructure pull in later aid from other donors? Re-test pre-specified in spec 9-23.

Stacked clean-event difference-in-differences (Cengiz et al. 2019). For each event year g (2003-2015):
  treated  districts where a GODAD transport activity (OECD 210) starts in g, with none in g-5..g-1
  controls districts with no transport activity in g-5..g+5
  window   k = -5..+5, k = 0 dropped (same-year bundling, commitment-date noise)
  outcome  new non-infrastructure activities (transport, energy, ICT excluded), activity weight w, excluding activities of the
           donors behind that stack's events, in both arms
PPML with stack x district and stack x year fixed effects (IRLS with the fixed effects swept out by weighted alternating
projections, as in ppmlhdfe), errors clustered by district. Primary inference: max of the cluster p and a randomisation
inference p (treatment reassigned among each stack's eligible districts, 999 draws).
Robustness R1-R3 and secondary analyses S1-S7 as listed in spec 9-23.
Output: outputs/tables/kh_infra_stacked.txt, kh_infra_stacked_event.csv; outputs/figures/cambodia/S7_infra_stacked_event.png
"""
import os, numpy as np, pandas as pd, geopandas as gpd, matplotlib
from scipy.stats import norm, chi2
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, K, U, ROOT = ns["DF"], ns["L"], ns["K"], ns["U"], ns["ROOT"]
T, PROCD, FIG = ROOT / "outputs/tables", ROOT / "data/processed/cambodia", ROOT / "outputs/figures/cambodia"
RNG = np.random.default_rng(20261103)
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

L = L.copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce")
L["sec3"] = L.project_id.map(K.groupby("project_id").sec3.first())
L["donor"] = L.project_id.map(K.groupby("project_id").donor.first())
L = L[L.unit.isin(DF.unit) & L.yr.between(1990, 2022)].reset_index(drop=True)
UNITS = DF.unit.values
OUT = L[~L.family.isin(["infra", "ict"])]
TRANSPORT = L[L.sec3 == 210][["unit", "yr", "donor"]]
INFRA = L[L.family == "infra"][["unit", "yr", "donor"]]
MJ = pd.read_csv(PROCD / "kh_major_roads_units.csv")
MAJOR = MJ[["unit", "start_year", "donor"]].rename(columns={"start_year": "yr"})
say("GODAD district-located activities by start year (all families):",
    ", ".join(f"{y}:{n}" for y, n in L.groupby("yr").size().loc[1995:2020].items()))


# ---------------- PPML with two high-dimensional fixed effects ----------------
def _demean(V, w, f1, f2, tol=1e-10, maxit=2000):
    V = V.copy(); n1, n2 = f1.max() + 1, f2.max() + 1
    s1, s2 = np.bincount(f1, w, n1), np.bincount(f2, w, n2)
    for _ in range(maxit):
        old = V.copy()
        for f, s, n in ((f1, s1, n1), (f2, s2, n2)):
            for j in range(V.shape[1]):
                V[:, j] -= (np.bincount(f, w * V[:, j], n) / s)[f]
        if np.max(np.abs(V - old)) < tol:
            break
    return V


def ppml(y, X, fe1, fe2, cl=None, se=True):
    keep = np.ones(len(y), bool)
    for _ in range(10):        # groups with no positive outcome carry no information in PPML
        k1 = pd.Series(y[keep]).groupby(fe1[keep]).transform("sum").values > 0
        k2 = pd.Series(y[keep]).groupby(fe2[keep]).transform("sum").values > 0
        ok = k1 & k2
        if ok.all():
            break
        idx = np.where(keep)[0]; keep[idx[~ok]] = False
    y, X = y[keep], X[keep]; f1, f2 = pd.factorize(fe1[keep])[0], pd.factorize(fe2[keep])[0]
    mu = (y + y.mean()) / 2; eta = np.log(mu); dev0 = np.inf
    for _ in range(100):
        z = eta + (y - mu) / mu
        D = _demean(np.column_stack([z, X]), mu, f1, f2)
        zt, Xt = D[:, 0], D[:, 1:]
        b = np.linalg.solve(Xt.T @ (mu[:, None] * Xt), Xt.T @ (mu * zt))
        eta = z - (zt - Xt @ b); mu = np.exp(eta)
        dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
        if abs(dev - dev0) / max(dev, 1e-9) < 1e-10:
            break
        dev0 = dev
    if not se:
        return b, None
    Xt = _demean(X.astype(float), mu, f1, f2)
    bread = np.linalg.inv(Xt.T @ (mu[:, None] * Xt))
    g = pd.factorize(cl[keep])[0]; G = g.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, g, Xt * (y - mu)[:, None])
    n, k = len(y), X.shape[1] + f2.max() + 1   # fixest-style: district FE are nested in the clusters, year FE are not
    V = bread @ (S.T @ S) @ bread * G / (G - 1) * (n - 1) / (n - k)
    return b, V


# ---------------- stacked panel ----------------
def stack(events, outcome=OUT, lo=-5, hi=5, units=UNITS, years=range(2003, 2016), drop_donor=True, treat_rule="own"):
    ev_years = events.groupby("unit").yr.apply(set).to_dict()
    has = lambda u, a, b: any(a <= t <= b for t in ev_years.get(u, ()))
    rows = []
    for g in years:
        if treat_rule == "own":
            tr = [u for u in units if g in ev_years.get(u, ()) and not has(u, g - 5, g - 1)]
            co = [u for u in units if not has(u, g - 5, g + hi)]
        else:                      # neighbours of event districts (S7)
            ev_u = [u for u in units if g in ev_years.get(u, ()) and not has(u, g - 5, g - 1)]
            near = set().union(*[NB.get(u, set()) for u in ev_u]) if ev_u else set()
            exposed = set().union(*[NB.get(u, set()) | {u} for u in units if has(u, g - 5, g + hi)])
            tr = [u for u in units if u in near and not has(u, g - 5, g + hi)]
            co = [u for u in units if u not in exposed]
        if not tr or not co:
            continue
        donors = set(events[(events.yr == g) & events.unit.isin(tr if treat_rule == "own" else ev_u)].donor.dropna())
        o = outcome[~outcome.donor.isin(donors)] if drop_donor else outcome
        yy = o[o.yr.between(g + lo, g + hi)].groupby(["unit", "yr"]).w.sum()
        for u in tr + co:
            for k in range(lo, hi + 1):
                if k == 0:
                    continue
                rows.append((g, u, k, g + k, int(u in tr), yy.get((u, g + k), 0.0)))
    P = pd.DataFrame(rows, columns=["stk", "unit", "k", "yr", "treat", "y"])
    P["post"] = (P.k > 0).astype(int)
    return P


def fe_ids(P, terciles=False):
    f1 = (P.stk.astype(str) + "_" + P.unit.astype(str)).values
    f2 = (P.stk.astype(str) + "_" + P.yr.astype(str) + ("_" + P.unit.map(TERC).astype(str) if terciles else "")).values
    return f1, f2


def did(P, terciles=False, inter=None):
    f1, f2 = fe_ids(P, terciles)
    if inter is None:
        X = (P.treat * P.post).values[:, None].astype(float)
    else:
        X = np.column_stack([P.treat * P.post * inter, P.treat * P.post * (1 - inter)]).astype(float)
    return ppml(P.y.values.astype(float), X, f1, f2, P.unit.values)


def fmt(b, s, p=None):
    p = 2 * norm.sf(abs(b / s)) if p is None else p
    return (f"{100 * (np.exp(b) - 1):+.0f}% (95% CI {100 * (np.exp(b - 1.96 * s) - 1):+.0f}% to {100 * (np.exp(b + 1.96 * s) - 1):+.0f}%; "
            f"90% upper {100 * (np.exp(b + 1.645 * s) - 1):+.0f}%; p {p:.3f}; MDE80 {100 * (np.exp(2.8 * s) - 1):.0f}%)")


def describe(P, label):
    t = P[P.treat == 1]
    say(f"  [{label}] stacks {P.stk.nunique()} | treated district-stacks {t.groupby(['stk', 'unit']).ngroups} "
        f"({t.unit.nunique()} districts) | control district-stacks {P[P.treat == 0].groupby(['stk', 'unit']).ngroups} | obs {len(P)}")


acc = DF.set_index("unit").access
TERC = pd.qcut(acc, 3, labels=False)
REMOTE = (acc < acc.median()).astype(int)
Ug = U[U.unit.isin(UNITS)].to_crs(32648)
NB = {}
for u, geom in zip(Ug.unit, Ug.geometry):
    NB[u] = set(Ug[Ug.geometry.touches(geom) | Ug.geometry.intersects(geom.buffer(50))].unit) - {u}

# ================= PRIMARY =================
say("== PRIMARY: new transport activity -> other donors' non-infrastructure activities, k = 1..5 vs -5..-1")
P0 = stack(TRANSPORT); describe(P0, "primary")
b, V = did(P0); b0, s0 = b[0], np.sqrt(V[0, 0]); p_cl = 2 * norm.sf(abs(b0 / s0))
draws = []
grp = P0.groupby("stk")
base = P0[["stk", "unit", "treat"]].drop_duplicates()
for _ in range(int(os.environ.get("RI_DRAWS", 999))):
    perm = base.copy()
    perm["treat"] = perm.groupby("stk").treat.transform(lambda s: RNG.permutation(s.values))
    Q = P0.drop(columns="treat").merge(perm, on=["stk", "unit"])
    draws.append(ppml(Q.y.values.astype(float), (Q.treat * Q.post).values[:, None].astype(float), *fe_ids(Q), se=False)[0][0])
draws = np.array(draws)
p_ri = (1 + np.sum(np.abs(draws) >= abs(b0))) / (1 + len(draws))
say("  effect " + fmt(b0, s0) + f" | cluster p {p_cl:.3f} | RI p {p_ri:.3f} | primary p = {max(p_cl, p_ri):.3f}")

# event study and pre-trend test
KS = [k for k in range(-5, 6) if k not in (-1, 0)]
X = np.column_stack([(P0.treat * (P0.k == k)).values for k in KS]).astype(float)
be, Ve = ppml(P0.y.values.astype(float), X, *fe_ids(P0), P0.unit.values)
lead = [KS.index(k) for k in (-5, -4, -3, -2)]
W = float(be[lead] @ np.linalg.solve(Ve[np.ix_(lead, lead)], be[lead])); p_pre = chi2.sf(W, len(lead))
EV = pd.DataFrame({"k": KS, "b": be, "se": np.sqrt(np.diag(Ve))})
EV = pd.concat([EV, pd.DataFrame({"k": [-1], "b": [0.0], "se": [0.0]})]).sort_values("k")
EV["pct"], EV["lo"], EV["hi"] = [100 * (np.exp(EV.b + c * EV.se) - 1) for c in (0, -1.96, 1.96)]
EV.to_csv(T / "kh_infra_stacked_event.csv", index=False)
say("  event time (vs k=-1): " + " | ".join(f"k={int(r.k)} {r.pct:+.0f}%" for r in EV.itertuples()))
say(f"  pre-trend joint test (k=-5..-2): chi2 {W:.2f}, p {p_pre:.3f}")
up90 = 100 * (np.exp(b0 + 1.645 * s0) - 1)
verdict = ("H1 pull" if (b0 > 0 and max(p_cl, p_ri) < 0.05 and p_pre >= 0.10) else
           "informative null: effects above +50% ruled out" if up90 < 50 else "inconclusive")
say(f"  VERDICT (rule fixed in spec 9-23): {verdict}")

# ================= ROBUSTNESS =================
say("== ROBUSTNESS")
b, V = did(P0, terciles=True); say("  R1 stack x year FE by access tercile: " + fmt(b[0], np.sqrt(V[0, 0])))
P2 = stack(TRANSPORT, units=DF.loc[~DF.capital, "unit"].values); b, V = did(P2); say("  R2 Phnom Penh excluded:              " + fmt(b[0], np.sqrt(V[0, 0])))
P3 = stack(TRANSPORT, drop_donor=False); b, V = did(P3); say("  R3 event donors kept in outcome:     " + fmt(b[0], np.sqrt(V[0, 0])))

# ================= SECONDARY =================
say("== SECONDARY (exploratory; Holm across S1-S7, cluster p)")
first = L.sort_values("yr").drop_duplicates(["unit", "donor"]); first = first[~first.family.isin(["infra", "ict"])]
sec = []
P = stack(INFRA); describe(P, "S1"); b, V = did(P); sec.append(("S1 transport + energy events", b[0], np.sqrt(V[0, 0])))
P = stack(pd.concat([TRANSPORT, MAJOR])); describe(P, "S2"); b, V = did(P); sec.append(("S2 + major ADB/Japan roads", b[0], np.sqrt(V[0, 0])))
P = stack(TRANSPORT, outcome=OUT[OUT.family == "social"]); b, V = did(P); sec.append(("S3 social-sector outcome", b[0], np.sqrt(V[0, 0])))
P = stack(TRANSPORT, outcome=first); b, V = did(P); sec.append(("S4 donors new to the district", b[0], np.sqrt(V[0, 0])))
b, V = did(P0, inter=P0.unit.map(REMOTE).values)
sec.append(("S5a remote half", b[0], np.sqrt(V[0, 0]))); sec.append(("S5b connected half", b[1], np.sqrt(V[1, 1])))
dd = b[1] - b[0]; sdd = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
P = stack(TRANSPORT, lo=-3, hi=8, years=range(2003, 2013)); P = P[(P.k < 0) | (P.k >= 6)]; describe(P, "S6")
b, V = did(P); sec.append(("S6 late window k=6..8 vs -3..-1", b[0], np.sqrt(V[0, 0])))
P = stack(TRANSPORT, treat_rule="neighbour"); describe(P, "S7"); b, V = did(P); sec.append(("S7 neighbouring districts", b[0], np.sqrt(V[0, 0])))
ps = np.array([2 * norm.sf(abs(b / s)) for _, b, s in sec]); order = np.argsort(ps); m = len(ps)
holm = np.empty(m); run = 0
for r, i in enumerate(order):
    run = max(run, min(1, (m - r) * ps[i])); holm[i] = run
for (lab, b, s), ph in zip(sec, holm):
    say(f"  {lab:<34s} " + fmt(b, s) + f" | Holm p {ph:.3f}")
say(f"  S5 connected minus remote: {100 * (np.exp(dd) - 1):+.0f}% (p {2 * norm.sf(abs(dd / sdd)):.3f})")

# ================= figure =================
INK, TEAL, GREY = "#24211b", "#0b6b6b", "#9a958a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.axhline(0, color=GREY, lw=.8); ax.axvline(0, color=GREY, lw=.8, ls="--")
E = EV.copy()
ax.fill_between(E.k, E.lo.clip(-100, 400), E.hi.clip(-100, 400), color=TEAL, alpha=.15, lw=0)
ax.plot(E.k, E.pct.clip(-100, 400), "o-", color=TEAL, ms=4)
ax.set_xticks(range(-5, 6)); ax.set_xlabel("Years since a new transport activity in the district (year 0 dropped)")
ax.set_ylabel("Other donors' new non-infra\nactivities, % vs year -1")
ax.set_title("Stacked clean-event design, 95% CI (spec 9-23)", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "S7_infra_stacked_event.png"); plt.close(fig)
(T / "kh_infra_stacked.txt").write_text("\n".join(log), encoding="utf-8")
