"""Bridge-opening case study (spec 9-32, exploratory): did other aid follow when a new bridge cut travel time?

Part A  with/without-bridge travel time on the single 2017 OSM network (kh_s03 engine): the bridge way's edges cost a
        ferry crossing of F minutes instead of driving time (F = 60 main; 30, 120 sensitivity). Population-weighted
        district mean time to the nearest town >= 20,000; delta = ln(T_without) - ln(T_with).
Part B  stacked event PPML, treated (delta >= ln 1.25 in travel time to Phnom Penh, spec 9-32a; nearest town as first specified) vs clean controls (delta < ln 1.05 for every bridge),
        years -5..-1 vs +1..+5 around opening (year 0 dropped), outcome = non-ICT, non-infrastructure district-located
        activities per year. District-clustered SE; randomisation inference re-drawing the treated set within each stack.
Outputs: outputs/tables/kh_bridges_dt.csv, kh_bridges.csv, kh_bridges_cases.csv, kh_bridges.txt
"""
import os, numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; T_OUT = ROOT / "outputs/tables"
DRAWS = int(os.environ.get("BRIDGE_DRAWS", 2000))
# name, OSM way ids in the 2017-01-01 extract (found by name; Koh Puos is tagged "Techo Morakot Bridge"), opening year
BRIDGES = [("Sekong", [51433825], 2008), ("Prek Kdam", [91369579], 2010), ("Prek Tamak", [91083631], 2010),
           ("Prek Pnov", [98572857], 2010), ("Koh Puos", [378249444], 2011), ("Ta Khmau", [255755623], 2015),
           ("Stung Treng", [320988098], 2015), ("Neak Loeung", [339921337], 2015)]
FERRY = (60, 30, 120)
DT_FILE = T_OUT / "kh_bridges_dt.csv"
log = []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))

# ---------------- Part A: counterfactual travel times (cached; delete kh_bridges_dt.csv to rebuild)
if not DT_FILE.exists() or os.environ.get("BRIDGE_REBUILD") == "1":
    src = (HERE / "kh_s03_access.py").read_text(encoding="utf-8")
    src = src.replace("self.e.append((a, b, x1, y1, x2, y2, hw, paved))", "self.e.append((a, b, x1, y1, x2, y2, hw, paved, w.id))")
    src = src.replace('columns=["a", "b", "x1", "y1", "x2", "y2", "hw", "paved"]', 'columns=["a", "b", "x1", "y1", "x2", "y2", "hw", "paved", "wid"]')
    stop = "cells = cells[np.isfinite(cells.ktm_wet)]"
    assert src.count("w.id))") == 1 and src.count('"paved", "wid"') == 1 and src.count(stop) == 1
    ns = {"__file__": str(HERE / "kh_s03_access.py"), "__name__": "kh_s03"}
    exec(src[:src.index(stop) + len(stop)], ns)
    E, cells, graph, dijkstra, snap, towns = (ns[k] for k in ("E", "cells", "graph", "dijkstra", "snap", "towns"))
    node = ns["main_idx"][ns["tree"].query(np.c_[cells.geometry.x * ns["kx"], cells.geometry.y])[1]]
    src_t = snap(towns); base_cost = (E.km / E.v_dry * 60).values

    def district_tt(cost, tag):
        G = graph(cost)
        res = {}
        for dest, t in [("town", dijkstra(G, indices=src_t, directed=False, min_only=True)), ("ktm", dijkstra(G, indices=ns["ktm"], directed=False))]:
            d = pd.DataFrame({"k": cells.k.values, "t": t[node] + cells.walk.values, "p": cells["pop"].values})
            res[f"{dest}|{tag}"] = d.groupby("k").apply(lambda g: np.average(g.t, weights=g.p), include_groups=False)
        return res

    out = district_tt(base_cost, "with")
    for name, wids, yr in BRIDGES:
        m = E.wid.isin(wids).values
        assert m.sum() > 0, name
        span = E.km.values[m].sum() / len(wids)
        say(f"{name}: {m.sum()} edges, {span:.2f} km, driving {base_cost[m].sum() / len(wids):.1f} min")
        for F in FERRY:
            c = base_cost.copy(); c[m] = F * E.km.values[m] / span   # whole crossing costs F minutes
            out |= district_tt(c, f"{name}|{F}")
    pd.DataFrame(out).rename_axis("unit").to_csv(DT_FILE)

TT = pd.read_csv(DT_FILE, index_col="unit")

# ---------------- Part B: stacked event study
ns = {"__file__": str(HERE / "kh_s06_ladder.py")}
exec((HERE / "kh_s06_ladder.py").read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L = ns["DF"], ns["L"]
src = (HERE / "kh_s35_infra_stacked.py").read_text(encoding="utf-8")
exec(src[src.index("def _demean"):src.index("# ---------------- stacked panel")], ns)
ppml = ns["ppml"]
L = L.copy(); L["yr"] = pd.to_numeric(L.start, errors="coerce")
L = L[L.unit.isin(DF.unit) & L.yr.notna()]
units = DF.unit.values; name_of = DF.set_index("unit").ADMIN_NAME; prov_of = DF.set_index("unit").prov.astype(str)
chk = pd.read_csv(ROOT / "data/processed/cambodia/kh_access.csv").set_index("unit" if "unit" in pd.read_csv(ROOT / "data/processed/cambodia/kh_access.csv", nrows=0).columns else "k").tt_town_dry
say(f"check: with-bridge times reproduce kh_access tt_town_dry, max abs diff {np.nanmax(np.abs(TT['town|with'] - chk.reindex(TT.index))):.4f} min")


def delta(F, dest="ktm"):
    return pd.DataFrame({b: np.log(TT[f"{dest}|{b}|{F}"] / TT[f"{dest}|with"]) for b, _, _ in BRIDGES}).reindex(units)


def assign(F=60, thr=np.log(1.25), year=None, dest="ktm"):
    dl = delta(F, dest); yr = dict((b, y) for b, _, y in BRIDGES) | (year or {})
    tr = {}
    for u in units:
        hits = [b for b in sorted(dl.columns, key=lambda b: yr[b]) if dl.loc[u, b] >= thr]
        if hits:
            tr[u] = hits[0]
    ctrl = [u for u in units if (dl.loc[u] < np.log(1.05)).all()]
    return tr, ctrl, yr, dl


def stack(frame, tr, ctrl, yr, shift=0):
    rows = []
    for b in dict.fromkeys(tr.values()):
        t0 = yr[b] - shift; tu = [u for u, bb in tr.items() if bb == b]
        y = frame[frame.yr.between(t0 - 5, t0 + 5) & (frame.yr != t0)].groupby(["unit", "yr"]).w.sum()
        for u in tu + ctrl:
            for t in [t for t in range(t0 - 5, t0 + 6) if t != t0]:
                rows.append((b, u, t, t - t0, int(u in tu), y.get((u, t), 0.0)))
    S = pd.DataFrame(rows, columns=["bridge", "unit", "yr", "ev", "treat", "y"])
    S["post"] = (S.ev > 0).astype(int); S["db"] = S.bridge + "|" + S.unit.astype(str); S["tb"] = S.bridge + "|" + S.yr.astype(str)
    return S


def fit(S, draws=DRAWS, rng=None):
    X = (S.treat * S.post).values[:, None].astype(float)
    tr_s = S[S.treat == 1]
    base = {"treated": tr_s.unit.nunique(), "bridges": tr_s.bridge.nunique(), "controls": S[S.treat == 0].unit.nunique(),
            "treated_pre": tr_s[tr_s.post == 0].y.sum(), "treated_post": tr_s[tr_s.post == 1].y.sum()}
    if min(base["treated_pre"], base["treated_post"]) == 0:   # all-zero pre or post: beta is not identified
        return {"pct": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "mde_pct": np.nan, "p_cluster": np.nan, "p_ri": np.nan, **base}
    b, V = ppml(S.y.values.astype(float), X, S.db.values, S.tb.values, S.unit.values)
    b, s = b[0], np.sqrt(V[0, 0]); p_ri = np.nan
    if draws:
        rng = rng or np.random.default_rng(20261104); bs = []
        grp = {k: g for k, g in S.groupby("bridge")}
        for _ in range(draws):
            parts = []
            for k, g in grp.items():
                us = g.unit.unique(); n_t = g[g.treat == 1].unit.nunique()
                pick = set(rng.choice(us, n_t, replace=False)); parts.append(g.unit.isin(pick).astype(float) * g.post)
            Xp = pd.concat(parts).reindex(S.index).values[:, None]
            try:
                bs.append(ppml(S.y.values.astype(float), Xp, S.db.values, S.tb.values, se=False)[0][0])
            except Exception:
                pass
        bs = np.array(bs); bs = bs[np.isfinite(bs)]
        p_ri = (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs))
    return {"pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * s) - 1), "ci_hi": 100 * (np.exp(b + 1.96 * s) - 1),
            "mde_pct": 100 * (np.exp(2.8 * s) - 1), "p_cluster": 2 * norm.sf(abs(b / s)), "p_ri": p_ri, **base}


rows = []


def run(label, frame, draws=DRAWS, F=60, thr=np.log(1.25), year=None, shift=0, dest="ktm"):
    tr, ctrl, yr, _ = assign(F, thr, year, dest)
    try:
        r = {"model": label, **fit(stack(frame, tr, ctrl, yr, shift), draws)}
    except np.linalg.LinAlgError:   # treated set too thin for the fixed effects
        r = {"model": label, "pct": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "mde_pct": np.nan, "p_cluster": np.nan, "p_ri": np.nan,
             "treated": len(tr), "bridges": len(set(tr.values())), "controls": len(ctrl), "treated_pre": np.nan, "treated_post": np.nan}
    rows.append(r)
    say(f"  {label:<40s} {r['pct']:+6.1f}% (95% CI {r['ci_lo']:+.0f}% to {r['ci_hi']:+.0f}%) MDE {r['mde_pct']:.0f}% | p RI {r['p_ri']:.3f} | "
        f"cluster {r['p_cluster']:.3f} | treated {r['treated']} in {r['bridges']} bridges, controls {r['controls']} | treated projects pre {r['treated_pre']:.0f} post {r['treated_post']:.0f}")
    return tr, ctrl, yr


main = L[~L.family.isin(["ict", "infra"])]
tr, ctrl, yr, dl = assign()
say("== treated districts (travel time to Phnom Penh would be >= 25% longer by ferry, F = 60 min; spec 9-32a)")
C = []
for u, b in sorted(tr.items(), key=lambda kv: (yr[kv[1]], kv[1])):
    C.append({"unit": u, "district": name_of[u], "province": prov_of[u], "bridge": b, "open": yr[b], "tt_with": TT.loc[u, "ktm|with"],
              "tt_without": TT.loc[u, f"ktm|{b}|60"], "pct_longer": 100 * (np.exp(dl.loc[u, b]) - 1)})
    say(f"  {b:<12s} {yr[b]} {str(name_of[u]):<22s} ({prov_of[u]:<16s}) {TT.loc[u, 'ktm|with']:6.1f} -> {TT.loc[u, f'ktm|{b}|60']:6.1f} min (+{C[-1]['pct_longer']:.0f}%)")
for b, _, y in BRIDGES:
    if b not in tr.values():
        say(f"  {b:<12s} {y} no district crosses the threshold (max +{100 * (np.exp(dl[b].max()) - 1):.0f}%)")
say(f"controls: {len(ctrl)} districts")

say("== AS FIRST SPECIFIED (spec 9-32): nearest town >= 20,000")
run("9-32 nearest town (as specified)", main, dest="town")
say("== MAIN (spec 9-32a): Phnom Penh, non-ICT, non-infrastructure activities, -5..-1 vs +1..+5")
run("main (Phnom Penh, F 60 min, >= 25%)", main)
say("== ROBUSTNESS")
run("ferry 30 min", main, 999, F=30)
run("ferry 120 min", main, 999, F=120)
run("threshold >= 10%", main, 999, thr=np.log(1.10))
run("Prek Kdam opened 2009", main, 999, year={"Prek Kdam": 2009})
run("EU/US donors only", main[main.dgroup == "EU/US"], 999)
run("all activities incl. infrastructure", L[L.family != "ict"], 999)
say("== PLACEBO: opening dated 3 years early")
run("placebo -3 years", main, 999, shift=3)
R = pd.DataFrame(rows); R.to_csv(T_OUT / "kh_bridges.csv", index=False)

# case timelines (every treated district) and the dispersal share in the treated districts' provinces
cs = []
for c in C:
    yy = main[(main.unit == c["unit"])].groupby("yr").w.sum()
    prov_units = DF[DF.prov.astype(str) == c["province"]].unit
    pv = main[main.unit.isin(prov_units)].groupby("yr").w.sum()
    for t in range(c["open"] - 5, c["open"] + 6):
        if t == c["open"]:
            continue
        cs.append({**c, "yr": t, "ev": t - c["open"], "projects": yy.get(t, 0.0), "province_projects": pv.get(t, 0.0)})
CS = pd.DataFrame(cs); CS.to_csv(T_OUT / "kh_bridges_cases.csv", index=False)
say("== DISPERSAL (descriptive): share of the province's projects that land in the bridge's treated districts")
for b in dict.fromkeys(c["bridge"] for c in C):
    g = CS[CS.bridge == b]; tu = g.unit.unique(); provs = g.province.unique()
    pu = DF[DF.prov.astype(str).isin(provs)].unit
    for lab, sel in [("pre", lambda e: (e >= -5) & (e <= -1)), ("post", lambda e: (e >= 1) & (e <= 5))]:
        t0 = yr[b]; m = main[main.unit.isin(pu) & sel(main.yr - t0)]
        n_all, n_tr = m.w.sum(), m[m.unit.isin(tu)].w.sum()
        say(f"  {b:<12s} {lab:<4s}: {n_tr:4.0f} of {n_all:4.0f} projects in {', '.join(provs)} ({100 * n_tr / max(n_all, 1):.0f}%)")
(T_OUT / "kh_bridges.txt").write_text("\n".join(log), encoding="utf-8")
