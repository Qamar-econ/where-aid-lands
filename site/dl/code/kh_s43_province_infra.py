"""Province-level infrastructure case study (spec 9-34, exploratory).

Part A  counterfactual travel time to Phnom Penh on the single 2017 OSM network (kh_s03 engine) for each access work:
        bridges -> ferry crossing of F minutes (F = 60; 30, 120 sensitivity); NR78 O Pong Moan-Ban Lung -> unpaved speeds.
        Province (IPUMS GEO1 group) population-weighted means; cell-level minutes saved (F = 60) for the map.
Part B  stacked event PPML on province-located activities (adm2 or adm1 precision), non-ICT and non-infrastructure,
        years -5..-1 vs +1..+5 around completion; treated = province time to Phnom Penh >= 10% longer without the work,
        controls < 2% for every work. Randomisation inference re-draws the treated province(s) within each stack.
Outputs: outputs/tables/kh_pinfra_dt.csv, kh_pinfra_cells.csv, kh_pinfra.csv, kh_pinfra_events.csv, kh_pinfra_panel.csv, kh_pinfra.txt
"""
import os, sys, re, numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; T_OUT = ROOT / "outputs/tables"; PROC = ROOT / "data/processed/cambodia"
sys.path.insert(0, str(HERE))
DRAWS = int(os.environ.get("PINFRA_DRAWS", 2000))
FERRY = (60, 30, 120)
# name, kind, selector, completion year (sources in spec 9-34)
WORKS = [("NR48 Koh Kong bridges", "ferry", "nr48", 2008), ("Sekong bridge", "ferry", [51433825], 2008),
         ("Prek Tamak bridge", "ferry", [91083631], 2010), ("NR78 Ratanakiri paving", "unpave", "nr78", 2012),
         ("Stung Treng bridge", "ferry", [320988098], 2015), ("Neak Loeung bridge", "ferry", [339921337], 2015)]
DT_FILE, CELL_FILE = T_OUT / "kh_pinfra_dt.csv", T_OUT / "kh_pinfra_cells.csv"
log = []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))

U = gpd.read_file(PROC / "kh_units.gpkg")[["GEOLEVEL2", "ADMIN_NAME", "PARENT", "geometry"]].rename(columns={"GEOLEVEL2": "unit"})
U["unit"] = U.unit.astype("int64"); U["PARENT"] = U.PARENT.astype(int)
COV = pd.read_csv(PROC / "kh_covariates.csv"); COV = COV[~COV.exclude].set_index("unit")
PROV_OF = COV.PARENT.astype(int); POP = COV.pop_2019

# ---------------- Part A (cached; PINFRA_REBUILD=1 to rebuild)
if not DT_FILE.exists() or os.environ.get("PINFRA_REBUILD") == "1":
    src = (HERE / "kh_s03_access.py").read_text(encoding="utf-8")
    a = "self.e.append((a, b, x1, y1, x2, y2, hw, paved))"
    b = "self.e.append((a, b, x1, y1, x2, y2, hw, paved, w.id, w.tags.get('ref', ''), w.tags.get('bridge', 'no')))"
    c0 = 'columns=["a", "b", "x1", "y1", "x2", "y2", "hw", "paved"]'
    c1 = 'columns=["a", "b", "x1", "y1", "x2", "y2", "hw", "paved", "wid", "ref", "bridge"]'
    stop = "cells = cells[np.isfinite(cells.ktm_wet)]"
    assert src.count(a) == 1 and src.count(c0) == 1 and src.count(stop) == 1
    ns = {"__file__": str(HERE / "kh_s03_access.py"), "__name__": "kh_s03"}
    exec(src.replace(a, b).replace(c0, c1)[:src.replace(a, b).replace(c0, c1).index(stop) + len(stop)], ns)
    E, cells, graph, dijkstra, BASE = (ns[k] for k in ("E", "cells", "graph", "dijkstra", "BASE"))
    node = ns["main_idx"][ns["tree"].query(np.c_[cells.geometry.x * ns["kx"], cells.geometry.y])[1]]
    base_cost = (E.km / E.v_dry * 60).values
    E["mx"] = (E.x1 + E.x2) / 2
    wkm = E.groupby("wid").km.transform("sum")

    def select(sel):
        if sel == "nr48":
            return (E.bridge.ne("no") & E.ref.str.contains(r"(?<!\d)48(?!\d)", regex=True) & E.mx.between(103.0, 103.8) & (wkm >= 0.15)).values
        if sel == "nr78":
            return (E.ref.str.contains(r"(?<!\d)78(?!\d)", regex=True) & (E.mx < 106.99) & E.hw.map(BASE).ge(10)).values
        return E.wid.isin(sel).values

    def cost_without(kind, m, F):
        c = base_cost.copy()
        if kind == "ferry":
            for w, ix in pd.Series(np.where(m)[0]).groupby(E.wid.values[m]):
                ix = ix.values; c[ix] = F * E.km.values[ix] / E.km.values[ix].sum()   # each crossing costs F minutes
        else:   # unpaved: kh_s03 rule min(v, 25) x 0.7, same gradient factor
            v0 = E.hw.map(BASE).astype(float).values[m]
            v = np.minimum(v0, 25) * float(os.environ.get("KH_UNPAVED", "0.7")) * np.clip(1 - 2.0 * E.grade.values[m], 0.4, 1)
            c[m] = E.km.values[m] / np.minimum(v, E.v_dry.values[m]) * 60
        return c

    def to_pp(cost):
        return dijkstra(graph(cost), indices=ns["ktm"], directed=False)[node] + cells.walk.values

    t0 = to_pp(base_cost)
    out = {"with": t0}; cellcols = {"lon": cells.geometry.x.values, "lat": cells.geometry.y.values, "pop": cells["pop"].values, "k": cells.k.values}
    for name, kind, sel, yr in WORKS:
        m = select(sel)
        say(f"{name}: {m.sum()} edges in {len(set(E.wid.values[m]))} OSM ways, {E.km.values[m].sum():.1f} km, driving {base_cost[m].sum():.1f} min")
        for F in (FERRY if kind == "ferry" else (60,)):
            tw = to_pp(cost_without(kind, m, F))
            out[f"{name}|{F}"] = tw
            if F == 60:
                cellcols[name] = np.round(tw - t0, 1)
    C = pd.DataFrame(out); C["k"] = cells.k.values; C["pop"] = cells["pop"].values
    dist = C.groupby("k").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["pop"]) for c in out}), include_groups=False)
    dist.rename_axis("unit").to_csv(DT_FILE)
    CC = pd.DataFrame(cellcols); CC = CC[(CC[[w[0] for w in WORKS]] > 0.5).any(axis=1)]
    CC.round({"lon": 4, "lat": 4, "pop": 0}).to_csv(CELL_FILE, index=False)

TT = pd.read_csv(DT_FILE, index_col="unit").reindex(COV.index)


def prov_delta(F=60):
    out = {}
    for p, g in TT.groupby(PROV_OF):
        w = POP[g.index]; base = np.average(g["with"], weights=w)
        out[p] = {n: np.log(np.average(g[f"{n}|{F if k == 'ferry' else 60}"], weights=w) / base) for n, k, _, _ in WORKS}
    return pd.DataFrame(out).T


# ---------------- Part B: province-located activities, all years
from kh_common import activityize, assign_units, donor_group
K = activityize(pd.read_csv(ROOT / "data/raw/cambodia/godad_khm.csv", low_memory=False))
if "dgroup" not in K:
    K["dgroup"] = donor_group(K)
K["prec"] = np.where(K.dgroup == "EU/US", np.select([K.precision_crs == 3, K.precision_crs == 4], ["adm2", "adm1"], "coarse"),
                     np.select([K.precision_code <= 3, K.precision_code == 4], ["adm2", "adm1"], "coarse"))
K["sec3"] = pd.to_numeric(K.sector_codes.astype(str).str.extract(r"(\d{3})")[0], errors="coerce")
K["family"] = np.select([K.sec3.isin([210, 230]), K.sec3.eq(220), K.sec3.between(110, 169), K.sec3.between(310, 339)],
                        ["infra", "ict", "social", "production"], "other")
g = assign_units(K[K.prec.isin(["adm2", "adm1"])], U[["unit", "PARENT", "geometry"]]).dropna(subset=["PARENT"])
g["prov"] = g.PARENT.astype(int)
P = g.groupby(["project_id", "prov"]).agg(yr=("startyear", "first"), family=("family", "first"), dgroup=("dgroup", "first"),
                                          adm2=("prec", lambda s: (s == "adm2").any())).reset_index()
P["w"] = 1 / P.groupby("project_id").prov.transform("nunique")
P = P[P.yr.notna() & P.prov.isin(PROV_OF.unique())]
provs = sorted(PROV_OF.unique())
say(f"province-located activities: {P.project_id.nunique()} ({(P.groupby('project_id').adm2.any()).sum()} with district precision) | "
    f"{len(provs)} provinces | per year 2003-2020: {P[P.yr.between(2003, 2020)].groupby('yr').w.sum().mean():.1f}")

ns = {"np": np, "pd": pd}
src = (HERE / "kh_s35_infra_stacked.py").read_text(encoding="utf-8")
exec(src[src.index("def _demean"):src.index("# ---------------- stacked panel")], ns)
ppml = ns["ppml"]
YR = {n: y for n, _, _, y in WORKS}


def assign(F=60, thr=np.log(1.10)):
    dl = prov_delta(F); tr = {}
    for p in provs:
        hits = [n for n, _, _, _ in sorted(WORKS, key=lambda w: w[3]) if dl.loc[p, n] >= thr]
        if hits:
            tr[p] = hits[0]
    ctrl = [p for p in provs if (dl.loc[p] < np.log(1.02)).all()]
    return tr, ctrl, dl


def stack(frame, tr, ctrl, shift=0):
    rows = []
    for ev in dict.fromkeys(tr.values()):
        t0 = YR[ev] - shift; tp = [p for p, e in tr.items() if e == ev]
        y = frame[frame.yr.between(t0 - 5, t0 + 5) & (frame.yr != t0)].groupby(["prov", "yr"]).w.sum()
        for p in tp + ctrl:
            for t in [t for t in range(t0 - 5, t0 + 6) if t != t0]:
                rows.append((ev, p, t, t - t0, int(p in tp), y.get((p, t), 0.0)))
    S = pd.DataFrame(rows, columns=["event", "prov", "yr", "ev", "treat", "y"])
    S["post"] = (S.ev > 0).astype(int); S["pe"] = S.event + "|" + S.prov.astype(str); S["te"] = S.event + "|" + S.yr.astype(str)
    return S


def beta(S, X):
    return ppml(S.y.values.astype(float), X, S.pe.values, S.te.values, se=False)[0][0]


def fit(S, draws=DRAWS, seed=20261106):
    tr_s = S[S.treat == 1]
    base = {"treated": tr_s.prov.nunique(), "events": tr_s.event.nunique(), "controls": S[S.treat == 0].prov.nunique(),
            "treated_pre": tr_s[tr_s.post == 0].y.sum(), "treated_post": tr_s[tr_s.post == 1].y.sum()}
    nan = {"pct": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "mde_pct": np.nan, "p_cluster": np.nan, "p_ri": np.nan, "ri_draws": 0}
    if min(base["treated_pre"], base["treated_post"]) == 0:
        return {**nan, **base}
    X = (S.treat * S.post).values[:, None].astype(float)
    b, V = ppml(S.y.values.astype(float), X, S.pe.values, S.te.values, S.prov.values)
    b, s = b[0], np.sqrt(V[0, 0])
    rng = np.random.default_rng(seed); grp = {k: gg for k, gg in S.groupby("event")}; bs = []
    single = len(grp) == 1 and base["treated"] == 1
    if single:   # one treated province: enumerate every province of the stack as the treated one (exact)
        g0 = next(iter(grp.values()))
        draws_list = [[p] for p in g0.prov.unique()]
    else:
        draws_list = range(draws)
    for dr in draws_list:
        parts = []
        for k, gg in grp.items():
            ps = gg.prov.unique(); n_t = gg[gg.treat == 1].prov.nunique()
            pick = set(dr) if single else set(rng.choice(ps, n_t, replace=False))
            parts.append(gg.prov.isin(pick).astype(float) * gg.post)
        try:
            bs.append(beta(S, pd.concat(parts).reindex(S.index).values[:, None]))
        except Exception:
            pass
    bs = np.array(bs); bs = bs[np.isfinite(bs)]
    if single:   # exact: the observed assignment is one of the enumerated ones, so it is already in bs
        assert np.isclose(bs, b, rtol=1e-6, atol=1e-8).any(), "observed assignment missing from the enumeration"
        p_ri = np.sum(np.abs(bs) >= abs(b) - 1e-12) / len(bs)
    else:
        p_ri = (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs))
    return {"pct": 100 * (np.exp(b) - 1), "ci_lo": 100 * (np.exp(b - 1.96 * s) - 1), "ci_hi": 100 * (np.exp(b + 1.96 * s) - 1),
            "mde_pct": 100 * (np.exp(2.8 * s) - 1), "p_cluster": 2 * norm.sf(abs(b / s)), "p_ri": p_ri, "ri_draws": len(bs), **base}


rows = []


def run(label, frame, draws=DRAWS, F=60, thr=np.log(1.10), shift=0, only=None):
    tr, ctrl, _ = assign(F, thr)
    if only:
        tr = {p: e for p, e in tr.items() if e == only}
    try:
        r = {"model": label, **fit(stack(frame, tr, ctrl, shift), draws)}
    except np.linalg.LinAlgError:
        r = {"model": label, "pct": np.nan, "treated": len(tr)}
    rows.append(r)
    say(f"  {label:<44s} {r['pct']:+6.1f}% (95% CI {r.get('ci_lo', np.nan):+.0f}% to {r.get('ci_hi', np.nan):+.0f}%) MDE {r.get('mde_pct', np.nan):.0f}% | "
        f"p RI {r.get('p_ri', np.nan):.3f} ({r.get('ri_draws', 0)} draws) | cluster {r.get('p_cluster', np.nan):.3f} | treated {r['treated']} "
        f"| controls {r.get('controls', np.nan)} | treated activities pre {r.get('treated_pre', np.nan):.1f} post {r.get('treated_post', np.nan):.1f}")
    return r


main = P[~P.family.isin(["ict", "infra"])]
tr, ctrl, dl = assign()
PN = {1: "Banteay Meanchey", 2: "Battambang", 3: "Kampong Cham", 4: "Kampong Chhnang", 5: "Kampong Speu", 6: "Kampong Thom", 7: "Kampot",
      8: "Kandal + Phnom Penh", 9: "Koh Kong + Sihanoukville", 10: "Kratie", 11: "Mondulkiri", 13: "Preah Vihear", 14: "Prey Veng", 15: "Pursat",
      16: "Ratanakiri", 17: "Siem Reap + Oddar Meanchey", 19: "Stung Treng", 20: "Svay Rieng", 21: "Takeo", 23: "Kep", 24: "Pailin"}
say("== province travel time to Phnom Penh without each work (F = 60 min), % longer")
say((100 * (np.exp(dl) - 1)).round(1).rename(index=PN).loc[lambda d: (d > 0.5).any(axis=1)].to_string())
say("== treated provinces (>= 10%): " + "; ".join(f"{PN[p]} <- {e} ({YR[e]})" for p, e in tr.items()) + f" | controls {len(ctrl)}")
say("== MAIN (spec 9-34)")
run("main (all treated, F 60, >= 10%)", main)
say("== BY EVENT (one treated province each: exact RI over the stack's provinces)")
EV = []
for e in dict.fromkeys(tr.values()):
    EV.append({"event": e, **run(f"event: {e}", main, only=e)})
say("== ROBUSTNESS")
run("district-precise (adm2) activities only", main[main.adm2], 999)
run("ferry 30 min", main, 999, F=30)
run("ferry 120 min", main, 999, F=120)
run("threshold >= 5%", main, 999, thr=np.log(1.05))
run("EU/US donors only", main[main.dgroup == "EU/US"], 999)
run("all activities incl. infrastructure", P[P.family != "ict"], 999)
say("== PLACEBO: completion dated 3 years early")
run("placebo -3 years", main, 999, shift=3)
pd.DataFrame(rows).to_csv(T_OUT / "kh_pinfra.csv", index=False)
pd.DataFrame(EV).to_csv(T_OUT / "kh_pinfra_events.csv", index=False)
# panel for the figure: every province x year, 2000-2020, with treated flags
pan = main[main.yr.between(2000, 2020)].groupby(["prov", "yr"]).w.sum().reindex(pd.MultiIndex.from_product([provs, range(2000, 2021)], names=["prov", "yr"]), fill_value=0).rename("activities").reset_index()
pan["province"] = pan.prov.map(PN); pan["treated_by"] = pan.prov.map(tr); pan["control"] = pan.prov.isin(ctrl)
pan["completed"] = pan.treated_by.map(YR); pan.to_csv(T_OUT / "kh_pinfra_panel.csv", index=False)
(T_OUT / "kh_pinfra.txt").write_text("\n".join(log), encoding="utf-8")
