"""Spec 9-36 (exploratory): four designs closer to cause than the main within-province comparison.

A  neighbour pairs: adjacent districts stacked by pair, pair fixed effects (local unobservables)
B  straight-line distance to the nearest town vs the road detour (travel time / distance), in one model
C  field vs institutional projects (DAC 3-digit, fixed in the spec): slopes by type, and the slope gap with district fixed effects
D  EU/US vs China within the same district: slope gap with district fixed effects
Outcome, sample and controls as kh_main.py. Randomisation inference: A, B Freedman-Lane within province; C, D permutation
of the type/donor label across projects. Output: outputs/tables/kh_designs.csv, kh_designs.txt
"""
import sys, numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import norm
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
src = (HERE / "kh_main.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_main.py"), "__name__": "kh_main_ns"}
sys.argv = sys.argv[:1]
exec(src[:src.index('say("== MAIN: one construct')], ns)
DF, MAIN, A0, U, counts, T, PROC = (ns[k] for k in ("DF", "MAIN", "A0", "U", "counts", "T", "PROC"))
ns["RNG"] = np.random.default_rng(20261109)
s35 = (HERE / "kh_s35_infra_stacked.py").read_text(encoding="utf-8"); pp = {"np": np, "pd": pd}
exec(s35[s35.index("def _demean"):s35.index("# ---------------- stacked panel")], pp)
ppml = pp["ppml"]
RNG = np.random.default_rng(20261110)
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
pc = lambda b: 100 * (np.exp(b) - 1)


def add(design, label, b, s, p_ri, p_other=np.nan, n=np.nan, extra=""):
    r = {"design": design, "model": label, "pct": pc(b), "ci_lo": pc(b - 1.96 * s), "ci_hi": pc(b + 1.96 * s), "p_ri": p_ri, "p_cluster": p_other, "n": n}
    rows.append(r)
    say(f"  {design} {label:<52s} {r['pct']:+6.0f}% (95% CI {r['ci_lo']:+.0f}% to {r['ci_hi']:+.0f}%) | p RI {p_ri:.3f} | p cluster {p_other:.3f} | n {n} {extra}")


d0 = counts(MAIN).reset_index(drop=True)
units = d0.unit.values

# ---------------- A neighbour pairs ----------------
say("== A neighbour pairs (pair fixed effects)")
G = U[U.unit.isin(units)].set_index("unit").loc[units].reset_index()
sidx = G.sindex; pairs = []
for i, geom in enumerate(G.geometry):
    for j in sidx.query(geom.buffer(100), predicate="intersects"):
        if j > i and G.geometry.iloc[j].distance(geom) < 100:
            pairs.append((G.unit.iloc[i], G.unit.iloc[j]))
P = pd.DataFrame(pairs, columns=["a", "b"]); P["pair"] = np.arange(len(P))
same = (d0.set_index("unit").prov.reindex(P.a).values == d0.set_index("unit").prov.reindex(P.b).values)
say(f"  {len(P)} adjacent pairs ({same.sum()} within a province, {(~same).sum()} across), districts in pairs {len(set(P.a) | set(P.b))}")
S = pd.concat([P[["pair", "a"]].rename(columns={"a": "unit"}), P[["pair", "b"]].rename(columns={"b": "unit"})]).merge(d0[["unit", "N", "reach", "lnpop", "prov"]], on="unit")
S["c"] = 0


def fitA(S, x):
    b, V = ppml(S.N.values.astype(float), np.c_[x, S.lnpop.values], S.pair.values, S.c.values, S.unit.values)
    return b[0], np.sqrt(V[0, 0])


bA, sA = fitA(S, S.reach.values)
import statsmodels.formula.api as smf
aux = smf.ols("reach ~ lnpop + C(prov)", data=d0).fit(); fv, e0 = aux.fittedvalues.values, aux.resid.values
idx = [np.where(d0.prov.values == k)[0] for k in d0.prov.unique()]
umap = pd.Series(np.arange(len(d0)), index=units); rowpos = umap.reindex(S.unit).values; bs = []
for _ in range(DRAWS):
    e = e0.copy()
    for ix in idx:
        e[ix] = RNG.permutation(e0[ix])
    bs.append(ppml(S.N.values.astype(float), np.c_[(fv + e)[rowpos], S.lnpop.values], S.pair.values, S.c.values, se=False)[0][0])
add("A", "travel time x2, neighbour pairs", bA, sA, (1 + np.sum(np.abs(bs) >= abs(bA))) / (1 + len(bs)), 2 * norm.sf(abs(bA / sA)), len(P))
Sw = S[S.pair.isin(P.pair[same])]
bW, sW = fitA(Sw, Sw.reach.values)
add("A", "travel time x2, pairs within a province", bW, sW, np.nan, 2 * norm.sf(abs(bW / sW)), int(same.sum()))
Gi = G.set_index("unit")   # boundaries are digitised separately, so shared edges are measured with a 200 m tolerance
P["edge_km"] = [Gi.geometry[x].boundary.intersection(Gi.geometry[y].buffer(200)).length / 1000 for x, y in zip(P.a, P.b)]
Ss = S[S.pair.isin(P.pair[P.edge_km >= 1])]
bS, sS = fitA(Ss, Ss.reach.values)
add("A", "travel time x2, pairs sharing >= 1 km of edge", bS, sS, np.nan, 2 * norm.sf(abs(bS / sS)), int((P.edge_km >= 1).sum()))

# ---------------- B distance vs detour ----------------
say("== B straight-line distance vs road detour")
C = pd.read_csv(PROC / "kh_cells_tt.csv")
cols = ["geonameid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
GN = pd.read_csv(ROOT / "data/raw/cambodia/geonames/KH.txt", sep="\t", names=cols, dtype={"pop": float}, quoting=3, low_memory=False)
GN = GN[(GN.fclass == "P") & (GN["pop"] >= 20000)]
tw = gpd.GeoSeries(gpd.points_from_xy(GN.lon, GN.lat), crs=4326).to_crs(32648); cl_ = gpd.GeoSeries(gpd.points_from_xy(C.lon, C.lat), crs=4326).to_crs(32648)
C["km"] = cKDTree(np.c_[tw.x, tw.y]).query(np.c_[cl_.x, cl_.y])[0] / 1000
Ekm = C.groupby("k").apply(lambda g: np.average(g.km, weights=g["pop"]), include_groups=False)
Tm = C.groupby("k").apply(lambda g: np.average(g.tt_town_dry, weights=g["pop"]), include_groups=False)
say(f"  towns {len(GN)} | check: cell-weighted travel time vs kh_access, max abs diff {np.nanmax(np.abs(Tm.reindex(units).values - d0.tt_town_dry.values)):.4f} min")
DF["dist"] = np.log2(DF.unit.map(Ekm).clip(lower=0.5)); DF["detour"] = np.log2(DF.tt_town_dry / DF.unit.map(Ekm).clip(lower=0.5))
d0 = counts(MAIN).reset_index(drop=True)
say(f"  median straight-line distance {d0.unit.map(Ekm).median():.1f} km, median minutes per km {np.median(d0.tt_town_dry / d0.unit.map(Ekm)):.2f} | "
    f"corr(dist, detour) {d0.dist.corr(d0.detour):+.2f}, corr(reach, dist) {d0.reach.corr(d0.dist):+.2f}, corr(reach, detour) {d0.reach.corr(d0.detour):+.2f}")
f, r = ns["fit"](d0, ["dist", "detour"])
for v, lab in [("dist", "straight-line distance x2 (detour fixed)"), ("detour", "road detour x2 (distance fixed)")]:
    p = ns["ri_p"](d0, v, [x for x in ["dist", "detour"] if x != v], f, r.params[v], DRAWS)
    add("B", lab, r.params[v], r.bse[v], p, r.pvalues[v], len(d0))
for v, lab in [("dist", "straight-line distance x2 alone"), ("detour", "road detour x2 alone")]:
    f1, r1 = ns["fit"](d0, [v]); p = ns["ri_p"](d0, v, [], f1, r1.params[v], min(DRAWS, 999))
    add("B", lab, r1.params[v], r1.bse[v], p, r1.pvalues[v], len(d0))

# ---------------- C field vs institutional, D EU/US vs China ----------------
s3 = pd.to_numeric(A0.drop_duplicates("project_id").set_index("project_id").sector_code.astype(str).str.extract(r"(\d{3})")[0], errors="coerce")
dg = A0.drop_duplicates("project_id").set_index("project_id").dgroup
FIELD = {112, 122, 130, 140, 311, 312, 313, 410, 430, 520, 730}
INST = {110, 111, 113, 114, 121, 150, 151, 152, 160, 240, 250, 320, 321, 331, 998}
INFRA = {210, 230, 231, 232, 233, 234, 235, 236}
typ = pd.Series(np.select([s3.isin(FIELD), s3.isin(INST), s3.isin(INFRA)], ["field", "inst", "infra"], "other"), index=s3.index)
M = MAIN.assign(type=MAIN.project_id.map(typ), dg=MAIN.project_id.map(dg))
say("== C field vs institutional projects")
say("  activities by type: " + ", ".join(f"{k} {M[M.type == k].project_id.nunique()} (weight {M[M.type == k].w.sum():.1f})" for k in ["field", "inst", "infra", "other"]))
for k, lab in [("field", "field projects alone"), ("inst", "institutional projects alone"), ("infra", "infrastructure projects alone")]:
    dk = counts(M[M.type == k]).reset_index(drop=True); fk, rk = ns["fit"](dk, ["reach"])
    add("C", lab + ", travel time x2", rk.params["reach"], rk.bse["reach"], ns["ri_p"](dk, "reach", [], fk, rk.params["reach"], min(DRAWS, 999)), rk.pvalues["reach"], round(dk.N.sum(), 1))
reach_of = d0.set_index("unit").reach


lnpop_of = d0.set_index("unit").lnpop; prov_of = d0.set_index("unit").prov


def gap_fit(Msub, lmap, levels, se=True, full=False):
    """Stack the two labels by district; district and label fixed effects; regressor = travel time for levels[0] only.
    full=True (post hoc check): also ln population for levels[0] and province x label fixed effects, so the gap is not
    a difference in population or province patterns that happens to correlate with travel time."""
    g = Msub.assign(L=Msub.project_id.map(lmap)).groupby(["unit", "L"]).w.sum().unstack(fill_value=0).reindex(index=units, columns=levels, fill_value=0)
    y = np.r_[g[levels[0]].values, g[levels[1]].values]; uu = np.r_[units, units]; tt = np.r_[np.zeros(len(units)), np.ones(len(units))]
    x = np.r_[reach_of.values, np.zeros(len(units))][:, None]
    if full:
        x = np.c_[x, np.r_[lnpop_of.values, np.zeros(len(units))]]
        tt = np.r_[prov_of.values, prov_of.values].astype(object) + "|" + tt.astype(int).astype(str)
    return ppml(y.astype(float), x, uu, tt, uu, se=se)


def gap(Msub, lab_col, levels, full=False):
    """Slope gap levels[0] minus levels[1]; randomisation inference permutes the label across projects."""
    proj = Msub.drop_duplicates("project_id").set_index("project_id")[lab_col]
    b, V = gap_fit(Msub, proj, levels, full=full); b, s = b[0], np.sqrt(V[0, 0]); bs = []
    for _ in range(DRAWS):
        bs.append(gap_fit(Msub, pd.Series(RNG.permutation(proj.values), index=proj.index), levels, se=False, full=full)[0][0])
    return b, s, (1 + np.sum(np.abs(bs) >= abs(b))) / (1 + len(bs))


Mc = M[M.type.isin(["field", "inst"])]
b, s, p = gap(Mc, "type", ["field", "inst"])
add("C", "slope gap field minus institutional, district FE", b, s, p, 2 * norm.sf(abs(b / s)), Mc.project_id.nunique())
b, s, p = gap(Mc, "type", ["field", "inst"], full=True)
add("C", "post hoc: same gap + ln pop gap + province x type FE", b, s, p, 2 * norm.sf(abs(b / s)), Mc.project_id.nunique())
for k in ("field", "inst"):   # post hoc: the 16 activities with a two-digit health/production code (120, 310) put in each group
    Mk = M.assign(type=M.type.replace("other", k)); Mk = Mk[Mk.type.isin(["field", "inst"])]
    b, s, p = gap(Mk, "type", ["field", "inst"])
    add("C", f"post hoc: gap with codes 120/310 counted as {k}", b, s, p, 2 * norm.sf(abs(b / s)), Mk.project_id.nunique())
say("== D EU/US vs China within the same district")
for k, lab in [("EU/US", "EU/US alone"), ("China", "China alone")]:
    dk = counts(M[M.dg == k]).reset_index(drop=True); fk, rk = ns["fit"](dk, ["reach"])
    add("D", lab + ", travel time x2", rk.params["reach"], rk.bse["reach"], ns["ri_p"](dk, "reach", [], fk, rk.params["reach"], min(DRAWS, 999)), rk.pvalues["reach"], round(dk.N.sum(), 1))
Md = M[M.dg.isin(["EU/US", "China"])]
b, s, p = gap(Md, "dg", ["China", "EU/US"])
add("D", "slope gap China minus EU/US, district FE", b, s, p, 2 * norm.sf(abs(b / s)), Md.project_id.nunique())
b, s, p = gap(Md, "dg", ["China", "EU/US"], full=True)
add("D", "post hoc: same gap + ln pop gap + province x donor FE", b, s, p, 2 * norm.sf(abs(b / s)), Md.project_id.nunique())
pd.DataFrame(rows).to_csv(T / "kh_designs.csv", index=False)
(T / "kh_designs.txt").write_text("\n".join(log), encoding="utf-8")
