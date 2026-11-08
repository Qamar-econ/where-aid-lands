"""Independent checks of the case studies (specs 9-32 to 9-34), run after them.

1  counterfactual engine: 'with' times reproduce kh_access (nearest town and Phnom Penh), in both kh_s41 and kh_s43 files
2  ferry substitution behaves: for districts behind a bridge, minutes added ~ F minus driving time over the bridge
3  9-34 outcome recomputed from the panel by hand (pre/post sums) and the ratio-of-ratios that PPML approximates
4  9-34 stability: leave one treated province out; leave one control province out (range); window +-3 years
5  9-33 province-FE capital model reproduces kh_main's travel-time estimate (same data, same FE)
Output: outputs/tables/kh_cases_verify.txt, kh_cases_verify.csv
"""
import os, numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; T = ROOT / "outputs/tables"; PROC = ROOT / "data/processed/cambodia"
log, rows = [], []
say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(x) for x in a)))
ACC = pd.read_csv(PROC / "kh_access.csv").set_index("k")

say("== 1 counterfactual engine reproduces kh_access")
B = pd.read_csv(T / "kh_bridges_dt.csv", index_col="unit"); PI = pd.read_csv(T / "kh_pinfra_dt.csv", index_col="unit")
for lab, a, b in [("kh_s41 town", B["town|with"], ACC.tt_town_dry), ("kh_s41 Phnom Penh", B["ktm|with"], ACC.tt_ktm_dry), ("kh_s43 Phnom Penh", PI["with"], ACC.tt_ktm_dry)]:
    d = np.nanmax(np.abs(a - b.reindex(a.index)))
    say(f"  {lab:<20s} max abs diff {d:.4f} min"); rows.append({"check": f"reproduce {lab}", "value": d, "ok": d < 0.01})

say("== 2 ferry substitution: added minutes in the most affected districts vs F - driving time over the bridge")
for name, F, drive in [("Neak Loeung bridge", 60, 2.7), ("Prek Tamak bridge", 60, 1.4), ("Neak Loeung bridge", 120, 2.7), ("Neak Loeung bridge", 30, 2.7)]:
    add = (PI[f"{name}|{F}"] - PI["with"]).max()
    say(f"  {name:<20s} F {F:>3d}: max added {add:6.1f} min (F - driving = {F - drive:.1f}; at most this unless a detour is shorter)")
    rows.append({"check": f"ferry {name} F{F}", "value": add, "ok": add <= F - drive + 0.5})

say("== 3 9-34 outcome by hand")
pan = pd.read_csv(T / "kh_pinfra_panel.csv")
res = pd.read_csv(T / "kh_pinfra.csv").set_index("model")
tp = tq = cp = cq = 0
for p, g in pan[pan.treated_by.notna()].groupby("province"):
    y0 = int(g.completed.iloc[0]); s = g.set_index("yr").activities
    pre, post = s.loc[y0 - 5:y0 - 1].sum(), s.loc[y0 + 1:y0 + 5].sum()
    c = pan[pan.control].groupby("yr").activities.sum(); cpre, cpost = c.loc[y0 - 5:y0 - 1].sum(), c.loc[y0 + 1:y0 + 5].sum()
    tp += pre; tq += post; cp += cpre; cq += cpost
    say(f"  {p:<11s} opened {y0}: treated {pre:.2f} -> {post:.2f} | controls {cpre:.1f} -> {cpost:.1f} (x{cpost / cpre:.2f})")
m = res.loc["main (all treated, F 60, >= 10%)"]
say(f"  pooled treated {tp:.2f} -> {tq:.2f} (kh_s43: {m.treated_pre:.2f} -> {m.treated_post:.2f}) | controls x{cq / cp:.2f} | "
    f"ratio of ratios {100 * ((tq / tp) / (cq / cp) - 1):+.0f}% vs PPML {m.pct:+.0f}%")
rows.append({"check": "9-34 pre/post by hand", "value": abs(tp - m.treated_pre) + abs(tq - m.treated_post), "ok": abs(tp - m.treated_pre) + abs(tq - m.treated_post) < 1e-6})

say("== 4 9-34 stability (999 draws each)")
src = (HERE / "kh_s43_province_infra.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "kh_s43_province_infra.py"), "__name__": "kh_s43_ns"}
os.environ.setdefault("PINFRA_DRAWS", "999")
exec(src[:src.index("\nrows = []\n")], ns)
assign, stack, fit, P = ns["assign"], ns["stack"], ns["fit"], ns["P"]
main = P[~P.family.isin(["ict", "infra"])]
tr, ctrl, _ = assign()
out = []
for drop in list(tr):
    r = fit(stack(main, {p: e for p, e in tr.items() if p != drop}, ctrl), 999); out.append(("drop treated", drop, r))
for drop in ctrl:
    r = fit(stack(main, tr, [c for c in ctrl if c != drop]), 0); out.append(("drop control", drop, r))
for lab, prov, r in out:
    rows.append({"check": f"{lab} {prov}", "value": r["pct"], "p_ri": r["p_ri"], "pre": r["treated_pre"], "post": r["treated_post"], "ok": True})
for lab, prov, r in out[:len(tr)]:
    say(f"  drop treated province {prov:>2d}: {r['pct']:+.0f}% (p RI {r['p_ri']:.3f}) | treated {r['treated_pre']:.1f} -> {r['treated_post']:.1f}")
dc = [r["pct"] for lab, _, r in out if lab == "drop control"]
say(f"  drop one control at a time ({len(dc)}): {min(dc):+.0f}% to {max(dc):+.0f}%")


def stack_w(frame, tr, ctrl, k):
    S = stack(frame, tr, ctrl)
    return S[S.ev.abs() <= k].reset_index(drop=True)


r3 = fit(stack_w(main, tr, ctrl, 3), 999)
say(f"  window -3..+3: {r3['pct']:+.0f}% (p RI {r3['p_ri']:.3f}) | treated {r3['treated_pre']:.1f} -> {r3['treated_post']:.1f}")
rows.append({"check": "window 3", "value": r3["pct"], "p_ri": r3["p_ri"], "pre": r3["treated_pre"], "post": r3["treated_post"], "ok": True})
up = sum(1 for _, _, r in out if r["pct"] > 0) + (r3["pct"] > 0)
say(f"  specifications with a positive estimate: {up} of {len(out) + 1}")

say("== 5 9-33 province-FE town model = kh_main travel-time estimate")
cap = pd.read_csv(T / "kh_capital.csv"); km = pd.read_csv(T / "kh_main_simple.csv")
a = cap[(cap.model == "province FE: town")].pct.iloc[0]; b = km[(km.model == "physical access alone") & (km["var"] == "reach")].pct.iloc[0]
say(f"  kh_s42 {a:+.3f}% vs kh_main {b:+.3f}%"); rows.append({"check": "9-33 province FE = kh_main", "value": abs(a - b), "ok": abs(a - b) < 1e-6})

V = pd.DataFrame(rows); V.to_csv(T / "kh_cases_verify.csv", index=False)
say(f"== {int(V.ok.sum())} of {len(V)} checks ok")
(T / "kh_cases_verify.txt").write_text("\n".join(log), encoding="utf-8")
