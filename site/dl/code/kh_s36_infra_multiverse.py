"""Specification curve for the stacked re-test (spec 9-23): is the verdict an artefact of arbitrary choices? (spec 9-24)

Reuses the estimator and stacked panel of kh_s35. Every combination of
  event      GODAD transport | transport + energy | transport + major ADB/Japan roads | major roads only
  window     +-3, +-4, +-5 years          clean pre-period  3 or 5 years
  year 0     dropped or counted as post    event donors     excluded or kept in the outcome
  outcome    non-infrastructure | social   time FE          stack x year | stack x year x access tercile
= 384 specifications. Reports the distribution of the post effect, share significant, share ruling out +50%.
Output: outputs/tables/kh_infra_multiverse.csv, .txt; outputs/figures/cambodia/S7_infra_multiverse.png
"""
import itertools, numpy as np, pandas as pd, matplotlib
from scipy.stats import norm
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).with_name("kh_s35_infra_stacked.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("# ================= PRIMARY")[0], ns)
stack, did, T, FIG = ns["stack"], ns["did"], ns["T"], ns["FIG"]
TR, INF, MAJ, OUT = ns["TRANSPORT"], ns["INFRA"], ns["MAJOR"], ns["OUT"]
EVENTS = {"transport": TR, "transport+energy": INF, "transport+major": pd.concat([TR, MAJ]), "major only": MAJ}
OUTC = {"non-infra": OUT, "social": OUT[OUT.family == "social"]}
rows = []
for (ev, win, pre, k0, dd, oc, terc) in itertools.product(EVENTS, (3, 4, 5), (3, 5), (False, True), (True, False), OUTC, (False, True)):
    try:
        P = stack(EVENTS[ev], outcome=OUTC[oc], lo=-win, hi=win, pre=pre, keep0=k0, drop_donor=dd)
        b, V = did(P, terciles=terc); s = np.sqrt(V[0, 0])
        rows.append(dict(event=ev, window=win, pre=pre, year0=("post" if k0 else "dropped"), donors=("excluded" if dd else "kept"), outcome=oc,
                         fe=("tercile" if terc else "stack-year"), b=b[0], se=s, pct=100 * (np.exp(b[0]) - 1), p=2 * norm.sf(abs(b[0] / s)),
                         up90=100 * (np.exp(b[0] + 1.645 * s) - 1), mde=100 * (np.exp(2.8 * s) - 1), treated=P[P.treat == 1].groupby(["stk", "unit"]).ngroups))
    except Exception as e:
        print("failed", ev, win, pre, k0, dd, oc, terc, e)
R = pd.DataFrame(rows); R.to_csv(T / "kh_infra_multiverse.csv", index=False)
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
say(f"specifications estimated: {len(R)}")
for name, d in [("all", R)] + [(f"event = {e}", R[R.event == e]) for e in EVENTS] + [(f"outcome = {o}", R[R.outcome == o]) for o in OUTC]:
    say(f"  {name:<26s} n {len(d):>3d} | median effect {d.pct.median():+.0f}% (IQR {d.pct.quantile(.25):+.0f}% to {d.pct.quantile(.75):+.0f}%) | "
        f"positive {100 * (d.b > 0).mean():.0f}% | p<0.05 {100 * (d.p < .05).mean():.0f}% (positive & p<0.05 {100 * ((d.p < .05) & (d.b > 0)).mean():.0f}%, negative & p<0.05 {100 * ((d.p < .05) & (d.b < 0)).mean():.0f}%) | "
        f"rules out +50% {100 * (d.up90 < 50).mean():.0f}% | median MDE80 {d.mde.median():.0f}%")
say("  expected share p<0.05 under no effect anywhere: 5% (specifications are correlated, so the realised share varies)")
for f in ["window", "pre", "year0", "donors", "fe"]:
    say(f"  by {f}: " + " | ".join(f"{k}: median {g.pct.median():+.0f}%, p<0.05 {100 * (g.p < .05).mean():.0f}%" for k, g in R.groupby(f)))
S = R.sort_values("pct").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
col = np.where(S.p < .05, "#a8461f", "#0b6b6b")
lo, hi = 100 * (np.exp(S.b - 1.96 * S.se) - 1), 100 * (np.exp(S.b + 1.96 * S.se) - 1)
ax.vlines(S.index, lo.clip(-100, 600), hi.clip(-100, 600), color="#9a958a", lw=.4, alpha=.6)
ax.scatter(S.index, S.pct.clip(-100, 600), c=col, s=5, zorder=3)
ax.axhline(0, color="#24211b", lw=.8)
ax.set_xlabel(f"{len(S)} specifications, sorted by estimate (rust = p < 0.05)"); ax.set_ylabel("Later aid after infrastructure, %")
ax.set_title("Specification curve, stacked clean-event design (95% CI, clipped at +600%)", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIG / "S7_infra_multiverse.png"); plt.close(fig)
(T / "kh_infra_multiverse.txt").write_text("\n".join(log), encoding="utf-8")
