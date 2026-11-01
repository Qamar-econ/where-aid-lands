"""Numbers and figures for the study guide: the same data analysed with tools the reader knows (OLS) and then with
the tools the study uses (fixed effects, PPML). Writes study/fig_*.png, study/eq_*.png and study/numbers.json."""
import json
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "study"
__file__s04 = ROOT / "scripts/kh_s04_models.py"
src = __file__s04.read_text(encoding="utf-8")
g = {"__file__": str(__file__s04)}
exec(src[:src.index("\nbase = A[A.level")], g)
DF, A, outcome = g["DF"], g["A"], g["outcome"]
d = outcome(A[A.level == "adm2"])
font_manager.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
INK, TEAL, RUST, GREY = "#24211b", "#0b6b6b", "#a8461f", "#9a958a"

cl = lambda f, dd: smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]})
num = {}
r1 = smf.ols("N ~ phys", data=d).fit()
num["ols_simple"] = {"b0": r1.params["Intercept"], "b1": r1.params["phys"], "se": r1.bse["phys"], "p": r1.pvalues["phys"], "r2": r1.rsquared}
r2 = smf.ols("N ~ phys + lnpop", data=d).fit()
num["ols_pop"] = {"b1": r2.params["phys"], "p": r2.pvalues["phys"], "r2": r2.rsquared}
r3 = cl("N ~ phys + lnpop + C(prov)", d)
num["ols_fe"] = {"b1": r3.params["phys"], "p": r3.pvalues["phys"], "r2": r3.rsquared}
num["n_districts"], num["n_zero"], num["mean_N"], num["max_N"] = len(d), int((d.N == 0).sum()), d.N.mean(), d.N.max()
pos = d[d.N > 0]
r4 = smf.ols("np.log(N) ~ phys + lnpop + C(prov)", data=pos).fit()
num["log_ols_dropping_zeros"] = {"n_used": len(pos), "b1": r4.params["phys"]}
r5 = smf.glm("N ~ phys + lnpop", data=d, family=sm.families.Poisson()).fit()
r6 = smf.glm("N ~ phys + lnpop + C(prov)", data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]})
num["ppml_nofe"] = {"b": r5.params["phys"], "pct": 100 * (np.exp(r5.params["phys"]) - 1)}
num["ppml_fe"] = {"b": r6.params["phys"], "pct": 100 * (np.exp(r6.params["phys"]) - 1), "se": r6.bse["phys"], "p": r6.pvalues["phys"], "theta": r6.params["lnpop"]}
num["sd_phys_raw"] = float(DF.access_pc1.std())
num["n_prov"] = int(d.prov.nunique())
(OUT / "numbers.json").write_text(json.dumps(num, indent=1, default=float), encoding="utf-8")
print(json.dumps(num, indent=1, default=float))

# fig 1: OLS line on the raw scatter (what the reader already knows)
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
ax.scatter(d.phys, d.N, s=14, color=TEAL, alpha=.6, edgecolor="none")
xs = np.linspace(d.phys.min(), d.phys.max(), 50)
ax.plot(xs, r1.params["Intercept"] + r1.params["phys"] * xs, color=RUST, lw=2, label="OLS 직선")
ax.plot(xs, np.exp(np.log(d.N.mean()) - r5.params["phys"] * d.phys.mean() + r5.params["phys"] * xs) * 1, color=INK, lw=1.6, ls="--", label="지수 곡선 (포아송)")
ax.axhline(0, color=GREY, lw=.8)
ax.set_xlabel("도로 접근성 지수 (표준화, SD)"); ax.set_ylabel("군의 신규 사업 수 N")
ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_scatter.png"); plt.close(fig)

# fig 2: many zeros
fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=200)
ax.hist(d.N, bins=np.arange(0, d.N.max() + 1, 1), color=TEAL, edgecolor="white")
ax.set_xlabel("군의 신규 사업 수 N (1 단위 구간)"); ax.set_ylabel("군 개수")
ax.annotate(f"사업 0인 군: {num['n_zero']}개", xy=(0.5, num["n_zero"]), xytext=(6, num["n_zero"] * .8), arrowprops=dict(arrowstyle="->", color=RUST), color=RUST)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "fig_zeros.png"); plt.close(fig)

# fig 3: fixed effects = comparing within each province (two example provinces)
fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
cols = [TEAL, RUST, "#5b6a80"]
for c, (k, gg) in zip(cols, d.groupby("prov").filter(lambda x: len(x) >= 9).groupby("prov")):
    ax.scatter(gg.phys, gg.N / gg.pop_2019 * 1e5, s=18, color=c, label={"2": "바탐방 주", "14": "프레이벵 주", "17": "시엠립 주"}.get(k, f"주 {k}"))
    z = np.polyfit(gg.phys, gg.N / gg.pop_2019 * 1e5, 1); xx = np.linspace(gg.phys.min(), gg.phys.max(), 10)
    ax.plot(xx, np.polyval(z, xx), color=c, lw=1.4)
ax.set_xlabel("도로 접근성 지수 (SD)"); ax.set_ylabel("인구 10만 명당 사업 수")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_within.png"); plt.close(fig)

# equations rendered with mathtext
EQ = {"ols": r"$y_i = \beta_0 + \beta_1 x_i + u_i$",
      "fe": r"$N_d = \alpha_{p(d)} + \beta\, X_d + \theta \ln \mathrm{Pop}_d + u_d$",
      "ppml": r"$\mathbb{E}[N_d] = \exp(\alpha_{p(d)} + \beta\, X_d + \theta \ln \mathrm{Pop}_d)$",
      "pct": r"$\%\ \mathrm{change} = 100\,(e^{\beta} - 1)$",
      "z": r"$z_d = (x_d - \bar{x})\,/\,s_x$",
      "pca": r"$S_d = \sum_{k=1}^{10} w_k\, z_{dk}$",
      "event": r"$\mathbb{E}[N_{dt}] = \exp(\mu_d + \lambda_t + \sum_{k \neq -1} \beta_k \mathbf{1}[t - g_d = k])$",
      "shapley": r"$\phi_j = \mathrm{average\ over\ orderings\ of}\ [v(S \cup \{j\}) - v(S)]$"}
for k, e in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01), dpi=220); fig.text(0, 0, e, fontsize=15, color=INK)
    fig.savefig(OUT / f"eq_{k}.png", bbox_inches="tight", pad_inches=0.06, transparent=True); plt.close(fig)
print("figures done")
