"""(1) Human double-coding sheet for Nepal geocodes + planning dates (stratified sample 200);
(2) figures: robustness forest plot (M1/M2 + R/D specs), observed planning->start lag distribution.
Outputs: data/manual/np2_validation_sheet.xlsx, outputs/figures/np2_fig1_forest.png, np2_fig2_planning_lag.png
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PL = pd.read_csv(ROOT / "data/processed/nepal/np2_planning.csv")
ST = pd.read_csv(ROOT / "data/processed/nepal/np2_project_status.csv")
G = pd.read_csv(ROOT / "data/processed/nepal/np2_geocodes.csv")
IDX = pd.read_csv(ROOT / "outputs/project_summaries/np2_summary_index.csv")
X = PL.merge(ST, on="uid")
X["summary_page"] = X.uid.map(IDX.set_index("uid").page)
units = G.groupby("uid").district.apply(lambda s: ", ".join(sorted(set(s))))
X["machine_districts"] = X.uid.map(units).fillna("")
rng = np.random.default_rng(7)
strata = [X[X.summary_page.notna() & (X.best_precision <= 4)], X[X.summary_page.notna() & (X.best_precision == 8)],
          X[X.summary_page.isna() & (X.best_precision <= 4)], X[X.summary_page.isna() & (X.best_precision == 8)]]
n = [60, 60, 40, 40]
S = pd.concat([s.sample(min(k, len(s)), random_state=rng.integers(1e6)) for s, k in zip(strata, n)])
cols = ["uid", "summary_page", "donor", "title", "usd", "start_date", "best_precision", "machine_districts", "planning_date", "method", "evidence"]
S = S[cols].assign(coder_precision="", coder_districts="", coder_central_institution="", coder_planning_date="", coder_planning_source="", coder_note="")
(ROOT / "data/manual").mkdir(exist_ok=True)
with pd.ExcelWriter(ROOT / "data/manual/np2_validation_sheet.xlsx") as xw:
    pd.DataFrame({"규칙": [
        "precision: 1 정확 좌표, 2 마을/소도시, 3 palika, 4 군, 6 주/지역, 8 전국·중앙기관",
        "districts: 사업이 실제로 수혜 대상으로 삼는 군만 (본부·사무소 주소 제외), 쉼표로 구분",
        "central_institution: 중앙부처 역량강화·예산지원 등 수도 소재 기관 대상이면 1",
        "planning_date: 개념 승인·PID·사업계획서·발굴 문서 등 실제 기획 착수 날짜 (없으면 공란)",
        "두 코더는 서로의 시트를 보지 말 것. 불일치는 제3자가 중재",
    ]}).to_excel(xw, sheet_name="codebook", index=False)
    S.to_excel(xw, sheet_name="coder1", index=False)
    S.to_excel(xw, sheet_name="coder2", index=False)
print("validation sample:", len(S))

# ---- Figure 1: forest plot
R = pd.read_csv(ROOT / "outputs/tables/np2_panel_results.csv")
D = []
for line in (ROOT / "outputs/results_np2_diagnostics.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("D1") and "net " in line:
        import re
        tag = line[:44].strip()
        b, se, p = map(float, re.search(r"net ([+-][\d.]+) \(se ([\d.]+), p ([\d.]+)\)", line).groups())
        D.append({"spec": "[post-hoc] " + tag, "term": "net_idx", "b": b, "se": se, "p": p})
F = pd.concat([R[R.term == "net_idx"][["spec", "term", "b", "se", "p"]], pd.DataFrame(D)], ignore_index=True)
F = F[~F.spec.str.startswith("R8 MDB")]
F = F.iloc[::-1].reset_index(drop=True)
INK, MUTED, SIG, NS = "#1f2328", "#6e7781", "#1f6feb", "#8c959f"
fig, ax = plt.subplots(figsize=(7.2, 0.24 * len(F) + 1.2), dpi=200)
for i, r in F.iterrows():
    c = SIG if r.p < 0.05 else NS
    lo, hi = 100 * (np.exp(r.b - 1.96 * r.se) - 1), 100 * (np.exp(r.b + 1.96 * r.se) - 1)
    ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
    ax.plot(100 * (np.exp(r.b) - 1), i, "o", ms=4.5, color=c, mec="white", mew=1)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(range(len(F))); ax.set_yticklabels(F.spec, fontsize=6.5, color=INK)
ax.set_xlabel("Change in new projects per +1 SD connectivity (%), 95% CI", fontsize=7.5, color=INK)
ax.tick_params(axis="x", labelsize=7, colors=MUTED)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e6e8eb", lw=0.6); ax.set_axisbelow(True)
ax.set_title("Nepal 2019–2024: connectivity at planning year → new development projects (PPML)", fontsize=8, color=INK, loc="left")
fig.tight_layout(rect=(0, 0.035, 1, 1)); (ROOT / "outputs/figures").mkdir(exist_ok=True)
fig.text(0.01, 0.01, "Blue: p < 0.05; grey: n.s.  M1 = province×year FE (between districts); M2 = district + year FE (within).",
         fontsize=6, color=MUTED)
fig.savefig(ROOT / "outputs/figures/np2_fig1_forest.png"); plt.close(fig)

# ---- Figure 2: observed planning lag
o = PL[PL.method.isin(["O1", "O1a", "O2"])].lag_years.dropna()
fig, ax = plt.subplots(figsize=(5.5, 2.6), dpi=200)
ax.hist(o, bins=np.arange(-1, 6.5, 0.25), color=SIG, edgecolor="white", lw=0.8)
ax.axvline(o.median(), color=INK, lw=1, ls="--")
ax.text(o.median() + 0.08, ax.get_ylim()[1] * 0.9, f"median {o.median():.2f} y (n = {len(o)})", fontsize=7, color=INK)
ax.set_xlabel("Years from documented planning start (ADB concept clearance / WB concept PID) to project start", fontsize=7, color=INK)
ax.set_ylabel("Projects", fontsize=7, color=INK); ax.tick_params(labelsize=7, colors=MUTED)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/np2_fig2_planning_lag.png"); plt.close(fig)
print("figures written")
