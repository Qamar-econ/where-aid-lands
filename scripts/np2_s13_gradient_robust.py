"""Robustness of the size gradient (road slope rises, connectivity slope falls with project size): is it a donor-type effect?
Re-uses the panel builder of np2_s12_heterogeneity.py. Output: outputs/results_np2_gradient_robust.txt
"""
import sys
from pathlib import Path

SRC = Path(__file__).with_name("np2_s12_heterogeneity.py").read_text(encoding="utf-8").split("# ---- concentration")[0]
exec(SRC)  # defines asg, panel(), IV, smf, sm, pd, ROOT


def grad(a, tag):
    st = []
    for r_, s_ in enumerate(["< $1M", "$1-10M", "> $10M"], 1):
        p = panel(a[a["size"] == s_]); p["grp"] = r_; p["gpt"] = p.pt + f"_{r_}"
        st.append(p)
    st = pd.concat(st)
    f = "N ~ " + " + ".join(f"{v} + {v}:grp" for v in IV) + " + lnpop + lnpop:grp + capital + C(gpt)"
    m = smf.glm(f, data=st, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(st.k)[0]}, maxiter=300)
    n = a.drop_duplicates("uid")["size"].value_counts().reindex(["< $1M", "$1-10M", "> $10M"]).fillna(0).astype(int).tolist()
    return f"{tag:<26} n(small/mid/large)={n} | " + " | ".join(
        f"{v}:size {m.params[v + ':grp']:+.2f} (p {m.pvalues[v + ':grp']:.3f})" for v in ["net_idx", "road"])


out = [grad(asg, "all"), grad(asg[asg.dclass != "MDB"], "non-MDB only"), grad(asg[asg.dclass == "bilateral"], "bilateral only"),
       grad(asg.assign(w=1.0), "binary counting"), grad(asg[asg.n_districts <= 10], "<= 10 districts")]
(ROOT / "outputs/results_np2_gradient_robust.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
