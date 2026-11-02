"""Compare a table you rebuilt with the published one:  python kh_compare.py yours.csv published.csv [key_column]

Rows are matched on the key column (default: the first column both files share among unit, k, project_id). For every
numeric column present in both, prints the number of matched rows, the largest absolute difference and the correlation.
Use it after each stage of the pipeline, e.g. your kh_access.csv against the downloaded kh_access.csv.
"""
import sys
import numpy as np, pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
key = sys.argv[3] if len(sys.argv) > 3 else next(c for c in ["unit", "k", "project_id"] if c in a and c in b)
m = a.merge(b, on=key, suffixes=("_yours", "_published"))
print(f"key = {key} | rows: yours {len(a)}, published {len(b)}, matched {len(m)}")
rows = []
for c in sorted(set(a.columns) & set(b.columns) - {key}):
    x, y = m[f"{c}_yours"], m[f"{c}_published"]
    if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
        rows.append({"column": c, "max_abs_diff": None, "corr": None, "identical": bool((x.astype(str) == y.astype(str)).all())}); continue
    ok = x.notna() & y.notna()
    rows.append({"column": c, "max_abs_diff": float(np.abs(x[ok] - y[ok]).max()) if ok.any() else None,
                 "corr": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else None,
                 "identical": bool(np.allclose(x[ok], y[ok], rtol=1e-9, atol=1e-12))})
print(pd.DataFrame(rows).to_string(index=False))
