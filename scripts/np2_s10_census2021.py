"""Parse NPHC 2021 Annex 11.1 (households owning amenities, by district) from the NSO housing & households thematic report
(giwmscdnone.gov.np/media/app/public/36/posts/1710821571_6.pdf, report pp. 124-126) and harmonise to the 75 districts of 2011
(Nawalparasi East+West -> nawalparasi, Rukum East+West -> rukum).
Each district row has 19 numbers: total HH, none N, none %, >=1 N, >=1 %, radio, TV, landline, mobile (ordinary), mobile (smart),
computer, internet, car, motorcycle, bicycle, fan, refrigerator, washing machine, AC.
Checks: none + >=1 = total; shares in [0, 1]. Output: data/processed/nepal/census2021_district_hh.csv
"""
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pages = (ROOT / "data/raw/nepal/census2021/housing_households_report.pdf.txt").read_text(encoding="utf-8").split("=====PAGE=====")
i0 = next(i for i, p in enumerate(pages) if "Annex 11.1" in p and "Taplejung" in p)
txt = "\n".join(pages[i0:i0 + 4])
start = txt.find("Annex 11.1")
txt = txt[start:]
end = txt.find("Annex 11.2")
txt = txt[:end] if end > 0 else txt
COLS = ["hh", "none_n", "none_pct", "any_n", "any_pct", "radio", "tv", "landline", "mobile_ord", "mobile_smart", "computer", "internet",
        "car", "motorcycle", "bicycle", "fan", "fridge", "washer", "ac"]
D = pd.read_csv(ROOT / "data/processed/nepal/districts.csv")[["k", "district"]]
ALIAS = {"nawalparasi (east)": "nawalparasi", "nawalparasi (west)": "nawalparasi", "rukum (east)": "rukum", "rukum (west)": "rukum",
         "udayapur": "udaypur", "kapilbastu": "kapilvastu", "terhathum": "tehrathum"}
ALIAS.update({a: b for a, b in [("kavrepalanchok", "kavre"), ("tanahu", "tanahun"), ("dhanusha", "dhanusa")] if b in set(D.k)})
# tokenise: names are lines with letters; numbers may be glued ("162,107 203,416") or mistyped ("39,,118", "2,5647")
lines = [l.strip() for l in txt.splitlines() if l.strip()]
rows, cur, nums = [], None, []
HEADER = re.compile(r"(?i)^(no|amenities|/assets|at least|amenity|asset|radios|televisions|land line|telephones|mobile|computer|internet|cars|motorcycles|"
                    r"scooters|bicycles|electric|refrigerators|washing|air|districts|total|households|n$|%$|annex|\[\d+\]|koshi|madhesh|bagmati|gandaki|"
                    r"lumbini|karnali|sudurpashchim|nepal|machines|conditioners|fans|\(ordinary\)|\(smart\)|phones)")
for l in lines:
    if re.fullmatch(r"\((?i:east|west)\)", l) and cur and not nums:
        cur = cur + " " + l; continue
    if re.search(r"[A-Za-z]", l):
        if HEADER.match(l) and not re.match(r"(?i)^(nawalparasi|rukum)", l):
            continue
        if cur and len(nums) >= 19:
            rows.append((cur, nums[:19]))
        cur, nums = l.strip(), []
    elif cur:
        for t in l.split():
            t = re.sub(r",+", "", t)
            try:
                nums.append(float(t))
            except ValueError:
                pass
if cur and len(nums) >= 19:
    rows.append((cur, nums[:19]))
R = pd.DataFrame([dict(zip(COLS, v), name=n) for n, v in rows])
R["key"] = R.name.str.lower().str.strip().map(lambda s: ALIAS.get(s, s))
R["check_sum"] = (R.none_n + R.any_n - R.hh).abs() <= 2
print("parsed rows:", len(R), "| sum check failures:", (~R.check_sum).sum())
print(R.loc[~R.check_sum, ["name", "hh", "none_n", "any_n"]].to_string())
H = R.groupby("key")[["hh"] + COLS[5:]].sum().reset_index().rename(columns={"key": "k"})
for c in ["mobile_ord", "mobile_smart", "computer", "internet", "tv", "radio"]:
    H[c + "_pct21"] = 100 * H[c] / H.hh
H = D.merge(H, on="k", how="left")
miss = H[H.hh.isna()].k.tolist()
print("unmatched 2011 districts:", miss, "| unmatched parsed names:", sorted(set(R.key) - set(D.k)))
bad = H[(H.internet_pct21 > 100) | (H.internet_pct21 < 0)]
print("share out of range:", len(bad))
H.to_csv(ROOT / "data/processed/nepal/census2021_district_hh.csv", index=False, encoding="utf-8-sig")
nat = 100 * H.internet.sum() / H.hh.sum()
print(f"national HH internet share (should be ~37.8%): {nat:.1f}% | total HH (should be ~6.66M): {H.hh.sum():,.0f}")
print(H.sort_values("internet_pct21")[["k", "internet_pct21"]].iloc[[0, 1, 2, -3, -2, -1]].to_string(index=False))
