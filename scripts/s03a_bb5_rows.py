"""Extract project rows from Budget Book 5 (2013-2025) tables.

Each table row -> text (non-numeric cells), development partner guess, first planned amount.
Also parses the official 'External Assistance by Municipality' counts (2016, 2017).
Outputs data/interim/bb5_rows.csv, data/interim/bb5_municipal_counts.csv
"""
import re
import pdfplumber, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, INT = ROOT / "data" / "raw", ROOT / "data" / "interim"

DONORS = {  # canonical -> regex
    "Korea (KOICA)": r"KOICA|Korea", "Japan (JICA)": r"JICA|Japan", "Australia": r"Australia|DFAT|AusAID",
    "New Zealand": r"New Zealand|\bNZ\b|MFAT", "USA": r"USAID|United States|\bUSA\b|\bMCC\b|Millennium Challenge",
    "European Union": r"\bEU\b|European Union|European Commission", "Portugal": r"Portugal|Cam[õo]es|IPAD",
    "Germany": r"German|GIZ|KfW|BMZ", "China": r"China|Chinese", "World Bank": r"World Bank|\bWB\b|\bIDA\b|\bIFC\b",
    "ADB": r"\bADB\b|Asian Development Bank", "UNDP": r"UNDP", "UNICEF": r"UNICEF", "WFP": r"\bWFP\b|World Food",
    "WHO": r"\bWHO\b|World Health", "FAO": r"\bFAO\b", "UNFPA": r"UNFPA", "ILO": r"\bILO\b", "UN Women": r"UN ?Women",
    "IOM": r"\bIOM\b", "Global Fund": r"Global Fund", "GAVI": r"GAVI|Gavi", "Brazil": r"Brazil", "Ireland": r"Irish|Ireland",
    "Norway": r"Norw", "Finland": r"Finland", "Spain": r"Spain|AECID", "Cuba": r"Cuba", "India": r"\bIndia\b",
    "Indonesia": r"Indonesia", "Thailand": r"Thailand|TICA", "Singapore": r"Singapore", "France": r"France|AFD",
    "UNESCO": r"UNESCO", "UNHCR": r"UNHCR", "UNOPS": r"UNOPS", "IFAD": r"IFAD", "GEF": r"\bGEF\b|Global Environment",
    "Timor-Leste Gov": r"$^",
}
DRX = [(k, re.compile(v)) for k, v in DONORS.items() if v != "$^"]
NUM = re.compile(r"^\(?-?[\d,\.]+\)?$|^-$")


def donors_in(t):
    return [k for k, r in DRX if r.search(t)]


rows = []
for f in sorted((RAW / "budget_bb5").glob("BB5_*.pdf")):
    year = int(f.stem.split("_")[1])
    if year < 2013:
        continue
    with pdfplumber.open(f) as pdf:
        for pi, page in enumerate(pdf.pages):
            for tb in page.extract_tables():
                for r in tb:
                    cells = [(c or "").replace("\n", " ").strip() for c in r]
                    txt = [c for c in cells if c and not NUM.match(c)]
                    nums = [c for c in cells if c and NUM.match(c) and c != "-"]
                    text = " | ".join(txt)
                    if len(text) < 40 or re.match(r"^(Project Title|Development|TOTAL|Total)", text):
                        continue
                    amt = None
                    for c in nums:
                        try:
                            amt = float(c.replace(",", "").strip("()")); break
                        except ValueError:
                            pass
                    rows.append({"book_year": year, "page": pi, "text": text, "donors": "; ".join(donors_in(text)),
                                 "amount_k": amt})
df = pd.DataFrame(rows)
# merge continuation rows (no donor & no amount) into previous row on same page
out = []
for r in df.to_dict("records"):
    if out and not r["donors"] and r["amount_k"] is None and out[-1]["page"] == r["page"] and out[-1]["book_year"] == r["book_year"]:
        out[-1]["text"] += " " + r["text"]
    else:
        out.append(r)
df = pd.DataFrame(out)
df["title"] = df.text.str.split(" \\| ").map(lambda xs: max(xs[:3], key=len)[:160] if xs else "")
df.to_csv(INT / "bb5_rows.csv", index=False, encoding="utf-8-sig")
print("BB5 rows", len(df), df.groupby("book_year").size().to_dict())
print("rows with donor", (df.donors != "").mean().round(2))

# official municipal counts / amounts
mc = []
for y in [2016, 2017]:
    t = (INT / "bb5_text" / f"{y}.txt").read_text(encoding="utf-8")
    sec = t[t.find("External Assistance by Municipality", t.find("External Assistance by Municipality") + 50):]
    for m in re.finditer(r"5\.3\.\d+\s+([A-Za-zçé\-]+)\s+Municipality\s+\((\d+)\s+projects\)(.{0,400})", sec, re.S):
        usd = re.search(r"(?:US\$|USD\$|USS\$|\$)\s*([\d\.]+)\s*million", m.group(3))
        mc.append({"year": y, "municipio": m.group(1), "n_projects": int(m.group(2)),
                   "usd_m": float(usd.group(1)) if usd else None})
mc = pd.DataFrame(mc)
mc.to_csv(INT / "bb5_municipal_counts.csv", index=False)
print(mc.to_string())
