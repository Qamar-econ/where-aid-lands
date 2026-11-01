"""Dump text of each budget book (BB5) to data/interim/bb5_text/<year>.txt."""
import pdfplumber
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "bb5_text"
OUT.mkdir(parents=True, exist_ok=True)
for f in sorted((ROOT / "data" / "raw" / "budget_bb5").glob("BB5_*.pdf")):
    y = f.stem.split("_")[1]
    with pdfplumber.open(f) as pdf:
        txt = "\n".join(f"<<PAGE {i}>>\n" + (p.extract_text() or "") for i, p in enumerate(pdf.pages))
    (OUT / f"{y}.txt").write_text(txt, encoding="utf-8")
    print(y, len(pdf.pages), "pages", len(txt), "chars")
