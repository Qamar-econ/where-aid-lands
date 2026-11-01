"""Country-agnostic rule-based geoparser for aid project texts.

A country is described by a config:
    unit_variants  {unit_code: [spellings]}        analysis-unit names (e.g. TL postos, Nepal palikas)
    upper_variants {upper_code: [spellings]}       higher admin level (TL municipalities, Nepal districts)
    unit_parent    {unit_code: upper_code}
    case_sensitive set of spellings matched case-sensitively (short / common-word names)
    custom         {spelling: compiled regex}      fully custom patterns (e.g. 'Same' with stop-words)
    hq_names       spellings whose mentions are often HQ / office references (capital city)
Each match is returned with its context so decisions are auditable.

Decision rules (documented in the codebook):
  R1  unit-level mention > upper-level mention of the same parent (the upper one is dropped)
  R2  mentions inside an HQ / address context are discarded (e.g. "based in Dili", "UN House, ... Dili", "PO Box")
  R3  tier: unit-level name -> 'B'; upper-level name -> 'C'
"""
import re
import unicodedata
from dataclasses import dataclass, field

HQ_BEFORE = re.compile(r"(?:\b(?:based|headquartered|headquarters|office|offices|located|registered|sede|escritório|"
                       r"com sede|seated)\s+(?:in|at|em|no|na)\s*(?:the\s+)?(?:capital\s+)?(?:city\s+of\s+)?$)|"
                       r"(?:\b(?:UN House|Rua|Street|St\.|Avenida|Av\.|Road|Jalan|Building|Hotel|Office no\.?|Kiu Kai)[^|.;]{0,40}$)", re.I)
HQ_AFTER = re.compile(r"^[\s,]*(?:\d{3,5}\b|(?:Timor-Leste|East Timor|Nepal)\s*(?:[,|]|\s)*(?:P\.?\s?O\.?\s?Box|Tel|Phone|\+\d))|"
                      r"^[\s-]*based\b|^\s*(?:office|headquarters|HQ|UN House)\b", re.I)


def norm(s):
    return unicodedata.normalize("NFC", str(s))


def default_pattern(name, case_sensitive=False):
    if re.search(r"[ऀ-ॿ぀-ヿ一-鿿]", name):     # Devanagari / Japanese / CJK
        return re.compile(re.escape(name))
    n = re.escape(norm(name)).replace(r"\-", r"[-\s]?").replace(r"\ ", r"[-\s]")
    return re.compile(r"\b" + n + r"\b", 0 if case_sensitive else re.IGNORECASE)


@dataclass
class Match:
    level: str      # 'unit' or 'upper'
    code: str
    term: str
    start: int
    context: str
    hq: bool = False


@dataclass
class Geocoder:
    unit_variants: dict
    upper_variants: dict
    unit_parent: dict
    case_sensitive: set = field(default_factory=set)
    custom: dict = field(default_factory=dict)
    hq_names: set = field(default_factory=set)

    def __post_init__(self):
        def pat(v):
            return self.custom.get(v) or default_pattern(v, v in self.case_sensitive)
        self._u = [(c, v, pat(v)) for c, vs in self.unit_variants.items() for v in vs]
        self._p = [(c, v, pat(v)) for c, vs in self.upper_variants.items() for v in vs]

    def matches(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        t = norm(text)
        out = []
        for level, table in (("unit", self._u), ("upper", self._p)):
            for code, v, p in table:
                for m in p.finditer(t):
                    before, after = t[max(0, m.start() - 60):m.start()], t[m.end():m.end() + 40]
                    hq = bool(HQ_BEFORE.search(before) or HQ_AFTER.search(after))
                    out.append(Match(level, code, v, m.start(), t[max(0, m.start() - 50):m.end() + 50], hq))
        return out

    def parse(self, text):
        """-> (set unit codes, set upper codes, list of kept Match, list of discarded HQ Match)"""
        ms = self.matches(text)
        kept = [m for m in ms if not m.hq]
        hq = [m for m in ms if m.hq]
        units = {m.code for m in kept if m.level == "unit"}
        uppers = {m.code for m in kept if m.level == "upper"} - {self.unit_parent[u] for u in units}   # R1
        return units, uppers, kept, hq
