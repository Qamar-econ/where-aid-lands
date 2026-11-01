"""Country-independent ODA coding helpers (sector taxonomy, keyword classifier, donor groups, ICT flag, de-dup).
Same definitions as used for Timor-Leste in s03_oda.py (codebook section 4.4)."""
import re
import pandas as pd
from rapidfuzz import fuzz

CAT = {}
for c in [210]: CAT[c] = "ACCESS_transport"
for c in [220]: CAT[c] = "ACCESS_comms"
for c in [230, 231, 232, 233, 234, 235, 236]: CAT[c] = "ACCESS_energy"
for c in [110, 111, 112, 113, 114]: CAT[c] = "SOC_education"
for c in [120, 121, 122, 123, 130]: CAT[c] = "SOC_health"
for c in [140]: CAT[c] = "SOC_water"
for c in [160]: CAT[c] = "SOC_other"
for c in [150, 151, 152]: CAT[c] = "GOV"
for c in [240, 250, 310, 311, 312, 313, 320, 321, 322, 323, 331, 332]: CAT[c] = "PROD"
for c in [410, 430]: CAT[c] = "ENV_MULTI"
for c in [700, 720, 730, 740, 520]: CAT[c] = "HUMAN"
for c in [510, 530, 600, 910, 920, 930, 998]: CAT[c] = "OTHER"
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}

KW = [
    ("ACCESS_transport", r"\broads?\b|\bbridges?\b|highway|\btransport(ation)?\b|\bports?\b|seaport|airport|ferry|maritime|aviation|runway|trail bridge"),
    ("ACCESS_energy", r"electric|power plant|\benergy|solar|hydro ?power|grid\b|electrification|generator|transmission line"),
    ("ACCESS_comms", r"telecom|internet|broadband|\bICT\b|digital|submarine cable|fib(re|er) optic|e-government|connectivity"),
    ("HUMAN", r"humanitarian|emergency|disaster response|flood response|relief|refugee|covid|pandemic response|earthquake (response|relief)"),
    ("SOC_water", r"\bwater\b|sanitation|\bWASH\b|hygiene|irrigation"),
    ("SOC_health", r"health|hospital|clinic|malaria|tuberculosis|\bHIV|nutrition|maternal|immuni[sz]|vaccin|medical|disease"),
    ("SOC_education", r"educat|school|teacher|university|scholarship|training|TVET|vocational|literacy|curricul"),
    ("PROD", r"agricult|farm|coffee|livestock|fisher|forest|rural development|market|business|enterprise|private sector|tourism|trade|employment|microfinance|value chain|food security"),
    ("GOV", r"governance|justice|police|parliament|elect|public financ|decentrali|federalism|civil service|capacity building|anti-corruption|audit|statistic|security sector|rule of law|human rights|gender|peace"),
    ("ENV_MULTI", r"climate|environment|biodiversity|resilien|disaster risk|reconstruction"),
]
KWX = [(c, re.compile(p, re.I)) for c, p in KW]


def kw_cat(text):
    for c, p in KWX:
        if isinstance(text, str) and p.search(text):
            return c
    return "OTHER"


def code_cat(code, text):
    c = CAT.get(int(code) if pd.notna(code) else -1)
    if c is None or c in ("OTHER", "ENV_MULTI"):
        k = kw_cat(text)
        if c is None or k.startswith("ACCESS") or c == "OTHER":
            return k
    return c


DONOR_GROUP = [("Korea", r"Korea|KOICA"), ("Japan", r"Japan|JICA"), ("Australia", r"Australia|DFAT|AusAID|AusAid"),
               ("USA", r"United States|USAID|U\.S\.|\bUSA\b|Millennium Challenge|Department of State|Peace Corps|Department of Agriculture"),
               ("UK", r"United Kingdom|DFID|FCDO|\bUK\b|Foreign, Commonwealth"), ("EU", r"European|\bEU\b"),
               ("Switzerland", r"Swiss|Switzerland|SDC"), ("Norway", r"Norw"), ("Germany", r"German|BMZ|GIZ|KfW"), ("China", r"China|Chinese"),
               ("India", r"\bIndia\b"), ("World Bank", r"World Bank|International Development Association|\bIDA\b|IFC|International Finance Corporation"),
               ("ADB", r"Asian Development Bank|\bADB\b"),
               ("UN agencies", r"United Nations|UNDP|UNICEF|WFP|World Food|WHO|World Health|FAO|Food and Agriculture|UNFPA|ILO|International Labour|UN Women|UNESCO|IOM|International Organization for Migration|UNHCR|UNOPS|UNIDO|OCHA|CERF|UN-Habitat|UN Pooled|UNCDF"),
               ("Global funds", r"Global Fund|Gavi|GAVI|Global Environment|GEF|IFAD|CGIAR|Green Climate"),
               ("NGO/foundation", r"Save the Children|Oxfam|CARE|Plan |World Vision|Caritas|GlobalGiving|Foundation|Helvetas|Practical Action|WaterAid|Mercy|Red Cross")]
DGX = [(g, re.compile(p)) for g, p in DONOR_GROUP]
BILAT = {"Korea", "Japan", "Australia", "USA", "UK", "EU", "Switzerland", "Norway", "Germany", "China", "India", "Other bilateral/other"}


def donor_group(d):
    for g, p in DGX:
        if isinstance(d, str) and p.search(d):
            return g
    return "Other bilateral/other"


def norm_title(t):
    t = re.sub(r"[^a-z0-9 ]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


def dup_of(title, dgroup, year, pool, thr=88):
    nt = norm_title(title)
    if len(nt) <= 12:
        return False
    cand = pool[pool.donor_group == dgroup]
    if pd.notna(year):
        cand = cand[cand.start_year.isna() | ((cand.start_year - year).abs() <= 2)]
    return any(fuzz.token_set_ratio(nt, t) >= thr for t in cand.nt)


ICT = re.compile(r"\bICTs?\b|information and communications? technolog|\bdigital|e-?learning|online (learning|platform|system|service)|"
                 r"\binternet\b|computer|e-?government|e-?governance|e-?justice|e-?health|e-?procurement|information (management )?systems?\b|"
                 r"\bdatabase|software|mobile (phone|money|app|application|banking|network)|\bSMS\b|tele-?medicine|tele-?health|"
                 r"broadband|connectivity|\bEMIS\b|\bHMIS\b|DHIS ?2|\bIFMIS\b|\bGIS\b|web[- ]based|digitali[sz]|telecommunication", re.I)
