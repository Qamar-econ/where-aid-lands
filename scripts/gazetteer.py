"""Place-name gazetteer and text geoparser for Timor-Leste postos / municipalities.

Variants are keyed by the official COD-AB posto / municipality name and mapped to pcodes
from the COD shapefile, so codes always match the boundary layer.
Names shared by a municipality and its capital posto (e.g. "Baucau") are read as the
municipality unless written as a town ("Baucau town", "Baucau Vila"), because project
documents use them overwhelmingly in the municipal sense.
"""
import re
import unicodedata
from pathlib import Path

import geopandas as gpd

_COD = Path(__file__).resolve().parents[1] / "data" / "raw" / "cod_ab" / "tls_admin_boundaries.shp"

# official posto name -> spelling variants (posto-specific names only)
POSTO_BY_NAME = {
    "Aileu Vila": ["Aileu Vila", "Aileu town"], "Laulara": ["Laulara"], "Liquidoe": ["Lequidoe", "Liquidoe", "Liquidoi"],
    "Remexio": ["Remexio"],
    "Ainaro": ["Ainaro Vila", "Ainaro town"], "Hatu-Builico": ["Hatu-Builico", "Hatubuilico", "Hato-Builico", "Hatu Builico"],
    "Hatu-Udo": ["Hatu-Udo", "Hatudo", "Hato-Udo", "Hatu Udo"], "Maubisse": ["Maubisse"],
    "Baguia": ["Baguia"], "Baucau": ["Baucau Vila", "Baucau town", "Baucau city"], "Laga": ["Laga"],
    "Quelicai": ["Quelicai", "Kelikai"], "Vemasse": ["Vemasse"], "Venilale": ["Venilale"],
    "Atabae": ["Atabae"], "Balibo": ["Balibo", "Balibó"], "Bobonaro": ["Bobonaro Vila", "Bobonaro town"],
    "Cailaco": ["Cailaco"], "Lolotoe": ["Lolotoe"], "Maliana": ["Maliana"],
    "Fatululic": ["Fatululic", "Fatululik"], "Fatumean": ["Fatumean"], "Forohem": ["Fohorem", "Forohem"],
    "Maukatar": ["Maucatar", "Maukatar"], "Suai": ["Suai"], "Tilomar": ["Tilomar"], "Zumalai": ["Zumalai"],
    "Atauro": ["Atauro", "Ataúro", "Atauru"], "Cristo Rei": ["Cristo Rei", "Becora", "Hera"],
    "Dom Aleixo": ["Dom Aleixo", "Comoro", "Bebonuk", "Bairro Pite"], "Metinaro": ["Metinaro"],
    "Nain Feto": ["Nain Feto", "Lahane"], "Vera Cruz": ["Vera Cruz"],
    "Atsabe": ["Atsabe"], "Ermera": ["Ermera Vila", "Gleno", "Emera Vila"], "Hatolia": ["Hatolia"],
    "Letefoho": ["Letefoho"], "Railaco": ["Railaco"],
    "Bazartete": ["Bazartete", "Tibar"], "Liquiçá": ["Liquiçá Vila", "Liquica town", "Liquiçá town"], "Maubara": ["Maubara"],
    "Iliomar": ["Iliomar"], "Lautém": ["Moro", "Lautem Vila", "Lautém Vila"], "Lospalos": ["Lospalos", "Los Palos"],
    "Luro": ["Luro"], "Tutuala": ["Tutuala"],
    "Alas": ["Alas"], "Fatuberliu": ["Fatuberliu", "Fatuberlio"], "Same": ["Same"], "Turiscai": ["Turiscai"],
    "Barique/Natarbora": ["Barique", "Natarbora"], "Laclo": ["Laclo"], "Laclubar": ["Laclubar"], "Laleia": ["Laleia"],
    "Manatuto": ["Manatuto Vila", "Manatuto town"], "Soibada": ["Soibada"],
    "Nitibe": ["Nitibe"], "Oesilo": ["Oesilo"], "Pante Macassar": ["Pante Macassar", "Pante Makasar", "Pante Makassar"],
    "Passabe": ["Passabe", "Pasabe"],
    "Lacluta": ["Lacluta"], "Ossu": ["Ossu"], "Uatucarbau": ["Uatucarbau", "Uato-Carbau", "Uatu-Carbau"],
    "Uato-Lari": ["Uato-Lari", "Uatolari", "Watulari", "Uato Lari"], "Viqueque": ["Viqueque Vila", "Viqueque town"],
}
MUN_BY_NAME = {
    "Aileu": ["Aileu"], "Ainaro": ["Ainaro"], "Baucau": ["Baucau"], "Bobonaro": ["Bobonaro"],
    "Covalima": ["Covalima", "Cova Lima", "Kovalima"], "Dili": ["Dili", "Díli"], "Ermera": ["Ermera"],
    "Liquiçá": ["Liquiçá", "Liquica", "Likisa"], "Lautém": ["Lautém", "Lautem"], "Manufahi": ["Manufahi"],
    "Manatuto": ["Manatuto"], "Oecussi": ["Oecusse", "Oecussi", "Oe-cusse", "Oe-Cusse", "Ambeno", "RAEOA", "ZEESM"],
    "Viqueque": ["Viqueque"],
}

# facilities / landmarks / non-Latin spellings -> posto (official name)
LANDMARK_POSTO = {
    "Dom Aleixo": ["Nicolau Lobato International Airport", "Presidente Nicolau Lobato", "Dili airport", "Dili International Airport",
                   "Delta Comoro", "Tasitolu", "Taci Tolu", "Guido Valadares", "HNGV", "National Hospital"],
    "Vera Cruz": ["Palácio do Governo", "Government Palace"],
    "Nain Feto": ["Dili port", "Port of Dili", "Dili Port", "UNTL", "National University of Timor"],
    "Bazartete": ["Tibar Bay", "Tibar port", "Tibar Port"],
    "Suai": ["Suai Supply Base", "Suai airport", "スアイ"], "Same": ["Betano"], "Viqueque": ["Beaco"],
    "Baucau": ["Cakung", "Baucau airport", "バウカウ市"], "Pante Macassar": ["Oecusse airport", "Oecussi airport"],
    "Laleia": ["Buluto"], "Maliana": ["マリアナ"], "Lospalos": ["ロスパロス"], "Atauro": ["Beloi", "アタウロ"],
}
MUN_EXTRA = {"Dili": ["ディリ"], "Ermera": ["Emera"],"Baucau": ["バウカウ", "Baukau"], "Manatuto": ["マナトゥト", "Manatutu"], "Liquiçá": ["リキサ", "Likisá"],
             "Ermera": ["エルメラ"], "Aileu": ["アイレウ"], "Ainaro": ["アイナロ", "Ainaru"], "Bobonaro": ["ボボナロ", "Bobonaru"],
             "Covalima": ["コバリマ", "Kova Lima", "Cova-Lima"], "Manufahi": ["マヌファヒ"], "Lautém": ["ラウテム"],
             "Viqueque": ["ビケケ", "Vikeke"], "Oecussi": ["オエクシ", "Oé-Cusse", "Ocussi", "Oecusse-Ambeno"]}
SUCO_STOP = {"Costa", "Cunha", "Estado", "Home", "Liurai", "Suco Liurai", "Suco-Liurai", "Parlamento", "Santa Cruz", "Vila Verde",
             "Samara", "Matata", "Taroman", "Lontas", "Manapa", "Samoro", "Atara", "Lauana", "Mindelo", "Orana", "Cassa", "Debos",
             "Haupu", "Lauala", "Leber", "Lebos", "Raça", "Guiço", "Taraço", "Açumano", "Camenaça", "Dilor", "Babulu", "Babulo"}

_a2 = gpd.read_file(_COD / "tls_admin2.shp")[["adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name"]]
_a3 = gpd.read_file(_COD / "tls_admin3.shp")[["adm3_name", "adm2_pcode"]]
_p = _a2.set_index("adm2_name").adm2_pcode.to_dict()
_m = _a2.drop_duplicates("adm1_name").set_index("adm1_name").adm1_pcode.to_dict()
assert set(POSTO_BY_NAME) == set(_p), set(POSTO_BY_NAME) ^ set(_p)
assert set(MUN_BY_NAME) == set(_m), set(MUN_BY_NAME) ^ set(_m)
POSTO = {_p[k]: list(v) for k, v in POSTO_BY_NAME.items()}
for k, v in LANDMARK_POSTO.items():
    POSTO[_p[k]] += v
MUN = {_m[k]: list(v) + MUN_EXTRA.get(k, []) for k, v in MUN_BY_NAME.items()}
# suco names (unique, >= 5 characters, not shared with posto/municipality names, not common words)
_taken = {x.lower() for vs in list(POSTO_BY_NAME.values()) + list(MUN_BY_NAME.values()) for x in vs} | {k.lower() for k in _p} | {k.lower() for k in _m}
_cnt = _a3.adm3_name.value_counts()
SUCO = {}
for nm, pc in zip(_a3.adm3_name, _a3.adm2_pcode):
    for part in str(nm).replace("(", "/").replace(")", "").split("/"):
        part = part.strip()
        if len(part) >= 5 and _cnt.get(nm, 0) == 1 and part not in SUCO_STOP and part.lower() not in _taken:
            SUCO.setdefault(pc, []).append(part)
for pc, vs in SUCO.items():
    POSTO[pc] += vs

# words that follow "Same" when used as an English word
_SAME_STOP = r"(?!\s+(?:time|as|year|day|period|level|way|area|sector|approach|type|project|amount|people|group|rate|kind|number))"


from geocoder import Geocoder, default_pattern  # noqa: E402

CUSTOM = {"Same": re.compile(r"(?<![.!?]\s)(?<!^)\bSame\b" + _SAME_STOP),
          "Maubisse": re.compile(r"\bMaubisse\b(?!\s+Declara)", re.IGNORECASE)}   # 'Maubisse Declaration' is a national policy
GEO = Geocoder(unit_variants=POSTO, upper_variants=MUN, unit_parent={pc: pc[:4] for pc in POSTO},
               case_sensitive={"Moro", "Alas", "Laga", "Hera"}, custom=CUSTOM, hq_names={"Dili", "Díli"})
_pat = lambda v: CUSTOM.get(v) or default_pattern(v, v in {"Moro", "Alas", "Laga", "Hera"})   # used by the audit script


def geoparse(text):
    """Return (set of posto pcodes, set of municipality pcodes) mentioned in text (HQ/address mentions removed)."""
    units, uppers, _, _ = GEO.parse(text)
    return units, uppers


def geoparse_detail(text):
    return GEO.parse(text)


NATIONWIDE = re.compile(r"\b(nation[- ]?wide|country[- ]?wide|all (12|twelve|13|thirteen|14|fourteen) (districts|municipalities)|all municipalities|all districts|"
                        r"all sucos|every suco|442 sucos|all (the )?villages|whole country|entire country|throughout (the country|Timor)|"
                        r"PNDS|National (Program|Programme) for (Suco|Village) Development|national (immuni[sz]ation|vaccination) campaign|"
                        r"all (public )?schools|all health (facilities|centres|centers))\b", re.I)
REGIONAL = re.compile(r"\b(Pacific (region|islands)|multi[- ]country|regional (program|programme|project|facility)|Asia[- ]Pacific|"
                      r"South[- ]?East Asia region|ASEAN-wide|Melanesia|Indonesia and Timor|Papua New Guinea|Solomon Islands|Fiji|Vanuatu)\b", re.I)
CENTRAL = re.compile(r"\b(Ministr(y|ies)|Minist[ée]rio|Secretar(y|iat) of State|Office of the Prime Minister|Prime Minister|Parliament|"
                     r"Council of Ministers|President(ial)? (Office|Palace)|Court of Appeal|Prosecutor|Ombudsman|PDHJ|Anti-Corruption Commission|"
                     r"Central Bank|BCTL|Banco Central|Petroleum Fund|Timor Sea|ANPM|National Directorate|Directorate[- ]General|"
                     r"PNTL|Pol[ií]cia Nacional|F-FDTL|defen[cs]e force|police development|public financial management|\bPFM\b|"
                     r"budget support|general budget|policy (reform|development|advice|dialogue)|legislat|civil service commission|"
                     r"capacity (building|development) (of|for|to) (the )?(ministry|government|national)|technical assistance to (the )?(government|ministry)|"
                     r"national (statistics|election|electoral)|\bCNE\b|STAE|Customs|Tax (administration|authority)|procurement|"
                     r"scholarship|Australia Awards|fellowship|study abroad|peacekeeping|UNMIT|UNMISET|UNOTIL|"
                     r"Instituto Nacional|National Institute|national laboratory|national (TB|malaria|HIV) programme|"
                     r"donor coordination|aid management|programme management|program management|operational costs|administrative costs)\b", re.I)
