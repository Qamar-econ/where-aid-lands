"""Nepal gazetteer config for geocoder.Geocoder: unit = 75 districts (2011 boundaries), upper = provinces (2015-),
zones (14) and development regions (5) used in pre-federal documents (each mapped to its districts)."""
import re
from pathlib import Path
import pandas as pd
from geocoder import Geocoder

ROOT = Path(__file__).resolve().parents[1]
D = pd.read_csv(ROOT / "data" / "processed" / "nepal" / "districts.csv")

VAR = {  # district key -> extra spellings / main towns
    "kathmandu": ["Kathmandu"], "lalitpur": ["Lalitpur", "Patan"], "bhaktapur": ["Bhaktapur"], "kavre": ["Kavre", "Kavrepalanchok", "Kavrepalanchowk", "Kabhre", "Kabhrepalanchok", "Dhulikhel", "Banepa"],
    "sindhupalchok": ["Sindhupalchok", "Sindhupalchowk", "Chautara"], "makwanpur": ["Makwanpur", "Makawanpur", "Hetauda"], "chitwan": ["Chitwan", "Chitawan", "Bharatpur"],
    "dhanusa": ["Dhanusa", "Dhanusha", "Janakpur"], "kapilvastu": ["Kapilvastu", "Kapilbastu"], "tanahun": ["Tanahun", "Tanahu", "Damauli"],
    "tehrathum": ["Tehrathum", "Terhathum"], "udaypur": ["Udayapur", "Udaypur", "Gaighat"], "nawalparasi": ["Nawalparasi", "Nawalpur", "Parasi"],
    "rukum": ["Rukum"], "solukhumbu": ["Solukhumbu", "Solu-Khumbu", "Salleri"], "sankhuwasabha": ["Sankhuwasabha", "Khandbari"],
    "kaski": ["Kaski", "Pokhara"], "rupandehi": ["Rupandehi", "Butwal", "Bhairahawa", "Siddharthanagar"], "banke": ["Banke", "Nepalgunj", "Nepalganj"],
    "kailali": ["Kailali", "Dhangadhi"], "morang": ["Morang", "Biratnagar"], "sunsari": ["Sunsari", "Dharan", "Itahari", "Inaruwa"], "parsa": ["Parsa", "Birgunj"],
    "surkhet": ["Surkhet", "Birendranagar"], "dang": ["Dang", "Ghorahi", "Tulsipur"], "kanchanpur": ["Kanchanpur", "Mahendranagar", "Bhimdatta"],
    "syangja": ["Syangja", "Syanja"], "sindhuli": ["Sindhuli"], "okhaldhunga": ["Okhaldhunga"], "gorkha": ["Gorkha", "Barpak"], "dolakha": ["Dolakha", "Charikot"],
    "rasuwa": ["Rasuwa", "Dhunche"], "nuwakot": ["Nuwakot", "Bidur"], "dhading": ["Dhading"], "ramechhap": ["Ramechhap"], "bardiya": ["Bardiya", "Bardia", "Gulariya"],
    "jhapa": ["Jhapa", "Birtamod", "Damak", "Bhadrapur"], "ilam": ["Ilam"], "dailekh": ["Dailekh"], "darchula": ["Darchula"], "achham": ["Achham", "Accham"],
}
SHORT_CS = {"Dang", "Bara", "Parsa", "Doti", "Ilam", "Mugu", "Banke", "Jumla", "Humla", "Dolpa", "Palpa", "Patan", "Parasi", "Salyan"}
unit_variants = {r.k: sorted(set([r.district] + VAR.get(r.k, []))) for r in D.itertuples()}

PROV = {"Koshi": ["Koshi Province", "Province No. 1", "Province 1", "Province One"], "Madhesh": ["Madhesh Province", "Madhesh Pradesh", "Province No. 2", "Province 2"],
        "Bagmati": ["Bagmati Province", "Province No. 3", "Province 3"], "Gandaki": ["Gandaki Province", "Province No. 4", "Province 4"],
        "Lumbini": ["Lumbini Province", "Province No. 5", "Province 5"], "Karnali": ["Karnali Province", "Province No. 6", "Province 6"],
        "Sudur Paschim": ["Sudurpashchim", "Sudurpaschim", "Sudur Paschim", "Far[- ]?West(ern)? Province", "Province No. 7", "Province 7"]}
ZONES = {"Mechi": ["taplejung", "panchthar", "ilam", "jhapa"], "Koshi Zone": ["morang", "sunsari", "dhankuta", "tehrathum", "sankhuwasabha", "bhojpur"],
         "Sagarmatha": ["solukhumbu", "okhaldhunga", "khotang", "udaypur", "saptari", "siraha"], "Janakpur": ["dhanusa", "mahottari", "sarlahi", "sindhuli", "ramechhap", "dolakha"],
         "Bagmati Zone": ["kathmandu", "lalitpur", "bhaktapur", "kavre", "sindhupalchok", "nuwakot", "rasuwa", "dhading"],
         "Narayani": ["makwanpur", "rautahat", "bara", "parsa", "chitwan"], "Gandaki Zone": ["gorkha", "lamjung", "tanahun", "syangja", "kaski", "manang"],
         "Dhaulagiri": ["mustang", "myagdi", "parbat", "baglung"], "Lumbini Zone": ["gulmi", "palpa", "nawalparasi", "rupandehi", "kapilvastu", "arghakhanchi"],
         "Rapti": ["pyuthan", "rolpa", "rukum", "salyan", "dang"], "Bheri": ["dailekh", "jajarkot", "surkhet", "banke", "bardiya"],
         "Karnali Zone": ["dolpa", "jumla", "kalikot", "mugu", "humla"], "Seti": ["bajura", "bajhang", "achham", "doti", "kailali"],
         "Mahakali": ["darchula", "baitadi", "dadeldhura", "kanchanpur"]}
REGIONS = {"Eastern Development Region": ["Mechi", "Koshi Zone", "Sagarmatha"], "Central Development Region": ["Janakpur", "Bagmati Zone", "Narayani"],
           "Western Development Region": ["Gandaki Zone", "Dhaulagiri", "Lumbini Zone"], "Mid-Western Development Region": ["Rapti", "Bheri", "Karnali Zone"],
           "Far-Western Development Region": ["Seti", "Mahakali"]}
UPPER_DISTRICTS = {}
for p, _ in PROV.items():
    UPPER_DISTRICTS["P:" + p] = list(D.loc[D.province == p, "k"])
for z, ds in ZONES.items():
    UPPER_DISTRICTS["Z:" + z] = ds
for r, zs in REGIONS.items():
    UPPER_DISTRICTS["R:" + r] = [d for z in zs for d in ZONES[z]]
upper_variants = {"P:" + p: v for p, v in PROV.items()}
upper_variants.update({"Z:" + z: [z if " " in z else z + " Zone", z + " zone"] for z in ZONES})
upper_variants.update({"R:" + r: [r, r.replace(" Development Region", " Region"), r.replace("-", " ")] for r in REGIONS})
custom = {v: re.compile(v.replace("[- ]?", "[- ]?"), re.I) for v in ["Far[- ]?West(ern)? Province"]}
# capital-valley districts are listed so that mentions resolve, but 'Kathmandu' is the classic HQ string
GEO = Geocoder(unit_variants=unit_variants, upper_variants=upper_variants,
               unit_parent={k: "P:" + p for k, p in zip(D.k, D.province)}, case_sensitive=SHORT_CS, custom=custom,
               hq_names={"Kathmandu"})


def geoparse(text):
    """-> (set district keys, set upper codes)"""
    u, up, _, _ = GEO.parse(text)
    # a zone/region mention is dropped if one of its districts is named (R1 generalised)
    up = {c for c in up if not (set(UPPER_DISTRICTS[c]) & u)}
    return u, up


NATIONWIDE = re.compile(r"\b(nation[- ]?wide|country[- ]?wide|all (75|seventy[- ]five|77) districts|all districts|all (753 )?(local levels|municipalities|palikas)|"
                        r"whole country|entire country|throughout (the country|Nepal)|national (immuni[sz]ation|vaccination) campaign|all (public )?schools)\b", re.I)
REGIONAL = re.compile(r"\b(South Asia region|multi[- ]country|regional (program|programme|project|facility)|Asia[- ]Pacific|SAARC|BBIN|"
                      r"Bangladesh|Bhutan|Sri Lanka|Pakistan|Afghanistan|Maldives|Myanmar|Hindu Kush)\b", re.I)
CENTRAL = re.compile(r"\b(Ministr(y|ies)|Government of Nepal|National Planning Commission|Nepal Rastra Bank|central bank|Parliament|Constituent Assembly|"
                     r"Election Commission|Supreme Court|Attorney General|Office of the Prime Minister|Public Service Commission|Auditor General|"
                     r"Commission for the Investigation of Abuse of Authority|CIAA|Nepal Police|Armed Police|Nepal Army|budget support|"
                     r"public financial management|\bPFM\b|policy (reform|development|advice|dialogue)|legislat|Department of [A-Z]|"
                     r"Central Bureau of Statistics|National Statistics Office|scholarship|fellowship|study abroad|Chevening|"
                     r"national (TB|malaria|HIV) programme|sector[- ]wide approach|SWAp|pooled fund|donor coordination|aid management|"
                     r"programme management|program management|administrative costs|operational costs)\b", re.I)
