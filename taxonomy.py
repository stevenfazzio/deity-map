"""Display labels for the enrichment enums, and the tradition categories built from them.

Shared by embed.py (the archetype map erases tradition, so it needs the categories),
category_labels.py (which maps Wikipedia's categories to the same traditions) and
render.py (colormaps and hovercards). taxonomy_audit.py checks TRADITIONS against the
rule that defines it.
"""

import numpy as np
import pandas as pd

KIND_LABELS = {
    "deity": "God or goddess",
    "minor_divinity": "Minor divinity",
    # Stored as deified_person, but shown without "deified", which says a person
    # became divine; some traditions hold instead that a god was born human.
    "deified_person": "Person worshipped as divine",
    "divine_group": "Divine group",
}
TRADITION_LABELS = {
    "greek": "Greek",
    "roman": "Roman & Italic",
    "egyptian": "Egyptian",
    "mesopotamian": "Mesopotamian",
    "levantine": "Levantine",
    "anatolian": "Anatolian",
    "iranian": "Iranian",
    "arabian": "Pre-Islamic Arabian",
    "hindu": "Hindu",
    "buddhist": "Buddhist",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "southeast_asian": "Southeast Asian",
    "norse_germanic": "Norse & Germanic",
    "celtic": "Celtic",
    "slavic": "Slavic",
    "baltic": "Baltic",
    "finnic_uralic": "Finnic & Uralic",
    "other_european": "Other European",
    "turkic_mongolic": "Turkic & Mongolic",
    "mesoamerican": "Mesoamerican",
    "south_american": "South American",
    "north_american": "North American",
    "oceanian": "Oceanian",
    "australian": "Aboriginal Australian",
    "african": "African",
    "african_diaspora": "African diaspora",
    "abrahamic": "Abrahamic",
    "modern": "Modern movements",
    "other": "Other",
}
# A coarser grouping of the traditions, for a colormap with few enough colours to tell
# apart at a glance.
REGION_OF = {
    "greek": "Greco-Roman",
    "roman": "Greco-Roman",
    "egyptian": "Egypt & Near East",
    "mesopotamian": "Egypt & Near East",
    "levantine": "Egypt & Near East",
    "anatolian": "Egypt & Near East",
    "iranian": "Egypt & Near East",
    "arabian": "Egypt & Near East",
    "hindu": "South Asian & Buddhist",
    "buddhist": "South Asian & Buddhist",
    "chinese": "East Asia",
    "japanese": "East Asia",
    "korean": "East Asia",
    "southeast_asian": "Southeast Asia",
    "norse_germanic": "Europe & northern Asia",
    "celtic": "Celtic",
    "slavic": "Europe & northern Asia",
    "baltic": "Europe & northern Asia",
    "finnic_uralic": "Europe & northern Asia",
    "other_european": "Europe & northern Asia",
    "turkic_mongolic": "Europe & northern Asia",
    "mesoamerican": "Americas",
    "south_american": "Americas",
    "north_american": "Americas",
    "oceanian": "Oceania & Australia",
    "australian": "Oceania & Australia",
    "african": "Africa & diaspora",
    "african_diaspora": "Africa & diaspora",
    "abrahamic": "Abrahamic & modern",
    "modern": "Abrahamic & modern",
    "other": "Other",
}
# Free-text answers for tradition "other", normalised: (pattern, name, region). The
# enrichment enum had no slot for these; TRADITIONS decides which get a colour.
OTHER_TRADITIONS = [
    (r"\bjain", "Jain", "South Asian & Buddhist"),
    (r"meitei|sanamahi", "Meitei (Sanamahism)", "South Asian & Buddhist"),
    (r"\bainu\b", "Ainu", "East Asia"),
    (r"proto-indo-european", "Proto-Indo-European", "Other"),
    (r"nuristani|kafiristan", "Nuristani", "South Asian & Buddhist"),
    (r"guanche", "Guanche", "Africa & diaspora"),
    (r"\bbon\b", "Bon", "South Asian & Buddhist"),
    (r"yazidi", "Yazidi", "Egypt & Near East"),
    (r"kirat|limbu", "Kirati", "South Asian & Buddhist"),
    (r"\bsikh", "Sikh", "South Asian & Buddhist"),
    (r"kushite|nubian|meroitic", "Kushite", "Africa & diaspora"),
    (r"\bromani\b|\bromany\b", "Romani", "Europe & northern Asia"),
]
# Every tradition either rater can name, with its region: enrich.py's enum less
# "other", then the names normalised from its free text.
TRADITION_REGIONS = {
    TRADITION_LABELS[k]: REGION_OF[k] for k in TRADITION_LABELS if k != "other"
} | {name: region for _, name, region in OTHER_TRADITIONS}

# The traditions with a colour, and a class in the archetype map's eraser, of their
# own. A tradition is on this list when
#   1. Wikipedia files its gods in a deity category of their own under "Deities by
#      continent" or "Deities by religion" (the rest of a region, "Other European", is
#      exempt: no single category can hold a residual), and
#   2. Wikipedia's categories put at least MIN_TRADITION_SIZE of the map's articles in it.
# The floor sits in a gap in the size distribution, so nearby floors pick the same
# list; taxonomy_audit.py prints the gap, and fails if the list and the rule disagree.
# Every other tradition is "Other" on the colormap and in the eraser, but keeps its
# name on the hovercard and its region on the region colormap.
MIN_TRADITION_SIZE = 15
RESIDUAL_TRADITIONS = {"Other European"}
TRADITIONS = [
    "Greek",
    "Roman & Italic",
    "Egyptian",
    "Mesopotamian",
    "Levantine",
    "Anatolian",
    "Iranian",
    "Pre-Islamic Arabian",
    "Hindu",
    "Buddhist",
    "Jain",
    "Meitei (Sanamahism)",
    "Chinese",
    "Japanese",
    "Ainu",
    "Korean",
    "Southeast Asian",
    "Norse & Germanic",
    "Celtic",
    "Slavic",
    "Baltic",
    "Finnic & Uralic",
    "Other European",
    "Turkic & Mongolic",
    "Mesoamerican",
    "South American",
    "North American",
    "Oceanian",
    "Aboriginal Australian",
    "African",
    "African diaspora",
    "Abrahamic",
    "Modern movements",
]
DOMAIN_LABELS = {
    "sky_weather": "Sky & weather",
    "sun": "Sun",
    "moon_night": "Moon & night",
    "earth_agriculture": "Earth & agriculture",
    "fertility_birth": "Fertility & birth",
    "water_sea": "Water & sea",
    "nature_animals": "Nature & animals",
    "love_beauty": "Love & beauty",
    "war": "War",
    "death_underworld": "Death & underworld",
    "wisdom_arts": "Wisdom & arts",
    "fire_craft": "Fire & craft",
    "home_family": "Home & family",
    "healing": "Healing",
    "wealth_fortune": "Wealth & fortune",
    "justice_order": "Justice & order",
    "creation_supreme": "Creator or supreme",
    "trickery_chaos": "Trickery & chaos",
    "protection_place": "Guardian of a place",
    "fate_time": "Fate & time",
    "travel_messenger": "Travel & messages",
    "unknown": "Not known",
    "other": "Other",
}
GENDER_LABELS = {
    "male": "Male",
    "female": "Female",
    "androgynous_or_variable": "Androgynous or variable",
    "none_or_unspecified": "Unspecified",
    "mixed_group": "Mixed group",
}


def upper_first(text: str) -> str:
    # Not str.capitalize(), which lowercases the rest: "planet Venus" -> "Planet venus".
    text = text.strip()
    return text[:1].upper() + text[1:]


def display(values: pd.Series, others: pd.Series, labels: dict) -> pd.Series:
    """Human label for an enum value, using the free-text escape hatch for `other`."""
    shown = values.map(labels)
    use_other = (values == "other") & others.fillna("").str.len().gt(0)
    shown[use_other] = others[use_other].map(upper_first)
    return shown.fillna("Unknown")


def resolve_traditions(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(hovercard label, colormap category, broad region) for each article's tradition."""
    label = display(df.tradition, df.tradition_other, TRADITION_LABELS)
    region = df.tradition.map(REGION_OF)
    other = df.tradition.eq("other")
    free = df.tradition_other.fillna("").str.lower()
    for pattern, name, where in OTHER_TRADITIONS:
        hit = other & free.str.contains(pattern, regex=True)
        label[hit] = name
        region[hit] = where
    return label, label.map(category_of), region


def category_of(tradition: str) -> str:
    """A tradition's colormap category: itself if it has a colour of its own, else Other."""
    return tradition if tradition in TRADITIONS else "Other"


def tradition_codes(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Integer codes for the tradition colormap's categories: TRADITIONS and Other."""
    cat = pd.Categorical(resolve_traditions(df)[1])
    return cat.codes.astype(np.int64), len(cat.categories)
