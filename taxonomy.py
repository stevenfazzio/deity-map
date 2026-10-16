"""Display labels for the enrichment enums, and the tradition categories built from them.

Shared by embed.py (the archetype map erases tradition, so it needs the categories)
and render.py (colormaps and hovercards).
"""

import numpy as np
import pandas as pd

KIND_LABELS = {
    "deity": "God or goddess",
    "minor_divinity": "Minor divinity",
    "deified_person": "Deified person",
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
# enrichment enum had no slot for these, but several recur often enough (Jain 29,
# Meitei 19, Ainu 11 of 4,551) to deserve their own colour.
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
]
# A normalised free-text tradition this common gets its own colormap category;
# rarer ones are plotted as "Other" but keep their name on the hovercard.
MIN_PROMOTED = 5
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
    counts = label[other].value_counts()
    promoted = set(counts[counts >= MIN_PROMOTED].index)
    category = label.where(~other | label.isin(promoted), "Other")
    return label, category, region


def tradition_codes(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Integer codes for the tradition colormap's categories (38 on the 2026-10 corpus)."""
    cat = pd.Categorical(resolve_traditions(df)[1])
    return cat.codes.astype(np.int64), len(cat.categories)
