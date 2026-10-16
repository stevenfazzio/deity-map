"""Tradition and domain labels from Wikipedia's categories: enrich.py's facets, from editors.

Two hand-written tables map a pattern over a category's name to a label in taxonomy.py's
vocabulary: TRADITION_PATTERNS to its tradition names (TRADITION_REGIONS), DOMAIN_PATTERNS
to enrich.py's domain enum. CATEGORY_OVERRIDES pins categories the patterns get wrong.
Traditions without a colour of their own fold into "Other" (taxonomy.category_of), as
they do for enrich.py's labels.

An article takes every label its own categories match, so syncretic figures can carry
two traditions. Only if none match does it look up the category tree (Oceanids → Greek
sea goddesses); the nearest level that yields a label wins, so a distant ancestor never
outvotes the article's own filing. Traditions look two levels up, domains one: a domain
two levels up is usually an eponym's (Children of Gaia → Gaia → Earth goddesses), which
says nothing about the article's own role.

    uv run category_labels.py

Prints coverage and agreement with enrich.py's labels, and writes two files to audit
the tables against: data/category_map.csv (every category that matched, with its
labels and article count) and data/category_disagreements.csv (articles where the
categories and enrich.py disagree, with the categories that decided it).
"""

import re
from collections import Counter
from functools import cache

import pandas as pd

from categories import CATEGORIES_PARQUET, CATEGORY_PARENTS_PARQUET
from common import DATA
from embed import load_corpus
from taxonomy import DOMAIN_LABELS, TRADITION_REGIONS, category_of, resolve_traditions

CATEGORY_MAP_CSV = DATA / "category_map.csv"
DISAGREEMENTS_CSV = DATA / "category_disagreements.csv"
MAX_LEVELS = {"traditions": 2, "domains": 1}  # categories.py fetches 2
# Biographical categories ("19th-century Japanese women") give a deified person's
# nationality, not the tradition that worships them.
IGNORED = re.compile(r"^\d{1,2}(st|nd|rd|th)-century\b")

DIVINE = r"gods|goddesses|deities|divinities|spirits|mythology|legendary creatures"
INDIAN_REGIONS = (
    r"Tamil\w*|Rajasthan\w*|Gujarat\w*|Kerala|Bengal\w*|Odisha|Odia|Maharashtra\w*|Marathi"
    r"|Karnataka|Kannada|Telugu|Andhra Pradesh|Punjab\w*|Himachal Pradesh|Uttarakhand"
)

# (pattern, label). Matched case-sensitively anywhere in the category name: proper
# nouns keep their capitals in category names, which keeps "Fon" from matching "font".
# Labels are taxonomy.TRADITION_REGIONS' names, including those without a colour of
# their own, so taxonomy_audit.py can count them.
TRADITION_PATTERNS = [
    # Greek
    (r"\bGreek\b(?! Orthodox)|\bGreco-|\bHellenic\b|\bMycenaean\b|\bOlympians?\b", "Greek"),
    # "Hellenistic Egyptian deities" are Egyptian gods worshipped in the Greek world
    (r"\bHellenistic\b(?! (Egyptian|Asian))", "Greek"),
    (r"\bNereids?\b|\bOceanids?\b|\bNaiads?\b|\bOreads?\b|\bDryads?\b|\bHamadryads\b", "Greek"),
    (r"\bMuses\b|\bTitans?\b|\bPotamoi\b|\bCharites\b|\bHorae\b|\bErinyes\b|\bMoirai\b", "Greek"),
    (r"\bIliad\b|\bOdyssey\b|\bHomeric\b|\bHesiod|\bTheogony\b|\bOrphic", "Greek"),
    (r"\b(Zeus|Hera|Poseidon|Apollo|Artemis|Athena|Dionysus|Hermes|Oceanus|Gaia)\b", "Greek"),
    # Roman & Italic
    (
        r"\bRoman\b(?! Catholic| province)|\bEtruscan\b|\bItalic\b|\bSabine\b|\bUmbrian\b|\bOscan\b",
        "Roman & Italic",
    ),
    (r"\bdi indigetes\b|\bCapitoline\b|\bLares\b|\bPenates\b", "Roman & Italic"),
    # Egyptian
    (r"\bEgyptian\b|\bEnnead\b|\bOgdoad\b", "Egyptian"),
    # Mesopotamian
    (
        r"\bMesopotamian?\b|\bSumer(ian)?\b|\bAkkadian\b|\bBabylonian?\b|\bAssyrian?\b",
        "Mesopotamian",
    ),
    (r"\bElamite\b|\bAnunnaki\b|\bInanna\b|\bIshtar\b|\bEnlil\b|\bEnki\b", "Mesopotamian"),
    # Levantine
    (r"\bCanaanite\b|\bUgaritic\b|\bPhoenician\b|\bPunic\b|\bCarthaginian\b", "Levantine"),
    (r"\bArame(an|ans)\b|\bPhilistine\b|\bLevantine\b|\bIsraelite\b|\bWest Semitic\b", "Levantine"),
    (r"\bAmmonite\b|\bMoabite\b|\bEdomite\b|\bBaal\b|\bEl \(deity\)|\bAsherah\b", "Levantine"),
    # Anatolian
    (r"\bHittite\b|\bHurrian\b|\bLuwian\b|\bHattian\b|\bPhrygian\b|\bLydian\b", "Anatolian"),
    (r"\bLycian\b|\bCarian\b|\bUrartian\b|\bAnatolian\b|\bKumarbi\b|\bTeshub\b", "Anatolian"),
    # Iranian
    (r"\bIranian\b|\bPersian\b|\bZoroastrian|\bYazatas?\b|\bAmesha Spentas?\b", "Iranian"),
    # Pre-Islamic Arabian
    (
        r"\bArabian\b|\bPre-Islamic\b|\bpre-Islamic\b|\bNabataean\b|\bSabaean\b",
        "Pre-Islamic Arabian",
    ),
    (r"\bSouth Arabia|\bQataban", "Pre-Islamic Arabian"),
    # Hindu
    (r"\bHindu|\bVedic\b|\bRigved|\bPuran|\bShaiv|\bShakt|\bVaishnav|\bSmarta\b", "Hindu"),
    (r"\b(Shiva|Vishnu|Parvati|Durga|Kali|Lakshmi|Ganesha|Krishna|Rama|Brahma)\b", "Hindu"),
    (r"\bAdityas\b|\bNavagraha\b|\bMatrikas\b|\bMahavidyas?\b|\bAvatars? of\b", "Hindu"),
    (r"\bMahabharata\b|\bRamayana\b", "Hindu"),
    # Indian regions and "Indian" only with a word for gods: "Culture of Bengal",
    # "Indian Ocean" and "Indian religions" (which takes in Sikhism) say nothing.
    (rf"\b({INDIAN_REGIONS})\b.*\b({DIVINE})|\b({DIVINE}) of ({INDIAN_REGIONS})\b", "Hindu"),
    (rf"(?<!American )(?<!West )\bIndian ({DIVINE}|folk)", "Hindu"),
    # Buddhist
    (
        r"\bBuddh|\bBodhisattvas?\b|\bDharmapalas?\b|\bVajrayana\b|\bMahayana\b|\bTheravada\b",
        "Buddhist",
    ),
    (r"\bYidams?\b|\bDakinis?\b|\bArhats?\b|\bTibetan\b", "Buddhist"),
    # Chinese
    (r"\bChinese\b|\bTaois[mt]\b|\bDaois[mt]\b|\bConfucian|\bInvestiture of the Gods\b", "Chinese"),
    (r"\bJourney to the West\b|\bFengshen\b|\bWufang\b", "Chinese"),
    # Japanese
    (r"\bJapanese\b|\bShinto\b|\b[Kk]ami\b|\bAmatsukami\b|\bKunitsukami\b", "Japanese"),
    (r"\bRyukyuan\b|\bOkinawan\b|\bKojiki\b|\bNihon Shoki\b", "Japanese"),
    # Korean
    (r"\bKorean\b", "Korean"),
    # Southeast Asian
    (
        r"\bPhilippine\b|\bFilipino\b|\bTagalog\b|\bVisayan\b|\bBikol\b|\bIfugao\b",
        "Southeast Asian",
    ),
    (
        r"\bIndonesian\b|\bJavanese\b|\bBalinese\b|\bMalay\b|\bThai\b|\bVietnamese\b",
        "Southeast Asian",
    ),
    (
        r"\bKhmer\b|\bCambodian\b|\bBurmese\b|\bMyanmar\b|\b[Nn]ats\b|\bLao\b|\bDayak\b",
        "Southeast Asian",
    ),
    (r"\bBatak\b|\bBugis\b|\bCham\b|\bAhom\b|\bSoutheast Asian\b", "Southeast Asian"),
    # Norse & Germanic
    (r"\bNorse\b|\bGermanic\b|\bÆsir\b|\bAesir\b|\bVanir\b|\bAnglo-Saxon\b", "Norse & Germanic"),
    (r"\bValkyries?\b|\bJötnar\b|\bScandinavian\b|\bFrisian\b|\bOld English\b", "Norse & Germanic"),
    # Celtic
    (r"\bCelt|\bGaulish\b|\bGallic\b|\bBrittonic\b|\bTuatha Dé Danann\b|\bFomorians\b", "Celtic"),
    # Modern nationalities only with a word for gods: not "Anglo-Scottish border"
    (rf"\b(Irish|Welsh|Scottish|Manx|Cornish|Breton) ({DIVINE}|mythology|folklore)", "Celtic"),
    # Slavic, Baltic
    (r"\bSlav|\bRussian\b|\bPolish\b|\bCzech\b|\bSerbian\b|\bCroatian\b", "Slavic"),
    (r"\bBulgarian\b|\bUkrainian\b|\bSlovak\b|\bSlovene\b|\bSorbian\b", "Slavic"),
    (r"\bBaltic\b(?! Sea)|\bLithuanian\b|\bLatvian\b|\bPrussian\b", "Baltic"),
    # Finnic & Uralic
    (
        r"\bFinnish\b|\bFinnic\b|\bEstonian\b|\bS[áa]a?mi\b|\bHungarian\b|\bMagyar\b",
        "Finnic & Uralic",
    ),
    (
        r"\bMari (mythology|paganism|religion|people)\b|\bUdmurt\b|\bKomi\b|\bMordvin|\bErzya\b|\bUralic\b|\bKhanty\b",
        "Finnic & Uralic",
    ),
    (r"\bMansi\b|\bKarelian\b|\bNenets\b|\bSamoyed", "Finnic & Uralic"),
    # Other European
    (r"\bThracian\b|\bDacian\b|\bIllyrian\b|\bAlbanian\b|\bPaleo-Balkan\b", "Other European"),
    (r"\bLusitanian\b|\bPaleo-Hispanic\b|\bBasque\b|\bScythian\b|\bArmenian\b", "Other European"),
    (
        r"\bGeorgian\b|\bCaucasian\b|\bCircassian\b|\bOssetian\b|\bAbkhaz|\bVainakh\b",
        "Other European",
    ),
    # Turkic & Mongolic
    (
        r"\bTurkic\b|\bTengri|\bMongol|\bSiberian\b|\bTungusic\b|\bEvenk\b|\bBuryat\b",
        "Turkic & Mongolic",
    ),
    (
        r"\bYakut\b|\bSakha\b|\bAltai\b|\bTatar\b|\bChuvash\b|\bKazakh\b|\bKyrgyz\b",
        "Turkic & Mongolic",
    ),
    # Mesoamerican
    (
        r"\bAztec\b|\bMaya\b|\bMayan\b|\bMesoamerican\b|\bNahua\b|\bMixtec\b|\bZapotec\b",
        "Mesoamerican",
    ),
    (r"\bOlmec\b|\bToltec\b|\bTotonac\b|\bHuastec\b|\bPur[ée]pecha\b|\bTarascan\b", "Mesoamerican"),
    # South American
    (
        r"\bInca\b|\bIncan\b|\bAndean\b|\bQuechua\b|\bAymara\b|\bMapuche\b|\bGuaran[íi]\b",
        "South American",
    ),
    (r"\bAmazon(ian)?\b|\bTupi\b|\bTa[íi]no\b|\bMuisca\b|\bChibcha\b|\bMoche\b", "South American"),
    (r"\bSouth American\b|\bArawak|\bCarib\b|\bSelk'nam\b|\bShuar\b", "South American"),
    # North American
    (
        r"\bNative American\b|\bAmerican Indian\b|\bFirst Nations\b|\bNorth American\b",
        "North American",
    ),
    (
        r"\bIndigenous peoples of (the )?(North America|Canada|the United States)\b",
        "North American",
    ),
    (
        r"\bInuit\b|\bAleut\b|\bHaida\b|\bTlingit\b|\bLakota\b|\bSioux\b|\bNavajo\b|\bHopi\b",
        "North American",
    ),
    (
        r"\bZuni\b|\bPueblo\b|\bCherokee\b|\bIroquois\b|\bHaudenosaunee\b|\bOjibwe\b",
        "North American",
    ),
    (
        r"\bAnishinaabe\b|\bAlgonquian\b|\bCree\b|\bKwakwaka'wakw\b|\bSalish\b|\bPawnee\b",
        "North American",
    ),
    (
        r"\bApache\b|\bChumash\b|\bOhlone\b|\bMiwok\b|\bMuscogee\b|\bKachinas?\b|\bKatsinam?\b",
        "North American",
    ),
    # Oceanian
    (r"\bPolynesian\b|\bHawaiian\b|\bM[āa]ori\b|\bTahitian\b|\bSamoan\b|\bTongan\b", "Oceanian"),
    (r"\bFijian\b|\bMarquesan\b|\bRapa Nui\b|\bEaster Island\b|\bCook Islands?\b", "Oceanian"),
    (r"\bNiue(an)?\b|\bTuvalu(an)?\b|\bMelanesian\b|\bMicronesian\b|\bPapua", "Oceanian"),
    (r"\bSolomon Islands\b|\bVanuatu\b|\bNew Caledonia\b|\bKanak\b|\bChamorro\b", "Oceanian"),
    (r"\bPalauan\b|\bKiribati\b|\bMangaia|\bTuamotu\b|\bTupua\b|\bOceania\b", "Oceanian"),
    # Aboriginal Australian
    (r"\bAustralian\b|\bAboriginal\b|\bDreamtime\b|\bDreaming\b", "Aboriginal Australian"),
    # African
    (r"\bAfrican\b(?! diaspora)|\bYoruba\b|\bIgbo\b|\bOdinani\b|\bAkan\b|\bAshanti\b", "African"),
    (r"\bFon\b|\bDahomey\b|\bEwe\b|\bZulu\b|\bXhosa\b|\bBantu\b|\bDinka\b|\bNuer\b", "African"),
    (r"\bShona\b|\bKikuyu\b|\bMaasai\b|\bBerber\b|\bAmazigh\b|\bDogon\b|\bBambara\b", "African"),
    (r"\bSerer\b|\bEfik\b|\bIbibio\b|\bEdo\b|\bLugbara\b|\bBaganda\b|\bGanda\b", "African"),
    (r"\bEthiopian\b|\bMalagasy\b|\bKhoikhoi\b|\bKhoisan\b|\bHausa\b|\bTswana\b", "African"),
    (r"\bSotho\b|\bMbuti\b|\bVodun\b|\bOrishas?\b|\bNigerian\b|\bGhanaian\b|\bKenyan\b", "African"),
    (r"\bUgandan\b|\bCongolese\b|\bCameroonian\b|\bTanzanian\b|\bKongo\b|\bLuba\b", "African"),
    # African diaspora
    (
        r"\bVodou\b|\bVoodoo\b|\bHaitian\b|\bSanter[íi]a\b|\bCandombl[ée]\b|\bUmbanda\b",
        "African diaspora",
    ),
    (
        r"\bQuimbanda\b|\bLucum[íi]\b|\bL[wo]a\b|\bPalo\b|\bObeah\b|\bHoodoo\b|\bWinti\b",
        "African diaspora",
    ),
    (r"\bAfro-(American|Brazilian|Cuban|Caribbean)\b|\bAfrican diaspora\b", "African diaspora"),
    # Abrahamic
    (r"\bChristian|\bCatholic\b|\bIslam|\bMuslim\b|\bJewish\b|\bJudaism\b|\bKabbal", "Abrahamic"),
    (r"\bGnostic|\bMandae|\bManichae|\bTrinity\b|\bAngels?\b|\bArchangels?\b|\bQuran", "Abrahamic"),
    # Modern movements
    (
        r"\bNew Age\b|\bWicca|\bNeopagan|\bModern pagan|\bNew religious movements?\b|\bThelem",
        "Modern movements",
    ),
    (
        r"\bDiscordian|\bTheosoph|\bSatanis|\bRodnover|\bHeathenry\b|\bUFO religion",
        "Modern movements",
    ),
    # traditions promoted from enrich.py's free text (taxonomy.OTHER_TRADITIONS)
    (r"\bJain|\bTirthankaras?\b", "Jain"),
    (r"\bMeitei\b|\bSanamahi|\bManipur|\bUmang Lai\b|\bLai Haraoba\b", "Meitei (Sanamahism)"),
    (r"\bAinu\b", "Ainu"),
    (r"\bProto-Indo-European\b|\bIndo-European\b", "Proto-Indo-European"),
    (r"\bNuristan|\bKafiristan\b|\bKalash\b", "Nuristani"),
    (r"\bGuanche\b|\bCanary Islands\b", "Guanche"),
    (r"\bB[öo]n\b(?! [Ff]estival)", "Bon"),
    (r"\bYazidi|\bYezidi", "Yazidi"),
    (r"\bSikh", "Sikh"),
    (r"\bKushite\b|\bNubian\b|\bMeroitic\b", "Kushite"),
    (r"\bKira(n)?t|\bLimbu\b", "Kirati"),
    (r"\bRomani\b", "Romani"),
]


def by_function(words: str) -> str:
    """Wikipedia's "Deities by function" names: "<words> gods / goddesses / deities"."""
    return rf"(?i:\b(?:{words})) (gods|goddesses|deities)$"


# (pattern, enrich.py domain). Only the "Deities by function" form and a few named groups:
# looser patterns catch "Solar dynasty", "Poetry by Ovid" and "Death of deities".
DOMAIN_PATTERNS = [
    (by_function(r"sky and weather|(?<!night )sky|weather|thunder|rain|wind|storm"), "sky_weather"),
    (by_function(r"solar|sun|dawn"), "sun"),
    (by_function(r"lunar|moon|night|night sky|stellar|star"), "moon_night"),
    (by_function(r"earth|agricultural|harvest|vegetation"), "earth_agriculture"),
    (by_function(r"fertility|mother|childbirth"), "fertility_birth"),
    (by_function(r"sea and river|sea|river|water|spring|lake"), "water_sea"),
    (
        r"^(Oceanids|Naiads|Nereids|Potamoi|Water spirits|River gods in Greek mythology)$",
        "water_sea",
    ),
    (by_function(r"nature|animal|hunting|mountain|tree|forest"), "nature_animals"),
    (r"^(Dryads|Oreads|Forest spirits)$", "nature_animals"),
    (by_function(r"love and lust|love|beauty"), "love_beauty"),
    (by_function(r"war|victory"), "war"),
    (by_function(r"death|underworld"), "death_underworld"),
    (r"^(Psychopomps|Chthonic beings)$", "death_underworld"),
    (
        by_function(r"wisdom|knowledge|arts|music and singing|literature|writing|poetry"),
        "wisdom_arts",
    ),
    (r"^Muses( \(mythology\))?$", "wisdom_arts"),
    (by_function(r"fire|smithing|volcano|crafts?"), "fire_craft"),
    (by_function(r"household|domestic and hearth|marriage|childhood"), "home_family"),
    (by_function(r"health|medicine|plague|smallpox|disease"), "healing"),
    (by_function(r"fortune|abundance|commerce|wealth|luck"), "wealth_fortune"),
    (by_function(r"justice|oath|law"), "justice_order"),
    (r"^Oaths$", "justice_order"),
    (by_function(r"creator"), "creation_supreme"),
    (r"^Kings of the gods$", "creation_supreme"),
    (by_function(r"trickster|chaos|destroyer"), "trickery_chaos"),
    (r"^Tricksters$", "trickery_chaos"),
    (by_function(r"tutelary|city|national"), "protection_place"),
    (by_function(r"time and fate|time|fate"), "fate_time"),
    (by_function(r"liminal|messenger|travel|crossroads"), "travel_messenger"),
    (r"^Crossroads mythology$", "travel_messenger"),
]

# Categories the patterns would label wrongly, pinned by hand: {category: (traditions,
# domains)}, None leaving that side to the patterns and () giving no label.
CATEGORY_OVERRIDES = {
    # who mentions a god, not whose god it is: Marduk and Tammuz are in it too
    "Deities in the Hebrew Bible": ((), None),
    # where a god was worshipped, not whose: 17 of its 22 are Egyptian (Isis, Ra, Hathor)
    "Nubian gods": ((), None),
    "Nubian goddesses": ((), None),
    # the orishas and lwa their followers identify with saints
    "Supernatural beings identified with Christian saints": ((), None),
}


def _compile(table):
    return [(re.compile(p), label) for p, label in table]


_TRADITIONS, _DOMAINS = _compile(TRADITION_PATTERNS), _compile(DOMAIN_PATTERNS)


@cache
def traditions_of(category: str) -> frozenset:
    if IGNORED.search(category):
        return frozenset()
    pinned = CATEGORY_OVERRIDES.get(category, (None, None))[0]
    if pinned is not None:
        return frozenset(pinned)
    return frozenset(label for rx, label in _TRADITIONS if rx.search(category))


@cache
def domains_of(category: str) -> frozenset:
    if IGNORED.search(category):
        return frozenset()
    pinned = CATEGORY_OVERRIDES.get(category, (None, None))[1]
    if pinned is not None:
        return frozenset(pinned)
    return frozenset(label for rx, label in _DOMAINS if rx.search(category))


def resolve(
    article_cats: dict, parents: dict, labels_of, max_level: int
) -> tuple[dict, dict, dict]:
    """Per pageid: (labels, the level they came from, the categories that gave them)."""
    labels, level_of, source = {}, {}, {}
    for pageid, cats in article_cats.items():
        frontier = set(cats)
        for level in range(max_level + 1):
            hits = {c: labels_of(c) for c in frontier if labels_of(c)}
            if hits:
                labels[pageid] = sorted(set().union(*hits.values()))
                level_of[pageid], source[pageid] = level, sorted(hits)
                break
            frontier = {p for c in frontier for p in parents.get(c, ())}
        else:
            labels[pageid], level_of[pageid], source[pageid] = [], None, []
    return labels, level_of, source


def structured_labels(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per corpus article: category-derived traditions and domains, as lists.

    `tradition_names` are the traditions as named; `traditions` are their colormap
    categories, with those lacking a colour of their own folded into Other.
    """
    cats = pd.read_parquet(CATEGORIES_PARQUET)
    tree = pd.read_parquet(CATEGORY_PARENTS_PARQUET)
    missing = set(corpus.pageid) - set(cats.pageid)
    # Also fires for an article with no visible categories at all; none had any on 2026-10-16.
    assert not missing, f"{len(missing)} articles have no fetched categories: run categories.py"
    article_cats = cats.groupby("pageid").category.apply(list).to_dict()
    article_cats = {p: article_cats.get(p, []) for p in corpus.pageid}
    parents = tree.groupby("category").parent.apply(list).to_dict()

    out = pd.DataFrame({"pageid": corpus.pageid})
    for name, labels_of in [("traditions", traditions_of), ("domains", domains_of)]:
        labels, level, source = resolve(article_cats, parents, labels_of, MAX_LEVELS[name])
        out[name] = out.pageid.map(labels)
        out[f"{name}_level"] = out.pageid.map(level).astype("Int64")
        out[f"{name}_from"] = out.pageid.map(source)
    out = out.rename(columns={"traditions": "tradition_names"})
    out.insert(
        1, "traditions", out.tradition_names.map(lambda ts: sorted({category_of(t) for t in ts}))
    )
    return out


def colormap_traditions(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(hovercard label, colormap category, broad region) per article, categories first.

    taxonomy.resolve_traditions' counterpart, with Wikipedia's categories deciding the
    colour. An article in one tradition takes it. One in several takes enrich.py's choice
    among them or, when enrich.py chose none of them, the one most of its categories name.
    One whose categories name no tradition takes enrich.py's. The hovercard keeps
    enrich.py's label wherever the two agree on the category, since its free text is
    more specific ("Nuragic Sardinian" where the categories only say Other).
    """
    llm_label, llm_category, llm_region = resolve_traditions(df)
    s = structured_labels(df)
    size = pd.Series([c for cats in s.traditions for c in cats]).value_counts()
    label, category, how = [], [], Counter()
    for cats, names, sources, llm_cat, llm_lab in zip(
        s.traditions, s.tradition_names, s.traditions_from, llm_category, llm_label
    ):
        if not cats:
            chosen, rule = llm_cat, "no category tradition: enrich.py's"
        elif len(cats) == 1:
            chosen, rule = cats[0], "one tradition"
        elif llm_cat in cats:
            chosen, rule = llm_cat, "several: enrich.py's choice among them"
        else:
            votes = Counter(category_of(t) for c in sources for t in traditions_of(c))
            chosen = max(cats, key=lambda c: (votes[c], size[c]))
            rule = "several, none enrich.py's: most of its categories"
        how[rule] += 1
        category.append(chosen)
        if chosen == llm_cat:
            label.append(llm_lab)
        elif chosen != "Other":
            label.append(chosen)
        else:  # the tradition that folded into Other: "Proto-Indo-European"
            label.append(min(n for n in names if category_of(n) == "Other"))
    print("tradition colours: " + "; ".join(f"{rule} {n}" for rule, n in how.items()))
    label = pd.Series(label, index=df.index)
    region = label.map(TRADITION_REGIONS).fillna(llm_region)
    return label, pd.Series(category, index=df.index), region


def check_vocabulary() -> None:
    unknown = {label for _, label in TRADITION_PATTERNS} - set(TRADITION_REGIONS)
    assert not unknown, f"TRADITION_PATTERNS uses names taxonomy.py doesn't have: {unknown}"
    unknown = {label for _, label in DOMAIN_PATTERNS} - set(DOMAIN_LABELS)
    assert not unknown, f"DOMAIN_PATTERNS uses labels enrich.py doesn't have: {unknown}"


def report(corpus: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    df = (
        corpus[["pageid", "name"]]
        .assign(
            llm_tradition=resolve_traditions(corpus)[1].to_numpy(),
            llm_domain=corpus.primary_domain.to_numpy(),
            llm_domains=[
                [p, *(sec if sec is not None else [])]
                for p, sec in zip(corpus.primary_domain, corpus.secondary_domains)
            ],
        )
        .merge(s, on="pageid", validate="1:1")
    )
    n = len(df)

    print(f"traditions from categories, {n} articles:")
    for level, k in df.traditions_level.value_counts(dropna=False).sort_index().items():
        where = "no label" if pd.isna(level) else f"level {level}"
        print(f"  {where:9s} {k:5d} ({k / n:.1%})")
    has = df[df.traditions.str.len() > 0]
    multi = has.traditions.str.len() > 1
    print(f"  {multi.sum()} labelled articles carry more than one tradition")
    agree = [llm in t for llm, t in zip(has.llm_tradition, has.traditions)]
    single = has[~multi]
    print(
        f"  enrich.py's tradition among them: {sum(agree) / len(has):.3f} of {len(has)}; "
        f"exact on single-label articles: {(single.traditions.str[0] == single.llm_tradition).mean():.3f}"
    )

    print(f"domains from categories, {n} articles:")
    for level, k in df.domains_level.value_counts(dropna=False).sort_index().items():
        where = "no label" if pd.isna(level) else f"level {level}"
        print(f"  {where:9s} {k:5d} ({k / n:.1%})")
    has_d = df[(df.domains.str.len() > 0) & ~df.llm_domain.isin(["unknown", "other"])]
    prim = [p in d for p, d in zip(has_d.llm_domain, has_d.domains)]
    anyd = [bool(set(a) & set(d)) for a, d in zip(has_d.llm_domains, has_d.domains)]
    print(
        f"  of {len(has_d)} with a category domain and a known enrich.py domain: "
        f"primary among them {sum(prim) / len(has_d):.3f}, any of enrich.py's {sum(anyd) / len(has_d):.3f}"
    )
    return df.assign(tradition_agrees=pd.Series(agree, index=has.index)).fillna(
        {"tradition_agrees": True}
    )


def write_reviews(corpus: pd.DataFrame, df: pd.DataFrame) -> None:
    cats = pd.read_parquet(CATEGORIES_PARQUET)
    used = (
        pd.concat(
            [
                df[["pageid", "traditions_from"]]
                .explode("traditions_from")
                .rename(columns={"traditions_from": "category"}),
                df[["pageid", "domains_from"]]
                .explode("domains_from")
                .rename(columns={"domains_from": "category"}),
                cats,
            ]
        )
        .dropna()
        .drop_duplicates()
    )
    counts = used.groupby("category").pageid.nunique()
    rows = [
        {
            "category": c,
            "articles": n,
            "traditions": "; ".join(sorted(traditions_of(c))),
            "domains": "; ".join(sorted(domains_of(c))),
            "overridden": c in CATEGORY_OVERRIDES,
        }
        for c, n in counts.items()
    ]
    cmap = pd.DataFrame(rows)
    cmap = cmap[(cmap.traditions != "") | (cmap.domains != "") | cmap.overridden]
    cmap.sort_values("articles", ascending=False).to_csv(CATEGORY_MAP_CSV, index=False)
    print(f"wrote {CATEGORY_MAP_CSV} ({len(cmap)} categories with a label)")

    dis = df[~df.tradition_agrees.astype(bool)]
    dis = dis.assign(
        traditions=dis.traditions.str.join("; "), traditions_from=dis.traditions_from.str.join("; ")
    )[["name", "llm_tradition", "traditions", "traditions_level", "traditions_from"]]
    dis.sort_values(["llm_tradition", "name"]).to_csv(DISAGREEMENTS_CSV, index=False)
    print(f"wrote {DISAGREEMENTS_CSV} ({len(dis)} articles)")


def main():
    corpus = load_corpus()
    check_vocabulary()
    s = structured_labels(corpus)
    df = report(corpus, s)
    write_reviews(corpus, df)


if __name__ == "__main__":
    main()
