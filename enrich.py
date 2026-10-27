"""Classify and describe every candidate article, via the Claude Message Batches API.

One call per article returns:
  - kind: what the article is about. This is the map's scope filter: the four
    in-scope kinds (IN_SCOPE) are plotted, everything else is dropped by embed.py.
  - tradition, primary_domain, secondary_domains, gender: the colormap facets.
  - summary: one line in the model's own words, for the hovercard and search.

Categoricals are schema enums so the colormaps stay clean; tradition and domain each
have an `_other` free-text escape hatch for the hovercard and search.

    uv run enrich.py --sample 60 --direct  # validation: probes + stratified sample
    uv run enrich.py                       # everything not yet enriched, as a batch

    # the same sample through another model, written to its own parquet for comparison
    uv run enrich.py --sample 60 --direct --model claude-haiku-5-5

    # the map's articles again under the current prompt, written to their own parquet
    uv run enrich.py --rerun

The re-run exists for the person-worshipped-as-divine rule (2026-10-26), which came
after the first run: category_labels.decided_kinds takes only its kind, so the other
facets and the summaries stay as first labelled.

Resumable: the batch id is checkpointed to disk right after submission, and a re-run
polls that batch instead of submitting a new one. Refusals are recorded as terminal;
errored or expired requests are left out so the next run retries them.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from common import ARTICLES_PARQUET, DATA, write_bytes_atomic, write_parquet_safely

# Chosen over Opus 5.5 on a 59-article sample (2026-10-15): it agreed with Opus on 57
# of 59 scope calls and every tradition, at half the price (~$22 vs ~$45 for the
# full pool). Haiku 5.5 was cheaper still but broke the prompt's explicit rules
# (angels as divine, domains guessed from etymology).
MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
ENRICHMENT_PARQUET = DATA / "enrichment.parquet"
PENDING_FILE = DATA / "enrich_pending.json"
RERUN_PARQUET = DATA / "enrichment_rerun.parquet"
RERUN_PENDING_FILE = DATA / "enrich_rerun_pending.json"
DIRECT_MAX, DIRECT_WORKERS = 500, 8
# Batch API rates per MTok (50% of list): input, output, cache read.
BATCH_RATES = {
    "claude-opus-5-5": (2.00, 10.00, 0.10),
    "claude-sonnet-5-5": (1.00, 5.00, 0.10),
    "claude-haiku-5-5": (0.05, 0.25, 0.005),
}
# Server-side refusal fallback exists for these; Haiku 5.5 has none.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}


def enrichment_path(model: str):
    """The map reads MODEL's file; any other model's results go beside it, for comparison."""
    return ENRICHMENT_PARQUET if model == MODEL else DATA / f"enrichment_{model}.parquet"


# Boundary cases checked by hand in every validation run: the famous deities that
# Wikidata's deity tree misses, plus out-of-scope cases the prompt has to reject.
PROBES = [
    "Papa Legba",
    "Guan Yu",
    "Coyote (mythology)",
    "Sun Wukong",
    "Allah",
    "Jesus",
    "Jesus in Christianity",
    "Swaminarayan",
    "Holy Spirit",
    "Cthulhu",
    "Waheguru",
    "Nanabozho",
    "Attis",
    "Muses",
    "Sun deity",
    "Achilles",
    "Hermes (DC Comics)",
]


class Kind(StrEnum):
    deity = "deity"
    minor_divinity = "minor_divinity"
    deified_person = "deified_person"
    divine_group = "divine_group"
    deity_concept = "deity_concept"
    non_divine_figure = "non_divine_figure"
    fictional = "fictional"
    not_a_being = "not_a_being"


IN_SCOPE = {Kind.deity, Kind.minor_divinity, Kind.deified_person, Kind.divine_group}


class Tradition(StrEnum):
    greek = "greek"
    roman = "roman"
    egyptian = "egyptian"
    mesopotamian = "mesopotamian"
    levantine = "levantine"
    anatolian = "anatolian"
    iranian = "iranian"
    arabian = "arabian"
    hindu = "hindu"
    buddhist = "buddhist"
    chinese = "chinese"
    japanese = "japanese"
    korean = "korean"
    southeast_asian = "southeast_asian"
    norse_germanic = "norse_germanic"
    celtic = "celtic"
    slavic = "slavic"
    baltic = "baltic"
    finnic_uralic = "finnic_uralic"
    other_european = "other_european"
    turkic_mongolic = "turkic_mongolic"
    mesoamerican = "mesoamerican"
    south_american = "south_american"
    north_american = "north_american"
    oceanian = "oceanian"
    australian = "australian"
    african = "african"
    african_diaspora = "african_diaspora"
    abrahamic = "abrahamic"
    modern = "modern"
    other = "other"


class Domain(StrEnum):
    sky_weather = "sky_weather"
    sun = "sun"
    moon_night = "moon_night"
    earth_agriculture = "earth_agriculture"
    fertility_birth = "fertility_birth"
    water_sea = "water_sea"
    nature_animals = "nature_animals"
    love_beauty = "love_beauty"
    war = "war"
    death_underworld = "death_underworld"
    wisdom_arts = "wisdom_arts"
    fire_craft = "fire_craft"
    home_family = "home_family"
    healing = "healing"
    wealth_fortune = "wealth_fortune"
    justice_order = "justice_order"
    creation_supreme = "creation_supreme"
    trickery_chaos = "trickery_chaos"
    protection_place = "protection_place"
    fate_time = "fate_time"
    travel_messenger = "travel_messenger"
    unknown = "unknown"
    other = "other"


class Gender(StrEnum):
    male = "male"
    female = "female"
    androgynous_or_variable = "androgynous_or_variable"
    none_or_unspecified = "none_or_unspecified"
    mixed_group = "mixed_group"


class Enrichment(BaseModel):
    kind: Kind
    tradition: Tradition
    tradition_other: str
    primary_domain: Domain
    primary_domain_other: str
    secondary_domains: list[Domain]
    gender: Gender
    summary: str


def enum_prop(enum: type[StrEnum]) -> dict:
    return {"type": "string", "enum": [e.value for e in enum]}


# Written out rather than taken from Enrichment.model_json_schema(), which routes
# enums through $defs/$ref; a flat schema is simplest for structured outputs.
SCHEMA = {
    "type": "object",
    "properties": {
        "kind": enum_prop(Kind),
        "tradition": enum_prop(Tradition),
        "tradition_other": {"type": "string"},
        "primary_domain": enum_prop(Domain),
        "primary_domain_other": {"type": "string"},
        "secondary_domains": {"type": "array", "items": enum_prop(Domain)},
        "gender": enum_prop(Gender),
        "summary": {"type": "string"},
    },
    "required": list(Enrichment.model_fields),
    "additionalProperties": False,
}

SYSTEM = """\
You are labelling English Wikipedia articles for an interactive map of the world's \
gods and goddesses. The candidate articles were gathered generously from Wikidata and \
from Wikipedia's deity categories, so many are not about deities at all. For each \
article you get its title, Wikipedia's short description, and its lead section. \
Return the fields below.

kind
What the article is mainly about. The map includes anything treated as divine by a \
real religious or mythological tradition, past or present.
- deity: an individual god or goddess. Includes supreme and monotheistic conceptions \
of God (e.g. "God in Christianity", "Allah"), persons of the Trinity, and avatars, \
forms and aspects that have their own article (e.g. a form of Parvati). Spirits that \
are the principal objects of worship in their tradition count here: kami, loa, \
orisha, vodun. Buddhas and bodhisattvas venerated as objects of devotion count here. \
A being whose lead says it is a particular person who lived is a deified_person \
instead (below).
- minor_divinity: a divine or semi-divine being that its tradition ranks below the \
gods: nymphs, demigods, daimones, divine attendants and messengers, local, \
household and nature spirits that receive cult or offerings.
- deified_person: a person who lived, in history or in legend, and is worshipped as \
divine, when the lead says both. Either the article tells the person's life (Guan Yu, \
Imhotep, Antinous, a deified emperor), or it is about a god that its lead identifies \
as a particular person, as its main account rather than one theory among several \
(Tenjin, the deified Sugawara no Michizane). Decide from the lead, not from theology: \
it does not matter whether the tradition says the person became divine or was a god \
born human, so a teacher or ruler whose followers hold them to be God incarnate or an \
avatar belongs here (Swaminarayan, Haile Selassie). One figure can have an article of \
each kind: the article on Jesus as a first-century preacher is deified_person, the \
article on Jesus in Christianity is a deity. A god that its lead presents as a god is \
a deity even if its myths give it a human birth (Krishna and Rama as avatars of \
Vishnu).
- divine_group: a named, specific collective of divinities treated together (e.g. \
the Muses, the Æsir, the Seven Lucky Gods).
- deity_concept: an article about a type, class or idea of deity rather than a \
particular being or named group (e.g. "Sun deity", "Mother goddess", "Dying-and-rising \
deity", an article explaining what kami are in general).
- non_divine_figure: a figure from myth, legend, folklore or religion who is not \
treated as divine: mortal heroes, monsters, legendary creatures, legendary kings, \
saints, prophets, the Virgin Mary, angels and demons that receive no worship, and \
spirits that grimoires and ceremonial magic summon rather than worship. A \
trickster or culture hero belongs here unless the article presents them as a god or \
as worshipped; when it does, use deity or minor_divinity.
- fictional: a deity or divine being created for modern fiction, games, comics, film \
or novels, including modern fictional versions of traditional gods (e.g. Cthulhu, \
Marvel's Thor, a Dungeons & Dragons god, a video game god).
- not_a_being: anything else: temples, shrines, festivals, texts, artworks, images, \
places, rituals, religious movements, people who are not worshipped as divine, films, \
songs.

tradition
The religious or cultural tradition the article primarily presents the being in. \
Choose by tradition, not by modern country: a Buddhist figure is buddhist wherever it \
is worshipped, a Shinto kami is japanese.
greek (including Mycenaean and Hellenistic), roman (including Etruscan and other \
Italic peoples), egyptian, mesopotamian (Sumerian, Akkadian, Babylonian, Assyrian, \
Elamite), levantine (Canaanite, Ugaritic, Phoenician, Israelite, Aramean, \
Philistine), anatolian (Hittite, Hurrian, Luwian, Phrygian, Lydian, Urartian), iranian \
(Zoroastrian and other Iranian), arabian (pre-Islamic Arabia), hindu (including \
Vedic), buddhist, chinese (folk religion, Taoism), japanese (Shinto, Ryukyuan), \
korean, southeast_asian (mainland and maritime: Philippine, Indonesian, Malay, Thai, \
Vietnamese, Khmer, Burmese and others), norse_germanic (Norse, Anglo-Saxon, \
continental Germanic), celtic (Gaulish, Irish, Welsh, British, Celtiberian), slavic, \
baltic, finnic_uralic (Finnish, Estonian, Sámi, Hungarian, Mari and others), \
other_european (Thracian, Dacian, Illyrian, Albanian, Lusitanian and other \
Paleo-Hispanic, Basque, Scythian, Armenian, Georgian and other Caucasian), \
turkic_mongolic (Tengrism and Siberian traditions), mesoamerican (Aztec, Maya and \
others), south_american (Inca and other Andean, Amazonian, Taíno and Caribbean), \
north_american (Indigenous North American, including Inuit), oceanian (Polynesian, \
Melanesian, Micronesian), australian (Aboriginal Australian), african (traditional \
religions of Africa: Yoruba, Akan, Fon, Igbo, Zulu, Dinka and others), \
african_diaspora (Vodou, Santería, Candomblé, Umbanda and others), abrahamic \
(Judaism, Christianity, Islam, Gnosticism and related), modern (new religious \
movements and modern esoteric traditions). Use other and name it in tradition_other \
when none fits. For articles outside the map's scope, still give your best guess.

primary_domain
The being's main sphere: sky_weather (sky, storms, thunder, rain, wind), sun, \
moon_night (moon, night, stars), earth_agriculture (earth, harvest, grain, \
vegetation), fertility_birth (fertility, childbirth, motherhood), water_sea (sea, \
rivers, lakes, springs), nature_animals (wilderness, forests, mountains, hunting, \
animals), love_beauty (love, desire, beauty), war (war, victory, strength), \
death_underworld, wisdom_arts (wisdom, knowledge, writing, poetry, music), fire_craft \
(fire, smithing, craftsmanship), home_family (hearth, household, marriage, children), \
healing (medicine, health, disease), wealth_fortune (wealth, luck, trade, prosperity), \
justice_order (law, oaths, kingship, cosmic order), creation_supreme (creator, \
supreme god, ruler of the gods, primordial being), trickery_chaos (trickery, \
mischief, chaos, destruction), protection_place (guardian or tutelary of a particular \
place, city, people or family), fate_time (fate, destiny, time), travel_messenger \
(roads, crossroads, travellers, messengers, boundaries). Many gods are known only \
from a name in an inscription or a single mention: if the lead gives no real \
evidence of a sphere, use unknown rather than guessing from the name or from \
analogy with other gods. Use other and name it in primary_domain_other when none fits.

secondary_domains
Up to two more spheres from the same list that the lead clearly attests, most \
important first. Empty if there are none. Never repeat primary_domain.

gender
male, female, androgynous_or_variable (both, neither consistently, or changing \
between sources), none_or_unspecified, or mixed_group for a group of mixed gender. \
Follow the tradition's own theology rather than grammatical pronouns: a God whom the \
tradition holds to be beyond gender is none_or_unspecified even if spoken of as "he".

summary
One sentence of 22 words or fewer saying what the being is: their tradition, role \
and what they are known for. The limit matters: cards are scanned one after another \
as the mouse sweeps the map, and a long summary makes its card taller than its \
neighbours. Prefer the one or two most telling facts over a complete list. Write it \
in your own words; do not quote the lead, use no quotation marks, and do not begin \
with the being's name. It appears in a hover card directly under the name. For an \
out-of-scope article, say what it is about instead.

Leave each *_other field as an empty string unless its field is other."""


# 160 of 11,623 summaries came back ending in residue from the JSON around them: "}",
# '"}', "</br>" or a mix ("”}</br>"). The prompt forbids quotation marks, so a quote
# before the brace is residue too.
SUMMARY_RESIDUE = re.compile(r'(?:\s|["”»]?\}|</?[a-z]+>|_placeholder_ignore)+$')


def clean_summary(text: str) -> str:
    return SUMMARY_RESIDUE.sub("", text)


def user_message(row) -> str:
    lead = (row.lead or "").strip() or "(empty)"
    return (
        f"Title: {row.title}\n"
        f"Short description: {row.description or '(none)'}\n\n"
        f"Lead section:\n{lead}"
    )


def custom_id(pageid: int) -> str:
    return f"p{pageid}"


def build_request(row, model: str = MODEL) -> Request:
    return Request(
        custom_id=custom_id(row.pageid),
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=8000,
            # Identical across every request, so it caches.
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=[{"role": "user", "content": user_message(row)}],
        ),
    )


def load_done(path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["pageid"])


def validation_sample(articles: pd.DataFrame, n: int) -> set[int]:
    """The probes plus an even split of the remainder across the three source strata.

    A plain random sample would be ~75% category-only articles, mostly out of scope,
    which says little about the facets of real deities.
    """
    rng = random.Random(0)
    probes = set(articles.loc[articles.title.isin(PROBES), "pageid"])
    strata = {
        "both": articles.in_wikidata & articles.in_category,
        "wikidata only": articles.in_wikidata & ~articles.in_category,
        "category only": ~articles.in_wikidata & articles.in_category,
    }
    per = max(0, n - len(probes)) // len(strata)
    picked = set(probes)
    for mask in strata.values():
        pool = sorted(set(articles.loc[mask, "pageid"]) - picked)
        picked |= set(rng.sample(pool, min(per, len(pool))))
    return picked


def submit(client, rows: list, pending_file) -> dict:
    batch = client.messages.batches.create(requests=[build_request(r) for r in rows])
    # Checkpoint before polling: a re-run resumes this batch rather than paying twice.
    pending = {
        "batch_id": batch.id,
        "pageids": [int(r.pageid) for r in rows],
        "model": MODEL,
        "effort": EFFORT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_bytes_atomic(json.dumps(pending, indent=2).encode(), pending_file)
    print(f"submitted batch {batch.id} with {len(rows)} requests")
    return pending


def wait(client, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(
            f"  {batch.processing_status}: processing={c.processing} "
            f"succeeded={c.succeeded} errored={c.errored} expired={c.expired}",
            flush=True,
        )
        if batch.processing_status == "ended":
            return
        time.sleep(60)


def parse_message(pageid: int, msg, source: str) -> tuple[dict | None, str | None]:
    """One finished Message -> (row to record, None) or (None, reason to retry)."""
    base = {
        "pageid": pageid,
        "model": msg.model,
        "input_tokens": msg.usage.input_tokens,
        "cache_write_tokens": msg.usage.cache_creation_input_tokens or 0,
        "cache_read_tokens": msg.usage.cache_read_input_tokens or 0,
        "output_tokens": msg.usage.output_tokens,
        "batch_id": source,
    }
    # A refusal is HTTP 200 with no usable content, so check before reading it.
    # Terminal: an identical retry is declined identically.
    if msg.stop_reason == "refusal":
        return {**base, "status": "refusal"}, None
    text = next((b.text for b in msg.content if b.type == "text"), None)
    if msg.stop_reason == "max_tokens" or text is None:
        return None, f"stop_reason={msg.stop_reason}"
    try:
        fields = Enrichment.model_validate_json(text).model_dump(mode="json")
    except ValidationError as e:
        return None, f"validation: {e.errors()[0]['msg']}"
    # The prompt caps these at two and forbids repeating the primary; enforce it.
    secondary = [d for d in fields["secondary_domains"] if d != fields["primary_domain"]]
    fields["secondary_domains"] = list(dict.fromkeys(secondary))[:2]
    fields["summary"] = clean_summary(fields["summary"])
    return {**base, "status": "ok", **fields}, None


def collect(client, pending: dict) -> tuple[list[dict], dict]:
    rows, failures = [], {}
    for result in client.messages.batches.results(pending["batch_id"]):
        pageid = int(result.custom_id[1:])
        if result.result.type != "succeeded":
            # errored / expired / canceled: not recorded, so the next run retries it
            failures[pageid] = result.result.type
            continue
        row, failure = parse_message(pageid, result.result.message, pending["batch_id"])
        if row is not None:
            rows.append(row)
        else:
            failures[pageid] = failure
    return rows, failures


def run_direct(client, todo: list, model: str) -> tuple[list[dict], dict]:
    """Concurrent Messages API calls at standard price, for sets too small to batch.

    A batch can sit queued for an hour; a few dozen direct calls take a minute.
    """
    assert len(todo) <= DIRECT_MAX, f"{len(todo)} requests: use the Batches API"

    def call(row):
        params = build_request(row, model)["params"]
        # Server-side fallback reruns a refused request on another model; the
        # Batches API rejects the parameter, so only this path uses it.
        fallback = (
            {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
            if model in FALLBACK_MODELS
            else {}
        )
        try:
            msg = client.beta.messages.create(**params, **fallback)
        except anthropic.APIError as e:
            return int(row.pageid), None, f"{type(e).__name__}: {e}"
        return int(row.pageid), *parse_message(int(row.pageid), msg, "direct")

    rows, failures = [], {}
    with ThreadPoolExecutor(DIRECT_WORKERS) as pool:
        for pageid, row, failure in pool.map(call, todo):
            if row is not None:
                rows.append(row)
            else:
                failures[pageid] = failure
    return rows, failures


def report(df: pd.DataFrame, articles: pd.DataFrame, model: str = MODEL) -> None:
    ok = df[df.status == "ok"].merge(articles[["pageid", "title"]], on="pageid")
    print(f"\n{len(ok)} ok, {(df.status == 'refusal').sum()} refusals")
    print(f"\nkind:\n{ok.kind.value_counts().to_string()}")
    scoped = ok[ok.kind.isin(IN_SCOPE)]
    print(f"\n{len(scoped)} in scope ({len(scoped) / max(len(ok), 1):.0%})")
    for col in ["tradition", "primary_domain", "gender"]:
        print(f"\n{col} (in scope):\n{scoped[col].value_counts().to_string()}")
        if f"{col}_other" in scoped:
            others = scoped.loc[scoped[col] == "other", f"{col}_other"]
            if len(others):
                print(f"  {col}_other: {sorted(others)}")
    words = ok.summary.str.split().str.len()
    print(f"\nsummary words: min={words.min()} median={words.median():.0f} max={words.max()}")
    rate_in, rate_out, rate_cache_read = BATCH_RATES[model]
    cost = (
        df.input_tokens * rate_in
        + df.cache_write_tokens * rate_in * 1.25
        + df.cache_read_tokens * rate_cache_read
        + df.output_tokens * rate_out
    ) / 1e6
    spent = (cost * df.batch_id.eq("direct").map({True: 2, False: 1})).sum()
    print(
        f"tokens/article: {df.input_tokens.mean():.0f} in, "
        f"{df.cache_read_tokens.mean():.0f} cache-read, "
        f"{df.cache_write_tokens.mean():.0f} cache-write, "
        f"{df.output_tokens.mean():.0f} out → ${cost.mean():.4f}/article at batch rates; "
        f"${spent:.2f} spent (direct calls bill at twice the batch rate)"
    )


def save(done, rows: list, failures: dict, n_submitted: int, articles, model: str, path) -> None:
    print(f"enrich: {n_submitted} submitted → {len(rows)} recorded ({len(failures)} to retry)")
    for pageid, why in failures.items():
        print(f"  RETRY {pageid}: {why}")
    new = pd.DataFrame(rows, columns=None if rows else ["pageid"])
    merged = pd.concat([done[~done.pageid.isin(new.pageid)], new], ignore_index=True)
    write_parquet_safely(merged.sort_values("pageid"), path)
    report(merged, articles, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--direct", action="store_true", help="concurrent calls instead of a batch (small sets)"
    )
    parser.add_argument("--model", default=MODEL, choices=sorted(BATCH_RATES))
    parser.add_argument(
        "--rerun", action="store_true", help="the map's articles again, into RERUN_PARQUET"
    )
    args = parser.parse_args()
    model = args.model
    assert not (args.rerun and model != MODEL), "the re-run is for the map's model"
    path = RERUN_PARQUET if args.rerun else enrichment_path(model)
    pending_file = RERUN_PENDING_FILE if args.rerun else PENDING_FILE

    client = anthropic.Anthropic()
    articles = pd.read_parquet(ARTICLES_PARQUET)
    done = load_done(path)

    if pending_file.exists():
        pending = json.loads(pending_file.read_text())
        print(f"resuming batch {pending['batch_id']} ({len(pending['pageids'])} requests)")
    else:
        pool = articles
        if args.rerun:
            # Imported here: embed imports this module.
            from embed import load_corpus

            pool = articles[articles.pageid.isin(load_corpus().pageid)]
        todo = pool[~pool.pageid.isin(done.pageid)].sort_values("pageid")
        if args.sample is not None:
            # Fixed seed over the whole corpus, then drop what's done, so the
            # validation set is stable across re-runs.
            todo = todo[todo.pageid.isin(validation_sample(articles, args.sample))]
        if todo.empty:
            print("nothing to enrich")
            report(done, articles, model)
            return
        print(f"{len(done)} already enriched; submitting {len(todo)}")
        if args.direct:
            rows, failures = run_direct(client, list(todo.itertuples()), model)
            save(done, rows, failures, len(todo), articles, model, path)
            return
        assert model == MODEL, "batches are for the map's model; compare others with --direct"
        pending = submit(client, list(todo.itertuples()), pending_file)

    wait(client, pending["batch_id"])
    rows, failures = collect(client, pending)
    save(done, rows, failures, len(pending["pageids"]), articles, model, path)
    os.remove(pending_file)


if __name__ == "__main__":
    main()
