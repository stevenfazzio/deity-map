"""Build the candidate pool: every English Wikipedia article that might be about a deity.

Two sources, because neither is complete or clean on its own (checked 2026-10-15):

- Wikidata: items whose `instance of` is deity (Q178885) or any subclass of it, with an
  enwiki article. Precise, but it misses whole groups whose classes aren't linked into
  the deity tree: Vodou loa, deified humans typed `human` (Guan Yu), tricksters,
  theonyms (Allah, Waheguru).
- Wikipedia's Category:Deities, crawled to depth 3. Recovers all of those, but pulls in
  temples, festivals, artworks and fiction, and the noise outgrows the signal past
  depth 3 (18k articles at depth 4, 25k at depth 5).

The union is deliberately generous: enrich.py classifies every candidate from its lead
section, and the map's scope is a filter on that classification.

    uv run candidates.py

Writes data/candidates.parquet (title, qid, sources, category depth) and keeps the raw
query responses in data/raw/ for reproducibility.
"""

import json
import time

import pandas as pd

from common import (
    CANDIDATES_PARQUET,
    DATA,
    WDQS,
    get_with_retry,
    make_session,
    write_bytes_atomic,
    write_parquet_safely,
)

RAW = DATA / "raw"
PETSCAN = "https://petscan.wmcloud.org/"
ROOT_CATEGORY = "Deities"
MAX_DEPTH = 3
DEITY_QID = "Q178885"

WIKIDATA_QUERY = f"""
SELECT ?item ?title (GROUP_CONCAT(DISTINCT ?classLabel; separator="|") AS ?classes) WHERE {{
  ?item wdt:P31/wdt:P279* wd:{DEITY_QID} .
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?title .
  OPTIONAL {{
    ?item wdt:P31 ?class .
    ?class rdfs:label ?classLabel . FILTER(LANG(?classLabel) = "en")
  }}
}} GROUP BY ?item ?title
"""


def fetch_wikidata(session) -> pd.DataFrame:
    resp = get_with_retry(
        session,
        WDQS,
        params={"query": WIKIDATA_QUERY, "format": "json"},  # XML otherwise
        timeout=180,
    )
    write_bytes_atomic(resp.content, RAW / "wikidata_deities.json")
    rows = [
        {
            "title": b["title"]["value"],
            "qid": b["item"]["value"].rsplit("/", 1)[1],
            "wd_classes": b["classes"]["value"],
        }
        for b in resp.json()["results"]["bindings"]
    ]
    df = pd.DataFrame(rows)
    print(f"wikidata: {len(df)} enwiki articles in the deity tree")
    return df


def fetch_category(session, depth: int) -> pd.DataFrame:
    params = {
        "language": "en",
        "project": "wikipedia",
        "categories": ROOT_CATEGORY,
        "depth": depth,
        "ns[0]": 1,
        "format": "json",
        "doit": 1,
    }
    resp = get_with_retry(session, PETSCAN, params=params, timeout=300)
    write_bytes_atomic(resp.content, RAW / f"petscan_{ROOT_CATEGORY}_depth{depth}.json")
    pages = resp.json()["*"][0]["a"]["*"]
    df = pd.DataFrame(
        {
            "title": [p["title"].replace("_", " ") for p in pages],
            "pageid": [p["id"] for p in pages],
            "qid": [p.get("q") for p in pages],
        }
    )
    print(f"Category:{ROOT_CATEGORY} depth {depth}: {len(df)} articles")
    return df


def main():
    session = make_session()
    wd = fetch_wikidata(session)

    # Crawl each depth so every article records the shallowest depth it appears at: a
    # useful diagnostic when judging what the deeper levels contribute.
    by_depth = []
    for depth in range(1, MAX_DEPTH + 1):
        cat = fetch_category(session, depth)
        cat["cat_depth"] = depth
        by_depth.append(cat)
        time.sleep(2)
    cat = (
        pd.concat(by_depth)
        .sort_values("cat_depth")
        .drop_duplicates("title", keep="first")
        .reset_index(drop=True)
    )

    df = wd.merge(cat, on="title", how="outer", suffixes=("_wd", "_cat"), indicator=True)
    df["in_wikidata"] = df._merge.isin(["both", "left_only"])
    df["in_category"] = df._merge.isin(["both", "right_only"])
    df["qid"] = df.qid_wd.fillna(df.qid_cat)
    df = df.drop(columns=["_merge", "qid_wd", "qid_cat"])
    print(
        f"union: {len(df)} titles "
        f"({(df.in_wikidata & df.in_category).sum()} both, "
        f"{(df.in_wikidata & ~df.in_category).sum()} wikidata only, "
        f"{(~df.in_wikidata & df.in_category).sum()} category only)"
    )

    # Known non-articles only; anything subtler is left to the classifier.
    is_list = df.title.str.match(r"^Lists? of ")
    print(f"dropped {is_list.sum()} list articles")
    df = df[~is_list]

    df["pageid"] = df.pageid.astype("Int64")
    df["cat_depth"] = df.cat_depth.astype("Int64")
    df = df.sort_values("title").reset_index(drop=True)
    write_parquet_safely(df, CANDIDATES_PARQUET)
    meta = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(df)}
    write_bytes_atomic(json.dumps(meta).encode(), RAW / "candidates_meta.json")
    print(f"wrote {CANDIDATES_PARQUET} ({len(df)} candidates)")


if __name__ == "__main__":
    main()
