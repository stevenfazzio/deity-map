"""Fetch lead sections and page metadata for every candidate.

Two passes, each checkpointed to JSONL so a re-run only fetches what's missing:

  A. prop=info|pageprops|description|pageimages, 50 titles/request: canonical title
     after redirects, pageid, QID, disambiguation flag, Wikipedia's short description,
     thumbnail, article length.
  B. prop=extracts (exintro, explaintext), 20 pages/request: the lead section. This is
     the semantic text, used to classify (enrich.py) and to embed (embed.py).

Pageviews are fetched later, by pageviews.py, for in-scope articles only.

    uv run fetch.py

Writes data/articles.parquet, one row per distinct article (several candidate titles
can redirect to the same page).
"""

import json
import time
from pathlib import Path

import pandas as pd

from common import (
    ARTICLES_PARQUET,
    CANDIDATES_PARQUET,
    DATA,
    make_session,
    wiki_query,
    write_parquet_safely,
)

CACHE = DATA / "fetch_cache"
PAGES_JSONL = CACHE / "pages.jsonl"
EXTRACTS_JSONL = CACHE / "extracts.jsonl"
THUMB_SIZE = 320
SLEEP_S = 0.2


def load_jsonl(path: Path, key: str) -> dict:
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                done[rec[key]] = rec
    return done


def fetch_pages(titles: list[str]) -> dict:
    """Pass A, keyed by the *requested* title so redirects map back to candidates."""
    done = load_jsonl(PAGES_JSONL, "requested")
    todo = [t for t in titles if t not in done]
    print(f"pages: {len(done)} cached, {len(todo)} to fetch")
    session = make_session()
    with open(PAGES_JSONL, "a") as out:
        for n, i in enumerate(range(0, len(todo), 50)):
            batch = todo[i : i + 50]
            data = wiki_query(
                session,
                {
                    "titles": "|".join(batch),
                    "redirects": "1",
                    "prop": "info|pageprops|description|pageimages",
                    "ppprop": "wikibase_item|disambiguation",
                    "piprop": "thumbnail",
                    "pithumbsize": str(THUMB_SIZE),
                },
            )["query"]
            # Follow normalization, then redirects, from each requested title.
            hop = {r["from"]: r["to"] for r in data.get("normalized", [])}
            hop |= {r["from"]: r["to"] for r in data.get("redirects", [])}
            pages = {p["title"]: p for p in data.get("pages", [])}
            for requested in batch:
                title = requested
                for _ in range(3):
                    title = hop.get(title, title)
                page = pages.get(title, {})
                rec = {
                    "requested": requested,
                    "title": page.get("title"),
                    "pageid": page.get("pageid"),
                    "missing": "pageid" not in page,
                    "qid": page.get("pageprops", {}).get("wikibase_item"),
                    "disambiguation": "disambiguation" in page.get("pageprops", {}),
                    "description": page.get("description"),
                    "thumb_url": page.get("thumbnail", {}).get("source"),
                    "length": page.get("length"),
                }
                out.write(json.dumps(rec) + "\n")
                done[requested] = rec
            out.flush()
            if n % 40 == 0:
                print(f"  pages: {len(done)}/{len(titles)}", flush=True)
            time.sleep(SLEEP_S)
    return done


def fetch_extracts(pageids: list[int]) -> dict:
    """Pass B: plain-text lead sections, keyed by pageid."""
    done = load_jsonl(EXTRACTS_JSONL, "pageid")
    todo = [p for p in pageids if p not in done]
    print(f"extracts: {len(done)} cached, {len(todo)} to fetch")
    session = make_session()
    with open(EXTRACTS_JSONL, "a") as out:
        for n, i in enumerate(range(0, len(todo), 20)):
            batch = todo[i : i + 20]
            data = wiki_query(
                session,
                {
                    "pageids": "|".join(map(str, batch)),
                    "prop": "extracts",
                    "exintro": "1",
                    "explaintext": "1",
                    "exlimit": "max",
                },
            )["query"]
            pages = {p["pageid"]: p for p in data.get("pages", [])}
            for pageid in batch:
                rec = {"pageid": pageid, "lead": pages.get(pageid, {}).get("extract")}
                out.write(json.dumps(rec) + "\n")
                done[pageid] = rec
            out.flush()
            if n % 100 == 0:
                print(f"  extracts: {len(done)}/{len(pageids)}", flush=True)
            time.sleep(SLEEP_S)
    return done


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    cands = pd.read_parquet(CANDIDATES_PARQUET)

    pages = fetch_pages(sorted(cands.title))
    resolved = pd.DataFrame([pages[t] for t in cands.title]).drop(columns="requested")
    resolved = resolved.rename(columns={"title": "canonical_title", "qid": "qid_page"})
    # PetScan's pageid is superseded by the one resolved through redirects.
    df = pd.concat([cands.drop(columns="pageid").reset_index(drop=True), resolved], axis=1)

    n_in = len(df)
    missing = df.missing
    print(f"dropped {missing.sum()} titles that no longer exist")
    df = df[~missing]
    disambig = df.disambiguation
    print(f"dropped {disambig.sum()} disambiguation pages")
    df = df[~disambig]

    # Several candidate titles can redirect to one page: merge them, keeping the
    # union of sources and the shallowest category depth.
    df["qid"] = df.qid_page.fillna(df.qid)
    df = (
        df.groupby("pageid", as_index=False)
        .agg(
            title=("canonical_title", "first"),
            qid=("qid", "first"),
            in_wikidata=("in_wikidata", "any"),
            in_category=("in_category", "any"),
            cat_depth=("cat_depth", "min"),
            wd_classes=("wd_classes", "first"),
            description=("description", "first"),
            thumb_url=("thumb_url", "first"),
            length=("length", "first"),
            n_candidate_titles=("title", "size"),
        )
        .sort_values("title")
        .reset_index(drop=True)
    )
    print(
        f"articles: {n_in} candidate titles → {len(df)} distinct articles "
        f"({n_in - missing.sum() - disambig.sum() - len(df)} merged as redirects)"
    )

    extracts = fetch_extracts(df.pageid.tolist())
    df["lead"] = df.pageid.map(lambda p: extracts[p]["lead"])
    df["lead_words"] = df.lead.fillna("").str.split().str.len()
    print(
        f"lead words: median {df.lead_words.median():.0f}, "
        f"{(df.lead_words == 0).sum()} empty, {(df.lead_words < 20).sum()} under 20"
    )

    write_parquet_safely(df, ARTICLES_PARQUET)
    print(f"wrote {ARTICLES_PARQUET} ({len(df)} articles)")


if __name__ == "__main__":
    main()
