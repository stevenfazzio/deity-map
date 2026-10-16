"""Monthly English Wikipedia pageviews for the articles on the map.

Runs after enrich.py and fetches only in-scope articles: the REST endpoint is one
request per article and rate-limits hard (6 parallel workers drew steady 429s), so
there's no point paying for the ~half of the candidate pool that never gets plotted.

Human traffic only, over a fixed 12-month window so the colormap doesn't drift between
runs. Views of redirects to an article are not included, and an article renamed
mid-window loses its views under the old title; both are small undercounts.

    uv run pageviews.py

Writes data/pageviews.parquet (pageid, views_12mo, views_median_month).
"""

import json
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from common import ARTICLES_PARQUET, DATA, get_with_retry, make_session, write_parquet_safely
from enrich import ENRICHMENT_PARQUET, IN_SCOPE

PAGEVIEWS_JSONL = DATA / "fetch_cache" / "pageviews.jsonl"
PAGEVIEWS_PARQUET = DATA / "pageviews.parquet"
PV_START, PV_END = "2025100100", "2026093000"
PV_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}"
)
WORKERS = 2
CHUNK = 200

_local = threading.local()


def load_done() -> dict:
    done = {}
    if PAGEVIEWS_JSONL.exists():
        with open(PAGEVIEWS_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["title"]] = rec
    return done


def fetch_one(title: str) -> dict:
    # One session per worker thread: requests.Session isn't documented as thread-safe.
    if not hasattr(_local, "session"):
        _local.session = make_session()
    url = PV_URL.format(
        title=urllib.parse.quote(title.replace(" ", "_"), safe=""), start=PV_START, end=PV_END
    )
    try:
        items = get_with_retry(_local.session, url).json()["items"]
    except requests.HTTPError as e:
        # 404 means no recorded views in the window (e.g. a brand-new article).
        if e.response is not None and e.response.status_code == 404:
            items = []
        else:
            raise
    return {"title": title, "monthly": {it["timestamp"][:6]: it["views"] for it in items}}


def main():
    articles = pd.read_parquet(ARTICLES_PARQUET)
    enrichment = pd.read_parquet(ENRICHMENT_PARQUET)
    in_scope = enrichment.loc[enrichment.kind.isin(IN_SCOPE), "pageid"]
    df = articles[articles.pageid.isin(in_scope)][["pageid", "title"]].reset_index(drop=True)

    done = load_done()
    todo = [t for t in sorted(df.title) if t not in done]
    print(
        f"pageviews: {len(df)} in-scope articles, {len(df) - len(todo)} cached, {len(todo)} to fetch"
    )
    PAGEVIEWS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(PAGEVIEWS_JSONL, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(todo), CHUNK):
            # A chunk at a time, so each chunk is on disk before the next starts.
            for rec in pool.map(fetch_one, todo[i : i + CHUNK]):
                out.write(json.dumps(rec) + "\n")
                done[rec["title"]] = rec
            out.flush()
            print(f"  pageviews: {i + CHUNK}/{len(todo)}", flush=True)

    months = pd.period_range(PV_START[:6], PV_END[:6], freq="M").strftime("%Y%m")
    mat = np.array([[done[t]["monthly"].get(m, 0) for m in months] for t in df.title])
    df["views_12mo"] = mat.sum(axis=1)
    # Median of the monthly totals: robust to a one-month spike from a game or film.
    df["views_median_month"] = np.median(mat, axis=1)
    print(
        f"pageviews: median {df.views_12mo.median():.0f}/yr, "
        f"{(df.views_12mo == 0).sum()} articles with none recorded"
    )
    write_parquet_safely(df[["pageid", "views_12mo", "views_median_month"]], PAGEVIEWS_PARQUET)
    print(f"wrote {PAGEVIEWS_PARQUET}")


if __name__ == "__main__":
    main()
