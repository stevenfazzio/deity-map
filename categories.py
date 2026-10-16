"""Fetch Wikipedia's categories for every in-scope article, the tree above them, and its deity tree.

The structured counterpart to enrich.py's tradition and domain labels: category_labels.py
maps these categories to the same traditions and domains, so the eraser can be fit on
labels that editors assigned, and the LLM's labels kept as an independent check.

Three passes, each checkpointed to JSONL so a re-run only fetches what's missing:

  A. prop=categories for each article, 50 pages/request, hidden (maintenance)
     categories excluded.
  B. the parents of those categories, PARENT_DEPTH levels up, so an article filed only
     under "Muses (mythology)" or "Navagraha" can be resolved through its parents.
  C. Wikipedia's own classification of gods, top down: the subcategories of "Deities by
     continent" and "Deities by religion", TREE_DEPTH levels down. taxonomy_audit.py
     checks that each tradition with a colour of its own has a deity category there.

    uv run categories.py

Writes data/categories.parquet (pageid, category), data/category_parents.parquet
(category, parent, depth) and data/deity_tree.parquet (category, parent, depth),
category names without the "Category:" prefix.
"""

import json
import time

import pandas as pd

from common import DATA, make_session, wiki_query, write_parquet_safely
from embed import load_corpus
from fetch import CACHE, SLEEP_S, load_jsonl

ARTICLE_CATS_JSONL = CACHE / "categories.jsonl"
PARENT_CATS_JSONL = CACHE / "category_parents.jsonl"
DEITY_TREE_JSONL = CACHE / "deity_tree.jsonl"
CATEGORIES_PARQUET = DATA / "categories.parquet"
CATEGORY_PARENTS_PARQUET = DATA / "category_parents.parquet"
DEITY_TREE_PARQUET = DATA / "deity_tree.parquet"
# Two levels resolve the sub-categories that carry no tradition in their own name
# (Oceanids → Greek nymphs); further up, the tree turns generic fast.
PARENT_DEPTH = 2
TREE_ROOTS = ["Deities by continent", "Deities by religion"]
# Three levels reach every culture's own deity category (European deities → Greek
# deities → Greek gods); below that come single gods' categories.
TREE_DEPTH = 3
PREFIX = "Category:"


def query_categories(session, params: dict) -> dict:
    """prop=categories for up to 50 pages, following continuation: {page title: [cats]}."""
    params = {"prop": "categories", "clshow": "!hidden", "cllimit": "max", **params}
    cats, cont = {}, {}
    while True:
        data = wiki_query(session, {**params, **cont})
        for page in data["query"].get("pages", []):
            # A missing pageid comes back without a title, a missing title without a pageid.
            key = page.get("pageid") if "pageids" in params else page.get("title")
            cats.setdefault(key, []).extend(
                c["title"].removeprefix(PREFIX) for c in page.get("categories", [])
            )
        if "continue" not in data:
            return cats
        cont = data["continue"]


def fetch_article_categories(pageids: list[int]) -> dict:
    """Pass A, keyed by pageid."""
    done = load_jsonl(ARTICLE_CATS_JSONL, "pageid")
    todo = [p for p in pageids if p not in done]
    print(f"article categories: {len(done)} cached, {len(todo)} to fetch")
    session = make_session()
    with open(ARTICLE_CATS_JSONL, "a") as out:
        for n, i in enumerate(range(0, len(todo), 50)):
            batch = todo[i : i + 50]
            cats = query_categories(session, {"pageids": "|".join(map(str, batch))})
            for pageid in batch:
                rec = {"pageid": pageid, "categories": sorted(set(cats.get(pageid, [])))}
                out.write(json.dumps(rec) + "\n")
                done[pageid] = rec
            out.flush()
            if n % 20 == 0:
                print(f"  article categories: {len(done)}/{len(pageids)}", flush=True)
            time.sleep(SLEEP_S)
    return done


def fetch_parents(categories: list[str]) -> dict:
    """Pass B for one level, keyed by category name."""
    done = load_jsonl(PARENT_CATS_JSONL, "category")
    todo = [c for c in categories if c not in done]
    print(f"category parents: {len(done)} cached, {len(todo)} to fetch")
    session = make_session()
    with open(PARENT_CATS_JSONL, "a") as out:
        for n, i in enumerate(range(0, len(todo), 50)):
            batch = todo[i : i + 50]
            cats = query_categories(session, {"titles": "|".join(PREFIX + c for c in batch)})
            for cat in batch:
                rec = {"category": cat, "parents": sorted(set(cats.get(PREFIX + cat, [])))}
                out.write(json.dumps(rec) + "\n")
                done[cat] = rec
            out.flush()
            if n % 20 == 0:
                print(f"  category parents: {len(done)} cached of this run's levels", flush=True)
            time.sleep(SLEEP_S)
    return done


def query_subcategories(session, category: str) -> list[str]:
    params = {"list": "categorymembers", "cmtitle": PREFIX + category, "cmtype": "subcat"}
    subs, cont = [], {}
    while True:
        data = wiki_query(session, {**params, "cmlimit": "max", **cont})
        subs += [m["title"].removeprefix(PREFIX) for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return subs
        cont = data["continue"]


def fetch_deity_tree() -> pd.DataFrame:
    """Pass C, one request per category expanded; cached by category name."""
    done = load_jsonl(DEITY_TREE_JSONL, "category")
    session = make_session()
    edges, seen, level = [], set(TREE_ROOTS), list(TREE_ROOTS)
    with open(DEITY_TREE_JSONL, "a") as out:
        for depth in range(1, TREE_DEPTH + 1):
            nxt = []
            for cat in level:
                if cat not in done:
                    rec = {
                        "category": cat,
                        "subcategories": sorted(query_subcategories(session, cat)),
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    done[cat] = rec
                    time.sleep(SLEEP_S)
                for sub in done[cat]["subcategories"]:
                    edges.append((sub, cat, depth))
                    if sub not in seen:
                        seen.add(sub)
                        nxt.append(sub)
            print(f"  deity tree depth {depth}: {len(nxt)} new categories", flush=True)
            level = nxt
    return pd.DataFrame(edges, columns=["category", "parent", "depth"])


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    article_cats = fetch_article_categories(corpus.pageid.tolist())
    rows = [(p, c) for p in corpus.pageid for c in article_cats[p]["categories"]]
    cats = pd.DataFrame(rows, columns=["pageid", "category"])
    no_cats = corpus.pageid.nunique() - cats.pageid.nunique()
    print(
        f"articles: {len(corpus)} → {len(cats)} (article, category) pairs, "
        f"{cats.category.nunique()} distinct categories, {no_cats} articles with none"
    )

    edges, seen, level = [], set(), sorted(set(cats.category))
    for depth in range(1, PARENT_DEPTH + 1):
        parents = fetch_parents(level)
        seen |= set(level)
        edges += [(c, p, depth) for c in level for p in parents[c]["parents"]]
        level = sorted({p for c in level for p in parents[c]["parents"]} - seen)
        print(f"  depth {depth}: {len(edges)} edges so far, {len(level)} new parents")
    tree = pd.DataFrame(edges, columns=["category", "parent", "depth"])
    deity_tree = fetch_deity_tree()

    write_parquet_safely(cats, CATEGORIES_PARQUET)
    write_parquet_safely(tree, CATEGORY_PARENTS_PARQUET)
    write_parquet_safely(deity_tree, DEITY_TREE_PARQUET)
    print(f"wrote {CATEGORIES_PARQUET}, {CATEGORY_PARENTS_PARQUET} and {DEITY_TREE_PARQUET}")


if __name__ == "__main__":
    main()
