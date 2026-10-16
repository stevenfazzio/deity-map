"""Check taxonomy.TRADITIONS against the rule that defines it, and report the evidence.

The rule (taxonomy.py): a tradition gets a colour, and a class in the archetype map's
eraser, when
  1. anchored: Wikipedia files its gods in a deity category of their own ("Hurrian
     deities", "Shinto kami") under "Deities by continent" or "Deities by religion",
     found by category_labels.py's pattern for it in data/deity_tree.parquet; regional
     residuals (taxonomy.RESIDUAL_TRADITIONS) are exempt, and
  2. big enough: Wikipedia's categories put at least MIN_TRADITION_SIZE of the map's
     articles in it (category_labels.py's tradition_names, multi-valued).

Reported alongside, not part of the rule:
  agreement  Cohen's kappa between enrich.py's label and the categories', for this
             tradition against the rest. Low where the raters split a boundary, or
             where enrich.py was inconsistent (it put 27 Meitei gods under Southeast Asian)
  separable  recall of a 5-fold logistic probe on the pantheon map's embeddings, over
             articles with exactly one category tradition: can the text tell it apart?

    uv run taxonomy_audit.py

Fails if TRADITIONS and the rule disagree. Prints every candidate tradition and the gap
the size floor sits in, and writes data/taxonomy_audit.csv.
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from categories import DEITY_TREE_PARQUET
from category_labels import structured_labels, traditions_of
from common import DATA
from embed import load_corpus
from taxonomy import (
    MIN_TRADITION_SIZE,
    RESIDUAL_TRADITIONS,
    TRADITION_REGIONS,
    TRADITIONS,
    resolve_traditions,
)
from variants import VARIANTS

AUDIT_CSV = DATA / "taxonomy_audit.csv"
# A category of a tradition's gods, rather than of one god ("Inanna") or a place.
COLLECTIVE = re.compile(r"\b(deities|gods|goddesses|divinities|kami|kamuy|nats|orishas|lwa)\b")
FLAG_BELOW = 0.6  # agreement or separability under this is flagged in the table
MIN_PROBE_CLASS = 5


def anchors(tree: pd.DataFrame) -> dict[str, list[str]]:
    """Tradition name -> its deity categories in Wikipedia's tree."""
    found = {name: [] for name in TRADITION_REGIONS}
    for cat in sorted(set(tree.category)):
        if COLLECTIVE.search(cat):
            for name in traditions_of(cat):
                found[name].append(cat)
    # Shortest first: "Greek deities" before "Consorts of Greek deities"
    return {name: sorted(cats, key=len) for name, cats in found.items()}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def separability(corpus: pd.DataFrame, names: pd.Series) -> dict[str, float]:
    """Per-tradition recall of a cross-validated probe on the pantheon embeddings."""
    npz = np.load(VARIANTS["pantheon"].embeddings)
    assert np.array_equal(npz["pageids"], corpus.pageid.to_numpy()), (
        "run embed.py --variant pantheon"
    )
    single = names.str.len() == 1
    y = names[single].str[0].to_numpy()
    counts = pd.Series(y).value_counts()
    keep = np.isin(y, counts[counts >= MIN_PROBE_CLASS].index)
    x, y = npz["vectors"][single.to_numpy()][keep], y[keep]
    pred = cross_val_predict(
        LogisticRegression(max_iter=3000, C=10.0),
        x,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=5,
    )
    print(f"probe on {len(y)} single-tradition articles: accuracy {(pred == y).mean():.3f}")
    return {name: float((pred[y == name] == name).mean()) for name in np.unique(y)}


def main():
    corpus = load_corpus()
    names = structured_labels(corpus).tradition_names
    llm = resolve_traditions(corpus)[0].to_numpy()
    anchored = anchors(pd.read_parquet(DEITY_TREE_PARQUET))
    recall = separability(corpus, names)

    rows = []
    for name, region in TRADITION_REGIONS.items():
        in_cats = names.map(lambda ts, n=name: n in ts).to_numpy()
        rows.append(
            {
                "tradition": name,
                "region": region,
                "articles": int(in_cats.sum()),
                "articles_enrich": int((llm == name).sum()),
                "anchor": "residual"
                if name in RESIDUAL_TRADITIONS
                else "; ".join(anchored[name][:3]),
                "agreement": kappa(llm == name, in_cats),
                "separable": recall.get(name, np.nan),
            }
        )
    t = pd.DataFrame(rows)
    t["passes"] = (t.anchor != "") & (t.articles >= MIN_TRADITION_SIZE)
    t["listed"] = t.tradition.isin(TRADITIONS)
    t = t.sort_values(["passes", "articles"], ascending=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    flags = (t.agreement < FLAG_BELOW) | (t.separable < FLAG_BELOW)
    shown = t.assign(flag=np.where(flags, "*", "")).drop(columns=["passes", "listed"])
    print(shown.round(2).to_string(index=False))
    print(f"* agreement or separability under {FLAG_BELOW}")

    candidates = t[t.anchor != ""]
    below = candidates[candidates.articles < MIN_TRADITION_SIZE].iloc[0]
    above = candidates[candidates.articles >= MIN_TRADITION_SIZE].iloc[-1]
    print(
        f"\nfloor {MIN_TRADITION_SIZE} sits in a gap: the largest anchored tradition below it has "
        f"{below.articles} articles ({below.tradition}), the smallest above {above.articles} "
        f"({above.tradition}), so any floor from {below.articles + 1} to {above.articles} "
        f"selects the same {t.passes.sum()} traditions"
    )
    unanchored = t[(t.anchor == "") & (t.articles >= MIN_TRADITION_SIZE)]
    if len(unanchored):
        print(f"big enough but no deity category of their own: {', '.join(unanchored.tradition)}")

    t.to_csv(AUDIT_CSV, index=False)
    print(f"wrote {AUDIT_CSV}")
    extra = sorted(set(t.tradition[t.listed & ~t.passes]))
    missing = sorted(set(t.tradition[t.passes & ~t.listed]))
    assert not extra and not missing, (
        f"taxonomy.TRADITIONS disagrees with its rule: listed but failing {extra}, "
        f"passing but not listed {missing}"
    )
    print(f"taxonomy.TRADITIONS matches the rule: {len(TRADITIONS)} traditions + Other")


if __name__ == "__main__":
    main()
