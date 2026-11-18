"""Score every map variant's layout with the README's neighbour metrics.

For the two published maps and the two ablations in variants.py (erasure only,
instruction only): the share of each deity's 15 nearest neighbours on the 2-d layout a
viewer sees that
  tradition     share its tradition colour (Wikipedia's categories decide, as on the map)
  domain        share enrich.py's main domain (unknown and other left out)
  function      share one of Wikipedia's "Deities by function" categories
                (category_labels.py), among pairs where both sides have one
  cross_domain  among neighbours from a different tradition, share the main domain
Chance is the same share over random pairs.

Reads each variant's cached UMAP layout (cluster.py writes it, --explore suffices).

    uv run python experiments/score_variants.py

Writes data/experiments/variant_scores.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from category_labels import colormap_traditions, structured_labels
from common import DATA
from embed import load_corpus
from taxonomy import DOMAIN_LABELS
from variants import VARIANTS

K = 15
SEED = 42
ORDER = ["pantheon", "erasure-only", "instruction-only", "archetype"]


def labels(corpus: pd.DataFrame) -> dict:
    tradition = pd.Categorical(colormap_traditions(corpus)[1]).codes
    dom_vocab = sorted(set(DOMAIN_LABELS) - {"unknown", "other"})
    domain = pd.Categorical(corpus.primary_domain, categories=dom_vocab).codes  # -1: none
    functions = structured_labels(corpus).domains
    vocab = sorted({d for ds in functions for d in ds})
    index = {d: i for i, d in enumerate(vocab)}
    function = np.zeros((len(corpus), len(vocab)), dtype=bool)
    for row, ds in enumerate(functions):
        function[row, [index[d] for d in ds]] = True
    return {"tradition": tradition, "domain": domain, "function": function}


def shares(nn: np.ndarray, lab: dict) -> dict[str, float]:
    i, j = np.repeat(np.arange(len(nn)), nn.shape[1]), nn.ravel()
    t, d, f = lab["tradition"], lab["domain"], lab["function"]
    known = (d[i] >= 0) & (d[j] >= 0)
    both_f = f[i].any(1) & f[j].any(1)
    cross = (t[i] != t[j]) & known
    return {
        "tradition": float((t[i] == t[j]).mean()),
        "domain": float((d[i] == d[j])[known].mean()),
        "function": float((f[i] & f[j]).any(1)[both_f].mean()),
        "cross_domain": float((d[i] == d[j])[cross].mean()),
    }


def main():
    corpus = load_corpus()
    pageids = corpus.pageid.to_numpy()
    lab = labels(corpus)

    rng = np.random.default_rng(SEED)
    rows = {"chance": shares(rng.integers(0, len(corpus), size=(len(corpus), K)), lab)}
    for name in ORDER:
        cached = np.load(VARIANTS[name].coords)
        assert np.array_equal(cached["pageids"], pageids), f"{name}: layout out of date"
        xy = cached["coords"]
        nn = NearestNeighbors(n_neighbors=K + 1).fit(xy).kneighbors(xy, return_distance=False)
        rows[name] = shares(nn[:, 1:], lab)

    table = pd.DataFrame(rows).T
    out = DATA / "experiments" / "variant_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(table.round(2).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
