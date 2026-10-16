"""Embed each in-scope article's name + lead section, for one map variant.

The map's corpus is defined here, by load_corpus(): the articles enrich.py classified
as one of the in-scope kinds, less the hand exclusions in exclusions.py. Every later
stage reads it through this function and checks its pageids against the embeddings file.

Both variants embed the same text with Qwen3-Embedding-8B (via OpenRouter, ~$0.01).
The archetype variant adds an instruction asking for the deity's roles, then erases
tradition with LEACE. The eraser is fit on the tradition categories (taxonomy.TRADITIONS
and Other) that Wikipedia's own categories give each article, through category_labels.py:
editors' labels, several where a god belongs to several traditions. enrich.py's labels
are kept out of the fit to serve as the check: embed.py fits a second eraser on a training
split and reports how well a probe still finds enrich.py's tradition in held-out articles.
The fitted eraser is saved, because cluster.py must apply the same one to Toponymy's
keyphrases. See variants.py for why these recipes.

    uv run embed.py --variant pantheon
    uv run embed.py --variant archetype

Writes data/embeddings_<variant>.npz (pageids, vectors), row-aligned with load_corpus(),
and for the archetype variant data/eraser_archetype.pt. Vectors come through
embedder.py's disk cache, so re-running is free for any article whose text hasn't changed.
"""

import argparse
import io
import re

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import ARTICLES_PARQUET, write_bytes_atomic, write_npz_atomic
from embedder import OpenRouterEmbedder
from enrich import ENRICHMENT_PARQUET, IN_SCOPE
from exclusions import DUPLICATES, EXCLUDED
from taxonomy import TRADITIONS, tradition_codes
from variants import VARIANTS, MapVariant, variant_arg


def display_name(title: str) -> str:
    """'Pele (deity)' -> 'Pele'. The lead says what the parenthetical did."""
    return re.sub(r"\s+\([^)]*\)$", "", title)


def load_corpus() -> pd.DataFrame:
    articles = pd.read_parquet(ARTICLES_PARQUET)
    enrichment = pd.read_parquet(ENRICHMENT_PARQUET)
    ok = enrichment[enrichment.status == "ok"]
    df = articles.merge(ok, on="pageid", how="inner", validate="1:1")
    in_scope = df[df.kind.isin(IN_SCOPE)]
    stale = (EXCLUDED | set(DUPLICATES.values())) - set(in_scope.title)
    assert not stale, f"exclusions.py names titles that are not in scope: {stale}"
    corpus = in_scope[~in_scope.title.isin(EXCLUDED)]
    corpus = corpus.sort_values("pageid").reset_index(drop=True)
    corpus["name"] = corpus.title.map(display_name)
    return corpus


def document_text(row) -> str:
    # The name anchors leads that open obliquely; a handful of articles have no lead
    # at all, and Wikipedia's short description is the best stand-in.
    body = (row.lead or "").strip() or (row.description or "")
    return f"{row.name}\n\n{body}"


def load_eraser(variant: MapVariant) -> LeaceEraser:
    return torch.load(variant.eraser, weights_only=False)


def tradition_matrix(corpus: pd.DataFrame) -> np.ndarray:
    """Multi-hot tradition categories from Wikipedia's categories, one row per article.

    All zero for the few articles whose categories name no tradition: they stay out of
    the eraser's fit, but are erased like the rest.
    """
    # Imported here: category_labels imports this module for load_corpus().
    from category_labels import structured_labels

    index = {c: i for i, c in enumerate([*TRADITIONS, "Other"])}
    z = np.zeros((len(corpus), len(index)), dtype=np.float32)
    for row, cats in enumerate(structured_labels(corpus).traditions):
        z[row, [index[c] for c in cats]] = 1
    return z


def fit_eraser(vectors: np.ndarray, z: np.ndarray) -> LeaceEraser:
    labelled = z.any(axis=1)
    return LeaceEraser.fit(torch.from_numpy(vectors[labelled]), torch.from_numpy(z[labelled]))


def erase(eraser: LeaceEraser, vectors: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return eraser(torch.from_numpy(vectors)).numpy().astype(np.float32)


def class_mean_spread(vectors: np.ndarray, z: np.ndarray) -> float:
    """RMS distance of each tradition's mean from the labelled articles' mean: ~0 once erased."""
    labelled = z.any(axis=1)
    v, z = vectors[labelled], z[labelled]
    present = z.sum(axis=0) > 0
    means = (z.T[present] @ v) / z.sum(axis=0)[present, None]
    return float(np.sqrt(((means - v.mean(axis=0)) ** 2).sum(axis=1).mean()))


def check_against_enrich(vectors: np.ndarray, z: np.ndarray, corpus: pd.DataFrame) -> None:
    """The independent check: erase with the categories, probe for enrich.py's labels.

    Eraser fit on a training split only, probe scored on the held-out articles, since an
    eraser fit on everything makes held-out probes look anti-predictive.
    """
    codes, _ = tradition_codes(corpus)
    tr, te = train_test_split(np.arange(len(codes)), test_size=0.2, random_state=42, stratify=codes)
    x = erase(fit_eraser(vectors[tr], z[tr]), vectors)
    probe = LogisticRegression(max_iter=3000, C=10.0).fit(x[tr], codes[tr])
    accuracy = (probe.predict(x[te]) == codes[te]).mean()
    majority = (codes[te] == np.bincount(codes[tr]).argmax()).mean()
    print(
        f"  check: a probe finds enrich.py's tradition in {accuracy:.1%} of held-out "
        f"articles after erasure (majority class {majority:.1%})"
    )


def main():
    parser = argparse.ArgumentParser()
    variant_arg(parser)
    variant = VARIANTS[parser.parse_args().variant]

    articles = pd.read_parquet(ARTICLES_PARQUET)
    enrichment = pd.read_parquet(ENRICHMENT_PARQUET)
    corpus = load_corpus()
    not_enriched = len(articles) - enrichment.pageid.isin(articles.pageid).sum()
    refused = (enrichment.status == "refusal").sum()
    print(
        f"embed ({variant.name}): {len(articles)} articles → {len(corpus)} in scope "
        f"({not_enriched} not yet enriched, {refused} refusals, {len(EXCLUDED)} excluded "
        f"by hand, the rest classified out of scope)"
    )
    assert not_enriched == 0, "run enrich.py to completion first"

    embedder = OpenRouterEmbedder(instruction=variant.instruction)
    texts = [document_text(r) for r in corpus.itertuples()]
    vectors = embedder.encode(texts, show_progress_bar=True).astype(np.float32)
    print(f"  {embedder.model_name}: ${embedder.cost_usd:.4f} spent on uncached text")
    assert vectors.shape[0] == len(corpus), (vectors.shape, len(corpus))
    assert np.isfinite(vectors).all()

    if variant.erase_tradition:
        z = tradition_matrix(corpus)
        print(
            f"  erasing {z.shape[1]} tradition categories from Wikipedia's categories: "
            f"{int(z.any(axis=1).sum())} articles labelled, {int((z.sum(axis=1) > 1).sum())} "
            f"with several"
        )
        check_against_enrich(vectors, z, corpus)
        before = class_mean_spread(vectors, z)
        eraser = fit_eraser(vectors, z)
        vectors = erase(eraser, vectors)
        after = class_mean_spread(vectors, z)
        print(f"  tradition erased: class-mean spread {before:.4f} → {after:.6f}")
        assert after < 1e-3 * before, "eraser did not remove the tradition means"
        buf = io.BytesIO()
        torch.save(eraser, buf)
        write_bytes_atomic(buf.getvalue(), variant.eraser)
        print(f"  wrote {variant.eraser}")

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  {vectors.shape}, norms {norms.min():.3f}–{norms.max():.3f}")

    def check(loaded):
        assert loaded["vectors"].shape == vectors.shape
        assert list(loaded["pageids"]) == list(corpus.pageid)

    write_npz_atomic(
        variant.embeddings, check=check, pageids=corpus.pageid.to_numpy(), vectors=vectors
    )
    print(f"wrote {variant.embeddings}")


if __name__ == "__main__":
    main()
