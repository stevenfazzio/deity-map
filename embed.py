"""Embed each in-scope article's name + lead section, for one map variant.

The map's corpus is defined here, by load_corpus(): the articles enrich.py classified
as one of the in-scope kinds. Every later stage reads it through this function and
checks its pageids against the embeddings file.

Both variants embed the same text with Qwen3-Embedding-8B (via OpenRouter, ~$0.01).
The archetype variant adds an instruction asking for the deity's roles, then erases
tradition with LEACE, fit on the 38 tradition categories the Tradition colormap shows.
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

from common import ARTICLES_PARQUET, write_bytes_atomic, write_npz_atomic
from embedder import OpenRouterEmbedder
from enrich import ENRICHMENT_PARQUET, IN_SCOPE
from taxonomy import tradition_codes
from variants import VARIANTS, MapVariant, variant_arg


def display_name(title: str) -> str:
    """'Pele (deity)' -> 'Pele'. The lead says what the parenthetical did."""
    return re.sub(r"\s+\([^)]*\)$", "", title)


def load_corpus() -> pd.DataFrame:
    articles = pd.read_parquet(ARTICLES_PARQUET)
    enrichment = pd.read_parquet(ENRICHMENT_PARQUET)
    ok = enrichment[enrichment.status == "ok"]
    df = articles.merge(ok, on="pageid", how="inner", validate="1:1")
    corpus = df[df.kind.isin(IN_SCOPE)].sort_values("pageid").reset_index(drop=True)
    corpus["name"] = corpus.title.map(display_name)
    return corpus


def document_text(row) -> str:
    # The name anchors leads that open obliquely; a handful of articles have no lead
    # at all, and Wikipedia's short description is the best stand-in.
    body = (row.lead or "").strip() or (row.description or "")
    return f"{row.name}\n\n{body}"


def load_eraser(variant: MapVariant) -> LeaceEraser:
    return torch.load(variant.eraser, weights_only=False)


def fit_eraser(vectors: np.ndarray, corpus: pd.DataFrame) -> LeaceEraser:
    codes, k = tradition_codes(corpus)
    z = torch.nn.functional.one_hot(torch.from_numpy(codes), k).float()
    return LeaceEraser.fit(torch.from_numpy(vectors), z)


def class_mean_spread(vectors: np.ndarray, corpus: pd.DataFrame) -> float:
    """RMS distance of the tradition means from the overall mean: ~0 once erased."""
    codes, k = tradition_codes(corpus)
    mu = vectors.mean(axis=0)
    means = np.stack([vectors[codes == c].mean(axis=0) for c in range(k)])
    return float(np.sqrt(((means - mu) ** 2).sum(axis=1).mean()))


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
        f"({not_enriched} not yet enriched, {refused} refusals, "
        f"the rest classified out of scope)"
    )
    assert not_enriched == 0, "run enrich.py to completion first"

    embedder = OpenRouterEmbedder(instruction=variant.instruction)
    texts = [document_text(r) for r in corpus.itertuples()]
    vectors = embedder.encode(texts, show_progress_bar=True).astype(np.float32)
    print(f"  {embedder.model_name}: ${embedder.cost_usd:.4f} spent on uncached text")
    assert vectors.shape[0] == len(corpus), (vectors.shape, len(corpus))
    assert np.isfinite(vectors).all()

    if variant.erase_tradition:
        before = class_mean_spread(vectors, corpus)
        eraser = fit_eraser(vectors, corpus)
        with torch.no_grad():
            vectors = eraser(torch.from_numpy(vectors)).numpy().astype(np.float32)
        after = class_mean_spread(vectors, corpus)
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
