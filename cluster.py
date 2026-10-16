"""Reduce one variant's embeddings to 2-d, then find and name regions with Toponymy.

Clustering runs on the same 2-d coordinates that get plotted, so named regions
match what a viewer sees. That makes UMAP's min_dist a clustering parameter:
it stays low so regions come out dense enough for density clustering.

Toponymy embeds keyphrases and candidate names during fit and compares them with the
article vectors, so its embedder is the variant's own: same model, same instruction,
and for the archetype map the same saved tradition eraser.

    uv run cluster.py --variant pantheon --explore            # free: layer counts
    uv run cluster.py --variant pantheon --min-cluster-size N # paid: fit + LLM naming

Writes data/labels_<variant>.parquet with x, y and label_layer_0..k, where layer 0 is
the FINEST layer (Toponymy's in-memory order, which DataMapPlot also expects).
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import umap
from toponymy import Toponymy, ToponymyClusterer

from common import write_bytes_atomic, write_npz_atomic, write_parquet_safely
from embed import document_text, load_corpus, load_eraser
from embedder import ErasedEmbedder, OpenRouterEmbedder
from namer import AsyncClaudeNamer
from variants import VARIANTS, MapVariant, variant_arg

UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.05,
    "metric": "cosine",
    "n_components": 2,
    "random_state": 42,
}
# Claude Opus 5.5 standard rates (the namer calls the Messages API directly, not batches).
USD_PER_MTOK_IN, USD_PER_MTOK_OUT = 4.00, 20.00


def load_embeddings(variant: MapVariant, corpus: pd.DataFrame) -> np.ndarray:
    data = np.load(variant.embeddings)
    assert list(data["pageids"]) == list(corpus.pageid), "embeddings out of sync with corpus"
    return data["vectors"]


def get_coords(variant: MapVariant, pageids: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    params_key = json.dumps(UMAP_PARAMS, sort_keys=True)
    # The vectors too: a new eraser changes them while the pageids and params stay put.
    vectors_key = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()
    if variant.coords.exists():
        cached = np.load(variant.coords)
        if (
            list(cached["pageids"]) == list(pageids)
            and str(cached["params"]) == params_key
            and str(cached.get("vectors_sha256", "")) == vectors_key
        ):
            return cached["coords"]
    # random_state forces n_jobs=1 (UMAP warns once); a fixed layout is worth the wait.
    print(f"running UMAP {UMAP_PARAMS}")
    coords = umap.UMAP(**UMAP_PARAMS).fit_transform(vectors).astype(np.float32)
    write_npz_atomic(
        variant.coords,
        pageids=pageids,
        coords=coords,
        params=params_key,
        vectors_sha256=vectors_key,
    )
    return coords


def make_clusterer(min_cluster_size: int) -> ToponymyClusterer:
    return ToponymyClusterer(min_clusters=4, base_min_cluster_size=min_cluster_size)


def make_embedder(variant: MapVariant):
    embedder = OpenRouterEmbedder(instruction=variant.instruction)
    if variant.erase_tradition:
        return ErasedEmbedder(embedder, load_eraser(variant))
    return embedder


def explore(coords: np.ndarray, vectors: np.ndarray, sizes: list[int]) -> None:
    for size in sizes:
        clusterer = make_clusterer(size)
        clusterer.fit(clusterable_vectors=coords, embedding_vectors=vectors)
        counts = []
        for layer in clusterer.cluster_layers_:  # finest first
            labels = layer.cluster_labels
            counts.append(f"{labels.max() + 1} ({(labels == -1).mean():.0%} unlabelled)")
        print(
            f"base_min_cluster_size={size}: {len(counts)} layers, finest→coarsest: "
            + " | ".join(counts)
        )


def fit_and_name(
    variant: MapVariant, corpus, coords, vectors, min_cluster_size: int, detail: tuple
):
    namer = AsyncClaudeNamer()
    namer.extra_prompting = variant.naming_note
    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=make_embedder(variant),
        clusterer=make_clusterer(min_cluster_size),
        object_description=variant.object_description,
        corpus_description=variant.corpus_description,
        # Spread across layers and rounded onto Toponymy's seven name-length tiers;
        # 0.5-1.0 asks for 3-6 words at the finest layer and 1-2 at the coarsest (0.4-0.8
        # gave 7-word finest names and 3-4-line continent labels on the first map).
        lowest_detail_level=detail[0],
        highest_detail_level=detail[1],
    )
    texts = [document_text(r) for r in corpus.itertuples()]
    # Toponymy.fit takes (objects, high-D embeddings, low-D map): the reverse of
    # the clusterer's argument order, so pass by keyword.
    topic_model.fit(texts, embedding_vectors=vectors, clusterable_vectors=coords)

    names = topic_model.topic_names_
    assert len(names[0]) >= len(names[-1]), "expected layer 0 to be the finest"
    out = pd.DataFrame({"pageid": corpus.pageid, "x": coords[:, 0], "y": coords[:, 1]})
    for i, vec in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = vec
    write_parquet_safely(out, variant.labels)
    write_bytes_atomic(json.dumps(names, indent=2).encode(), variant.topic_names)

    for i, layer_names in enumerate(names):
        share = (out[f"label_layer_{i}"] == "Unlabelled").mean()
        words = np.median([len(n.split()) for n in layer_names]) if layer_names else 0
        print(
            f"layer {i}: {len(layer_names)} names, median {words:.0f} words, {share:.0%} unlabelled"
        )
    u = namer.usage
    cost = (u["input_tokens"] * USD_PER_MTOK_IN + u["output_tokens"] * USD_PER_MTOK_OUT) / 1e6
    print(
        f"naming: {u['calls']} calls ({u['cached']} more from cache), "
        f"{u['input_tokens']} in / {u['output_tokens']} out tokens ≈ ${cost:.2f}"
    )
    print(f"wrote {variant.labels} and {variant.topic_names}")


def main():
    parser = argparse.ArgumentParser()
    variant_arg(parser)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 12, 20])
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--detail", type=float, nargs=2, default=[0.5, 1.0])
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    corpus = load_corpus()
    vectors = load_embeddings(variant, corpus)
    coords = get_coords(variant, corpus.pageid.to_numpy(), vectors)

    if args.explore:
        explore(coords, vectors, args.sizes)
    elif args.min_cluster_size is not None:
        fit_and_name(variant, corpus, coords, vectors, args.min_cluster_size, tuple(args.detail))
    else:
        parser.error("pass --explore or --min-cluster-size N")


if __name__ == "__main__":
    main()
