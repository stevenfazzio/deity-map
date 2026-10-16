"""Experiment: erase tradition from article embeddings, and see what the map becomes.

The published map is organised by pantheon: a deity's 15 nearest neighbours on the
map share its tradition 79% of the time (chance 6%). This asks whether some
combination of embedded text and linear erasure gives a map organised by what gods
*do* (domain) instead, for a second, "by archetype" map.

Embedded texts (bases):
  lead             name + lead section, as published
  lead_instructed  the same text with a Qwen3-Embedding instruction (ROLE_INSTRUCTION)
                   asking for the deity's roles and domains
  summary          the one-line summaries enrich.py wrote. Written in the same call
                   that chose the domain label, so domain scores here are partly circular

Erasures (fit on the 38 tradition categories the Tradition colormap uses):
  raw           none
  centroid      project out the span of the class-mean differences (rank <= 37)
  recenter      subtract each article's own class mean. Label-dependent, so it cannot
                be applied to Toponymy's keyphrases at naming time; a reference only
  leace         LeaceEraser on one-hot tradition: no linear probe beats a constant
  leace_region  LeaceEraser on the 12 broad regions only: a gentler erasure

Scores:
  probe          held-out multinomial logistic accuracy for tradition, eraser fit on
                 the training split only (fitting on everything makes held-out
                 results look anti-predictive; see living-people-map stage 05b)
  hd_/2d_<field> share of each article's 15 nearest neighbours with the same value,
                 in full dimension (cosine) and on the UMAP layout a viewer sees
  cross          share of 2-d neighbour pairs from different traditions
  cross_domain   among those cross-tradition pairs, the share with the same main
                 domain (pairs with an unknown domain excluded)

Embedding models (--model):
  qwen3-0.6b  Qwen3-Embedding-0.6B run locally (the published map's model)
  qwen3-4b    Qwen3-Embedding-4B through OpenRouter (~$0.04 for everything here).
  qwen3-8b    Qwen3-Embedding-8B through OpenRouter (~$0.02; 4096 dimensions).
              Also runs lead_instructed@1024: the same vectors truncated to their first
              1024 dimensions (Matryoshka) and renormalised, which separates "better
              model" from "LEACE's covariance estimated in 2.5x the dimensions"

    uv run python experiments/erase_tradition.py --model qwen3-0.6b
    uv run python experiments/erase_tradition.py --model qwen3-4b

Writes data/experiments/<model>/coords_<base>_<erasure>.npz, erasure_scores.csv and
erasure_comparison.html. Embeddings come from embedder.py's cache after the first run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import umap
from concept_erasure import LeaceEraser
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster import UMAP_PARAMS
from common import DATA, write_bytes_atomic, write_npz_atomic
from embed import document_text, load_corpus
from embedder import LocalEmbedder, OpenRouterEmbedder
from render import color_mapping
from taxonomy import DOMAIN_LABELS, GENDER_LABELS, KIND_LABELS, resolve_traditions
from variants import ROLE_INSTRUCTION

MODELS = {
    "qwen3-0.6b": LocalEmbedder,
    "qwen3-4b": lambda **kw: OpenRouterEmbedder(
        model="qwen/qwen3-embedding-4b", provider="DeepInfra", **kw
    ),
    # DeepInfra, as for the 4B; SiliconFlow also serves it, but fp8-quantised.
    "qwen3-8b": lambda **kw: OpenRouterEmbedder(
        model="qwen/qwen3-embedding-8b", provider="DeepInfra", **kw
    ),
}
K = 15
SEED = 42
ALL_ERASURES = ["raw", "centroid", "recenter", "leace", "leace_region"]
MAIN = ["raw", "centroid", "leace"]
RUNS = (
    [("lead", e) for e in ALL_ERASURES]
    + [("lead_instructed", e) for e in MAIN]
    + [("summary", e) for e in MAIN]
    + [("lead_instructed@1024", e) for e in MAIN]  # only when the model is wider
)
SHOWN = [
    ("lead", "raw"),
    ("lead", "leace"),
    ("lead_instructed", "raw"),
    ("lead_instructed", "centroid"),
    ("lead_instructed", "leace"),
    ("summary", "leace"),
]


def one_hot(y: np.ndarray, k: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.from_numpy(y), k).float()


def fit_centroid(x: np.ndarray, y: np.ndarray, k: int):
    mu = x.mean(axis=0)
    diffs = np.stack([x[y == c].mean(axis=0) - mu for c in range(k) if (y == c).any()])
    _, s, vt = np.linalg.svd(diffs, full_matrices=False)
    basis = vt[s > 1e-6 * s.max()]  # orthonormal rows spanning the mean differences
    return lambda z, _labels=None: z - (z @ basis.T) @ basis


def fit_recenter(x: np.ndarray, y: np.ndarray, k: int):
    mu = x.mean(axis=0)
    means = np.stack([x[y == c].mean(axis=0) if (y == c).any() else mu for c in range(k)])
    return lambda z, labels: z - means[labels] + mu


def fit_leace(x: np.ndarray, y: np.ndarray, k: int):
    eraser = LeaceEraser.fit(torch.from_numpy(x), one_hot(y, k))
    return lambda z, _labels=None: eraser(torch.from_numpy(z)).numpy().astype(np.float32)


ERASURES = {
    "raw": None,
    "centroid": (fit_centroid, "tradition"),
    "recenter": (fit_recenter, "tradition"),
    "leace": (fit_leace, "tradition"),
    "leace_region": (fit_leace, "region"),
}


def codes(values: pd.Series) -> tuple[np.ndarray, int]:
    cat = pd.Categorical(values)
    return cat.codes.astype(np.int64), len(cat.categories)


corpus = load_corpus()
trad_label, trad_cat, region = resolve_traditions(corpus)
fields = {
    "tradition": trad_cat.to_numpy(),
    "region": region.to_numpy(),
    "domain": corpus.primary_domain.map(DOMAIN_LABELS).to_numpy(),
    "kind": corpus.kind.map(KIND_LABELS).to_numpy(),
    "gender": corpus.gender.map(GENDER_LABELS).to_numpy(),
}
labels = {"tradition": codes(trad_cat), "region": codes(region)}


def truncate(x: np.ndarray, dims: int) -> np.ndarray:
    """Matryoshka truncation: Qwen3-Embedding front-loads information, so the first
    `dims` coordinates, renormalised, are a usable smaller embedding."""
    t = x[:, :dims]
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def load_bases(embedder_cls) -> dict[str, np.ndarray]:
    texts = [document_text(r) for r in corpus.itertuples()]
    plain, instructed = embedder_cls(), embedder_cls(instruction=ROLE_INSTRUCTION)
    bases = {
        "lead": plain.encode(texts, show_progress_bar=True),
        "lead_instructed": instructed.encode(texts, show_progress_bar=True),
        "summary": plain.encode(corpus.summary.tolist(), show_progress_bar=True),
    }
    bases = {k: v.astype(np.float32) for k, v in bases.items()}
    if bases["lead_instructed"].shape[1] > 1024:
        bases["lead_instructed@1024"] = truncate(bases["lead_instructed"], 1024)
    for e in (plain, instructed):
        if hasattr(e, "cost_usd"):
            print(f"  {e.model_name}: ${e.cost_usd:.4f} spent this run")
    # How far the instruction moves each article: a direct read on how hard it steers.
    cos = (bases["lead"] * bases["lead_instructed"]).sum(axis=1)
    print(f"instruction strength: mean cosine(plain, instructed) = {cos.mean():.3f}")
    return bases


def erase(x: np.ndarray, erasure: str) -> np.ndarray:
    if ERASURES[erasure] is None:
        return x
    fit, target = ERASURES[erasure]
    y, k = labels[target]
    return fit(x, y, k)(x, y)


def probe_accuracy(x: np.ndarray, erasure: str) -> float:
    """Split first, fit the eraser on train only, then probe tradition."""
    y_trad = labels["tradition"][0]
    idx_tr, idx_te = train_test_split(
        np.arange(len(x)), test_size=0.2, random_state=SEED, stratify=y_trad
    )
    x_tr, x_te = x[idx_tr], x[idx_te]
    if ERASURES[erasure] is not None:
        fit, target = ERASURES[erasure]
        y_fit, k = labels[target]
        transform = fit(x_tr, y_fit[idx_tr], k)
        x_tr = transform(x_tr, y_fit[idx_tr])
        x_te = transform(x_te, y_fit[idx_te])
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(x_tr, y_trad[idx_tr])
    return float((clf.predict(x_te) == y_trad[idx_te]).mean())


def neighbours(points: np.ndarray, cosine: bool) -> np.ndarray:
    if cosine:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(points).kneighbors(points, return_distance=False)
    return nn[:, 1:]


def purity(nn: np.ndarray) -> dict[str, float]:
    return {name: float((v[nn] == v[:, None]).mean()) for name, v in fields.items()}


def cross_tradition(nn: np.ndarray) -> tuple[float, float]:
    trad, dom = fields["tradition"], corpus.primary_domain.to_numpy()
    i, j = np.repeat(np.arange(len(nn)), K), nn.ravel()
    cross = trad[i] != trad[j]
    known = (dom[i] != "unknown") & (dom[j] != "unknown")
    return float(cross.mean()), float((dom[i] == dom[j])[cross & known].mean())


def chance(v: np.ndarray) -> float:
    return float((pd.Series(v).value_counts(normalize=True) ** 2).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen3-0.6b")
    args = parser.parse_args()
    global OUT
    OUT = DATA / "experiments" / args.model
    OUT.mkdir(parents=True, exist_ok=True)
    base_chance = {f: chance(v) for f, v in fields.items()}
    dom = corpus.primary_domain
    majority = pd.Series(labels["tradition"][0]).value_counts(normalize=True).iloc[0]
    print("chance purity: " + ", ".join(f"{f} {c:.2f}" for f, c in base_chance.items()))
    print(
        f"tradition probe baseline {majority:.2f}; cross-domain chance "
        f"{chance(dom[dom != 'unknown'].to_numpy()):.2f}\n"
    )

    bases = load_bases(MODELS[args.model])
    results, coords = {}, {}
    for base, erasure in RUNS:
        if base not in bases:
            continue
        name = f"{base}/{erasure}"
        x = erase(bases[base], erasure)
        path = OUT / f"coords_{base}_{erasure}.npz"
        if path.exists() and list(np.load(path)["pageids"]) == list(corpus.pageid):
            xy = np.load(path)["coords"]
        else:
            xy = umap.UMAP(**UMAP_PARAMS).fit_transform(x).astype(np.float32)
            write_npz_atomic(path, pageids=corpus.pageid.to_numpy(), coords=xy)
        nn2 = neighbours(xy, cosine=False)
        hd, flat = purity(neighbours(x, cosine=True)), purity(nn2)
        cross, cross_domain = cross_tradition(nn2)
        results[name] = {
            "probe": probe_accuracy(bases[base], erasure),
            **{f"2d_{f}": v for f, v in flat.items()},
            "cross": cross,
            "cross_domain": cross_domain,
            **{f"hd_{f}": v for f, v in hd.items()},
        }
        coords[name] = xy
        r = results[name]
        print(
            f"{name:26s} probe {r['probe']:.2f} | 2d tradition {r['2d_tradition']:.2f} "
            f"domain {r['2d_domain']:.2f} kind {r['2d_kind']:.2f} gender {r['2d_gender']:.2f} "
            f"| cross {cross:.0%}, same domain {cross_domain:.2f}",
            flush=True,
        )

    table = pd.DataFrame(results).T
    table.to_csv(OUT / "erasure_scores.csv")
    page = comparison_page({f"{b}/{e}": coords[f"{b}/{e}"] for b, e in SHOWN}, table, base_chance)
    write_bytes_atomic(page.encode("utf-8"), OUT / "erasure_comparison.html")
    print(f"\nwrote {OUT / 'erasure_scores.csv'} and erasure_comparison.html")


def comparison_page(coords: dict, table: pd.DataFrame, base_chance: dict) -> str:
    """Small multiples: layouts coloured by domain and by broad region."""
    hover = (
        (corpus.name + "<br>" + pd.Series(trad_label) + " · " + pd.Series(fields["domain"]))
        + "<br>"
        + corpus.summary
    )
    parts = []
    for field in ["domain", "region"]:
        values = pd.Series(fields[field])
        mapping = color_mapping(values)
        names = list(coords)
        titles = [
            f"<b>{n}</b><br>{field} {table.loc[n, '2d_' + field]:.2f} · "
            f"tradition {table.loc[n, '2d_tradition']:.2f} · "
            f"cross-domain {table.loc[n, 'cross_domain']:.2f}"
            for n in names
        ]
        fig = make_subplots(
            rows=1, cols=len(names), subplot_titles=titles, horizontal_spacing=0.015
        )
        for col, n in enumerate(names, start=1):
            xy = coords[n]
            for cat in values.value_counts().index:
                m = (values == cat).to_numpy()
                fig.add_trace(
                    go.Scattergl(
                        x=xy[m, 0],
                        y=xy[m, 1],
                        mode="markers",
                        name=cat,
                        legendgroup=cat,
                        showlegend=col == 1,
                        marker={"size": 3, "color": mapping[cat]},
                        text=hover[m],
                        hoverinfo="text",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_annotations(font_size=11)
        fig.update_layout(
            height=440,
            width=2600,
            margin={"t": 80, "b": 10, "l": 10, "r": 10},
            title=f"Coloured by {field} (chance purity {base_chance[field]:.2f})",
            plot_bgcolor="white",
            legend={"itemsizing": "constant", "font": {"size": 10}},
        )
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if not parts else False))
    head = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Tradition erasure</title></head><body>"
    )
    return head + "\n".join(parts) + "</body></html>"


if __name__ == "__main__":
    main()
