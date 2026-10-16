"""Experiment: fit the tradition eraser on Wikipedia's categories, and at other grains.

experiments/erase_tradition.py chose LEACE on enrich.py's 38 tradition categories, and
scored it with the same labels it was fit on. This fits the eraser on labels from
category_labels.py instead, and scores every eraser against both raters, so neither
rater grades its own work. It also tries coarser and finer partitions of tradition,
built from the same categories.

All on the archetype map's vectors (Qwen3-Embedding-8B with ROLE_INSTRUCTION), from
embedder.py's cache. Erasers:
  raw         none
  llm38       enrich.py's tradition, one-hot over the colormap's 38 categories (published)
  cat38       category_labels.py's traditions, multi-hot over the same 38
  cat_region  the same, coarsened to taxonomy.REGION_OF's 12 regions
  cat_fine    cat38 plus the finer cultures in FINE_PATTERNS (Hittite, Hurrian and
              Luwian within Anatolian; Aztec and Maya; Yoruba, Akan ...)
Articles with no category tradition (19) are left out of the category erasers' fit,
but still erased.

Probes: held-out multinomial logistic accuracy, with the eraser fit on the training
split only, against
  llm_tradition  enrich.py's 38 categories
  cat_tradition  category_labels.py's 38, on articles with exactly one
  fine           the FINE_PATTERNS cultures, on articles with exactly one
  llm_domain     enrich.py's primary domain, unknown and other excluded
  cat_domain     category_labels.py's domains, on articles with exactly one
Each baseline is the training split's majority class.

Neighbours: with the eraser fit on every article, as the map does, the share of each
article's 15 nearest neighbours (cosine in full dimension, and on the UMAP layout)
with the same enrich.py tradition, the same fine culture, the same enrich.py domain,
and an overlapping set of category domains. Pairs where either side lacks the label
are left out; chance is the same share over random pairs.

    uv run python experiments/erase_by_categories.py

Writes data/experiments/categories/erasure_scores.csv, caches the UMAP layouts there, and
writes comparison.html: the cat38 and cat_fine layouts side by side under each colormap.

Run on 2026-10-16 with the 38-category taxonomy, which the names here (llm38, cat38)
keep; taxonomy.TRADITIONS has since become 33 + Other (see taxonomy_audit.py), and a
re-run uses that.
"""

import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import umap
from concept_erasure import LeaceEraser
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from category_labels import structured_labels
from cluster import UMAP_PARAMS
from common import DATA, write_npz_atomic
from embed import document_text, load_corpus
from embedder import OpenRouterEmbedder
from render import NEUTRAL_COLORS, categorical_palette
from taxonomy import (
    DOMAIN_LABELS,
    OTHER_TRADITIONS,
    REGION_OF,
    TRADITION_LABELS,
    resolve_traditions,
)
from variants import ROLE_INSTRUCTION

OUT = DATA / "experiments" / "categories"
K = 15
SEED = 42
N_JOBS = 6
MIN_FINE = 5  # a finer culture needs this many articles to be its own class
SCHEMES = ["raw", "llm38", "cat38", "cat_region", "cat_fine"]
SHOWN = {
    "cat38": "38 traditions erased",
    "cat_fine": "38 traditions + finer cultures erased",
}
# Non-answers on the finer-culture colormap, in greys of their own (see render.py)
PAGE_NEUTRALS = NEUTRAL_COLORS | {"No finer culture": "#e6e6e6", "Several": "#9e9e9e"}

# Finer cultures, matched in an article's own categories only. Ethnolinguistic or
# sectarian splits, never named groups of gods: Nereids, Muses or Adityas are roles
# within a tradition, and erasing them would erase role.
FINE_PATTERNS = [
    (r"\bHittite\b", "Hittite"),
    (r"\bHurrian\b", "Hurrian"),
    (r"\bLuwian\b", "Luwian"),
    (r"\bHattian\b", "Hattian"),
    (r"\bPhrygian\b", "Phrygian"),
    (r"\bUrartian\b", "Urartian"),
    (r"\bElamite\b", "Elamite"),
    (r"\bUgaritic\b", "Ugaritic"),
    (r"\bPhoenician\b|\bPunic\b|\bCarthaginian\b", "Phoenician"),
    (r"\bCanaanite\b", "Canaanite"),
    (r"\bEtruscan\b", "Etruscan"),
    (r"\bSabine\b", "Sabine"),
    (r"\bGallo-Roman\b", "Gallo-Roman"),
    (r"\bGaulish\b|\bGallic\b", "Gaulish"),
    (r"\bIrish\b", "Irish"),
    (r"\bWelsh\b|\bBrittonic\b", "Brittonic"),
    (r"\bNorse\b|\bÆsir\b|\bVanir\b", "Norse"),
    (r"\bAnglo-Saxon\b", "Anglo-Saxon"),
    (r"\bFinnish\b", "Finnish"),
    (r"\bS[áa]a?mi\b", "Sámi"),
    (r"\bEstonian\b", "Estonian"),
    (r"\bLithuanian\b", "Lithuanian"),
    (r"\bLatvian\b", "Latvian"),
    (r"\bAlbanian\b", "Albanian"),
    (r"\bBasque\b", "Basque"),
    (r"\bLusitanian\b", "Lusitanian"),
    (r"\bArmenian\b", "Armenian"),
    (r"\bThracian\b", "Thracian"),
    (r"\bGeorgian\b", "Georgian"),
    (r"\bOssetian\b", "Ossetian"),
    (r"\bDacian\b", "Dacian"),
    (r"\bIllyrian\b", "Illyrian"),
    (r"\bTurkic\b", "Turkic"),
    (r"\bSiberian\b", "Siberian"),
    (r"\bMongol", "Mongolic"),
    (r"\bYakut\b|\bSakha\b", "Yakut"),
    (r"\bAztec\b|\bNahua\b", "Aztec"),
    (r"\bMaya\b|\bMayan\b", "Maya"),
    (r"\bInca\b|\bIncan\b|\bQuechua\b", "Inca"),
    (r"\bMuisca\b", "Muisca"),
    (r"\bMapuche\b", "Mapuche"),
    (r"\bGuaran[íi]\b", "Guaraní"),
    (r"\bAymara\b", "Aymara"),
    (r"\bInuit\b", "Inuit"),
    (r"\bHaudenosaunee\b|\bIroquois\b", "Haudenosaunee"),
    (r"\bAlgonquian\b|\bAnishinaabe\b|\bOjibwe\b|\bCree\b", "Algonquian"),
    (r"\bHopi\b|\bZuni\b|\bPueblo\b|\bKachinas?\b|\bKatsina", "Pueblo"),
    (r"\bHaida\b|\bTlingit\b|\bKwakwaka'wakw\b|\bSalish\b", "Northwest Coast"),
    (r"\bM[āa]ori\b", "Māori"),
    (r"\bHawaiian\b", "Hawaiian"),
    (r"\bFijian\b", "Fijian"),
    (r"\bMicronesian\b|\bKiribati\b|\bChamorro\b|\bPalauan\b", "Micronesian"),
    (r"\bRapa Nui\b|\bEaster Island\b", "Rapa Nui"),
    (r"\bSamoan\b", "Samoan"),
    (r"\bTongan\b", "Tongan"),
    (r"\bMangaia|\bCook Islands?\b", "Cook Islands"),
    (r"\bPapua|\bMelanesian\b|\bSolomon Islands\b|\bVanuatu\b", "Melanesian"),
    (r"\bYoruba\b|\bOrishas?\b", "Yoruba"),
    (r"\bBantu\b|\bZulu\b|\bXhosa\b|\bShona\b|\bKongo\b|\bGanda\b|\bLuba\b|\bKikuyu\b", "Bantu"),
    (r"\bBerber\b|\bAmazigh\b", "Berber"),
    (r"\bIgbo\b|\bOdinani\b", "Igbo"),
    (r"\bAkan\b|\bAshanti\b", "Akan"),
    (r"\bFon\b|\bDahomey\b|\bVodun\b|\bEwe\b", "Fon and Ewe"),
    (r"\bSerer\b", "Serer"),
    (r"\bHaitian\b|\bVodou\b|\bVoodoo\b", "Vodou"),
    (r"\bSanter[íi]a\b|\bLucum[íi]\b|\bAfro-Cuban\b", "Santería"),
    (r"\bCandombl[ée]\b|\bUmbanda\b|\bAfro-Brazilian\b|\bQuimbanda\b", "Afro-Brazilian"),
    (r"\bRigved|\bVedic\b", "Vedic"),
    (r"\bTamil\b", "Tamil"),
    (r"\bHindu folk\b|\bRegional Hindu\b", "Hindu folk"),
    (r"\bShakt|\bMahavidya|\bMatrikas\b|\btantric\b", "Shakta"),
    (r"\bShaiv|\bForms of Shiva\b", "Shaiva"),
    (r"\bVaishnav|\bForms of Vishnu\b|\bAvatars of Vishnu\b", "Vaishnava"),
    (r"\bTibetan\b|\bVajrayana\b", "Vajrayana"),
    (r"\bSinhalese\b|\bTheravada\b", "Theravada"),
    (r"\bTaois|\bDaois", "Taoist"),
    (r"\bChinese folk\b", "Chinese folk"),
    (r"\bShinto\b|\b[Kk]ami\b", "Shinto"),
    (r"\bRyukyuan\b|\bOkinawan\b", "Ryukyuan"),
    (r"\bBurmese\b|\bMyanmar\b|\b[Nn]ats\b", "Burmese"),
    (r"\bThai\b", "Thai"),
    (r"\bVietnamese\b", "Vietnamese"),
    (r"\bIndonesian\b|\bJavanese\b|\bBalinese\b|\bBatak\b|\bDayak\b", "Indonesian"),
    (r"\bKhmer\b|\bCambodian\b", "Khmer"),
    (r"\bPhilippine\b|\bFilipino\b|\bTagalog\b|\bVisayan\b|\bBikol\b|\bIfugao\b", "Philippine"),
    (r"\bGnostic", "Gnostic"),
    (r"\bJewish\b|\bJudaism\b|\bKabbal", "Jewish"),
    (r"\bIslam|\bMuslim\b", "Islamic"),
    (r"\bChristian", "Christian"),
]


def multi_hot(lists, vocab: list[str]) -> np.ndarray:
    index = {v: i for i, v in enumerate(vocab)}
    m = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    for row, labels in enumerate(lists):
        for label in labels:
            if label in index:
                m[row, index[label]] = 1
    return m


def single_codes(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(code, mask) for rows with exactly one label."""
    mask = m.sum(axis=1) == 1
    return np.where(mask, m.argmax(axis=1), -1), mask


def build_labels(corpus: pd.DataFrame) -> dict:
    s = structured_labels(corpus)
    vocab38 = sorted(set(resolve_traditions(corpus)[1]))
    cat38 = multi_hot(s.traditions, vocab38)

    region_of = {TRADITION_LABELS[k]: r for k, r in REGION_OF.items()}
    region_of |= {name: region for _, name, region in OTHER_TRADITIONS}
    regions = sorted(set(region_of.values()))
    cat_region = multi_hot([[region_of[t] for t in ts] for ts in s.traditions], regions)

    cats = pd.read_parquet(DATA / "categories.parquet")
    own = cats.groupby("pageid").category.apply(list).reindex(corpus.pageid).tolist()
    compiled = [(re.compile(p), name) for p, name in FINE_PATTERNS]
    fine_lists = [sorted({n for rx, n in compiled for c in cs if rx.search(c)}) for cs in own]
    counts = pd.Series([f for fl in fine_lists for f in fl]).value_counts()
    fine_vocab = sorted(counts[counts >= MIN_FINE].index)
    fine = multi_hot(fine_lists, fine_vocab)
    kept = [[f for f in fl if f in fine_vocab] for fl in fine_lists]
    fine_display = [
        "No finer culture" if not k else k[0] if len(k) == 1 else f"Several ({', '.join(k)})"
        for k in kept
    ]

    dom_vocab = sorted(set(DOMAIN_LABELS) - {"unknown", "other"})
    cat_dom = multi_hot(s.domains, dom_vocab)
    llm_trad = pd.Categorical(resolve_traditions(corpus)[1], categories=vocab38).codes
    llm_dom = pd.Categorical(corpus.primary_domain, categories=dom_vocab).codes  # -1 if unknown

    print(
        f"labels: 38 traditions; {len(regions)} regions; {len(fine_vocab)} finer cultures "
        f"(>= {MIN_FINE} articles) on {(fine.sum(1) > 0).sum()} articles; "
        f"{(cat38.sum(1) == 0).sum()} articles without a category tradition"
    )
    return {
        "fit": {
            "llm38": np.eye(len(vocab38), dtype=np.float32)[llm_trad],
            "cat38": cat38,
            "cat_region": cat_region,
            "cat_fine": np.hstack([cat38, fine]),
        },
        "probe": {
            "llm_tradition": (llm_trad, np.ones(len(corpus), bool)),
            "cat_tradition": single_codes(cat38),
            "fine": single_codes(fine),
            "llm_domain": (llm_dom, llm_dom >= 0),
            "cat_domain": single_codes(cat_dom),
        },
        "neighbour": {
            "llm_tradition": (llm_trad, None),
            "fine": single_codes(fine),
            "llm_domain": (llm_dom, llm_dom >= 0),
        },
        "cat_dom": cat_dom,
        "fine_display": fine_display,
    }


def fit_eraser(x: np.ndarray, z: np.ndarray, rows: np.ndarray):
    rows = rows[z[rows].sum(axis=1) > 0]  # unlabelled articles stay out of the fit
    eraser = LeaceEraser.fit(torch.from_numpy(x[rows]), torch.from_numpy(z[rows]))

    def apply(v: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return eraser(torch.from_numpy(v)).numpy().astype(np.float32)

    return apply


def probe(x, y, mask, idx_tr, idx_te) -> tuple[float, float]:
    tr, te = idx_tr[mask[idx_tr]], idx_te[mask[idx_te]]
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(x[tr], y[tr])
    majority = np.bincount(y[tr]).argmax()
    return float((clf.predict(x[te]) == y[te]).mean()), float((y[te] == majority).mean())


def layout(x: np.ndarray, scheme: str, pageids: np.ndarray) -> np.ndarray:
    path = OUT / f"coords_{scheme}.npz"
    if path.exists() and np.array_equal(np.load(path)["pageids"], pageids):
        return np.load(path)["coords"]
    xy = umap.UMAP(**UMAP_PARAMS).fit_transform(x).astype(np.float32)
    write_npz_atomic(path, pageids=pageids, coords=xy)
    return xy


def neighbours(points: np.ndarray, cosine: bool) -> np.ndarray:
    if cosine:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(points).kneighbors(points, return_distance=False)
    return nn[:, 1:]


def shares(nn: np.ndarray, labels: dict) -> dict[str, float]:
    i, j = np.repeat(np.arange(len(nn)), nn.shape[1]), nn.ravel()
    out = {}
    for name, (v, mask) in labels["neighbour"].items():
        keep = np.ones(len(i), bool) if mask is None else mask[i] & mask[j]
        out[name] = float((v[i] == v[j])[keep].mean())
    d = labels["cat_dom"]
    keep = (d[i].sum(1) > 0) & (d[j].sum(1) > 0)
    out["cat_domain"] = float(((d[i] * d[j]).sum(1) > 0)[keep].mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    labels = build_labels(corpus)
    embedder = OpenRouterEmbedder(instruction=ROLE_INSTRUCTION)
    x = embedder.encode([document_text(r) for r in corpus.itertuples()]).astype(np.float32)
    print(f"vectors {x.shape}; ${embedder.cost_usd:.4f} spent on uncached text")

    rng = np.random.default_rng(SEED)
    rand = rng.integers(0, len(x), size=(len(x), K))
    chance = shares(rand, labels)
    print("chance: " + ", ".join(f"{k} {v:.3f}" for k, v in chance.items()))

    idx_tr, idx_te = train_test_split(
        np.arange(len(x)),
        test_size=0.2,
        random_state=SEED,
        stratify=labels["probe"]["llm_tradition"][0],
    )
    split, full = {"raw": x}, {"raw": x}
    for scheme in SCHEMES[1:]:
        z = labels["fit"][scheme]
        split[scheme] = fit_eraser(x, z, idx_tr)(x)
        full[scheme] = fit_eraser(x, z, np.arange(len(x)))(x)

    jobs = [
        delayed(probe)(split[s], y, mask, idx_tr, idx_te)
        for s in SCHEMES
        for y, mask in labels["probe"].values()
    ] + [delayed(layout)(full[s], s, corpus.pageid.to_numpy()) for s in SCHEMES]
    done = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    n_probe = len(labels["probe"])
    probes, coords = done[: len(SCHEMES) * n_probe], done[len(SCHEMES) * n_probe :]

    rows = {}
    for si, s in enumerate(SCHEMES):
        row = {}
        for pi, name in enumerate(labels["probe"]):
            acc, base = probes[si * n_probe + pi]
            row[f"probe_{name}"], row[f"base_{name}"] = acc, base
        row |= {f"hd_{k}": v for k, v in shares(neighbours(full[s], True), labels).items()}
        row |= {f"2d_{k}": v for k, v in shares(neighbours(coords[si], False), labels).items()}
        rows[s] = row
    table = pd.DataFrame(rows).T
    table.to_csv(OUT / "erasure_scores.csv")

    pd.set_option("display.width", 200)
    probe_cols = [c for c in table.columns if c.startswith("probe_")]
    print("\nprobes (held-out accuracy; baseline is the majority class):")
    print(table[probe_cols].rename(columns=lambda c: c[6:]).round(3).to_string())
    print(
        "baseline  "
        + "  ".join(
            f"{c[5:]} {table[c].iloc[0]:.3f}" for c in table.columns if c.startswith("base_")
        )
    )
    for prefix in ["hd_", "2d_"]:
        print(f"\n{prefix[:-1]} neighbour shares:")
        cols = [c for c in table.columns if c.startswith(prefix)]
        print(table[cols].rename(columns=lambda c: c[3:]).round(3).to_string())
    shown = {s: coords[SCHEMES.index(s)] for s in SHOWN}
    page = comparison_page(corpus, labels, shown, table, chance)
    (OUT / "comparison.html").write_text(page, encoding="utf-8")
    print(f"\nwrote {OUT / 'erasure_scores.csv'} and comparison.html")


def color_mapping(values: pd.Series) -> dict[str, str]:
    """render.color_mapping, with greys for this page's non-answers too."""
    order = values.value_counts().index.tolist()
    neutrals = {c: PAGE_NEUTRALS[c] for c in order if c in PAGE_NEUTRALS}
    named = [c for c in order if c not in neutrals]
    return dict(zip(named, categorical_palette(len(named), list(neutrals.values())))) | neutrals


def ring_traces(fig, color: str) -> None:
    """An empty ring in each panel, which the page's script moves onto points."""
    for col in (1, 2):
        fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="markers",
                marker={"size": 13, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": color}},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )


def layouts_figure(values: pd.Series, coords: dict, titles: list[str]):
    """Both layouts under one colormap; a legend entry hides or isolates it in both.

    Points carry only their article's index: the page's script shows the card, which is
    stored once rather than in each of the 8 panels, and rings the article in both.
    """
    mapping = color_mapping(values)
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.02)
    # Greys drawn first so coloured points sit on top; legendrank keeps frequency order.
    draw = [c for c in mapping if c in PAGE_NEUTRALS] + [
        c for c in mapping if c not in PAGE_NEUTRALS
    ]
    rank = {c: i for i, c in enumerate(values.value_counts().index)}
    for col, xy in enumerate(coords.values(), start=1):
        for cat in draw:
            m = (values == cat).to_numpy()
            fig.add_trace(
                go.Scattergl(
                    x=xy[m, 0],
                    y=xy[m, 1],
                    mode="markers",
                    name=f"{cat} ({m.sum()})",
                    legendgroup=cat,
                    legendrank=rank[cat],
                    showlegend=col == 1,
                    marker={"size": 4, "color": mapping[cat]},
                    customdata=np.flatnonzero(m).astype(np.int32),
                    hoverinfo="none",  # still fires plotly_hover, for the page's card
                ),
                row=1,
                col=col,
            )
    ring_traces(fig, "#d62728")  # the hovered article, in both panels
    ring_traces(fig, "#111111")  # search matches
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_yaxes(scaleanchor="x", row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", row=1, col=2)
    fig.update_annotations(font_size=13)
    fig.update_layout(
        height=760,
        margin={"t": 70, "b": 10, "l": 10, "r": 10},
        plot_bgcolor="white",
        legend={"itemsizing": "constant", "font": {"size": 11}},
    )
    return fig


def comparison_page(
    corpus: pd.DataFrame, labels: dict, coords: dict, table: pd.DataFrame, chance: dict
) -> str:
    _, tradition, region = resolve_traditions(corpus)
    colormaps = {
        "Finer culture": pd.Series(labels["fine_display"]),
        "Main domain (enrich.py)": corpus.primary_domain.map(DOMAIN_LABELS).reset_index(drop=True),
        "Tradition (the published colormap)": tradition.reset_index(drop=True),
        "Broad region": region.reset_index(drop=True),
    }
    # The card and search show which cultures "Several" means; the colormap lumps them.
    fields = pd.DataFrame({k: v.to_numpy() for k, v in colormaps.items()})
    colormaps["Finer culture"] = colormaps["Finer culture"].str.replace(r" \(.*\)$", "", regex=True)
    # 54 enrich.py summaries end in a literal "</br>", some after stray "”}"
    summaries = corpus.summary.fillna("").str.replace(r"[\s”»}]*</br>\s*$", "", regex=True)
    cards = [
        f"<b>{html.escape(n)}</b><table>"
        + "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in row.items()
        )
        + f"</table><p>{html.escape(sm)}</p>"
        for n, (_, row), sm in zip(corpus.name, fields.iterrows(), summaries)
    ]
    titles = [f"<b>{SHOWN[s]}</b>" for s in coords]
    metrics = ["llm_tradition", "fine", "cat_domain"]
    score_rows = [("Chance", [chance[m] for m in metrics])] + [
        (SHOWN[s], [table.loc[s, f"2d_{m}"] for m in metrics]) for s in coords
    ]
    scores = "".join(
        f"<tr><td>{html.escape(name)}</td>" + "".join(f"<td>{v:.3f}</td>" for v in vals) + "</tr>"
        for name, vals in score_rows
    )
    parts = []
    for i, (name, values) in enumerate(colormaps.items()):
        div = layouts_figure(values, coords, titles).to_html(
            full_html=False,
            include_plotlyjs="cdn" if i == 0 else False,
            div_id=f"fig{i}",
            config={"responsive": True},
        )
        parts.append(f"<h2>Coloured by {html.escape(name)}</h2>{div}")

    data = {
        "cards": cards,
        "search": [
            " | ".join([n, *map(str, row)]).lower()
            for n, (_, row) in zip(corpus.name, fields.iterrows())
        ],
        "xy": [np.round(xy, 3).tolist() for xy in coords.values()],
        "figs": [f"fig{i}" for i in range(len(colormaps))],
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    n_fine = labels["fit"]["cat_fine"].shape[1] - labels["fit"]["cat38"].shape[1]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erasure grain comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px 24px; color: #222; background: #fff; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }} h2 {{ font-size: 16px; margin: 28px 0 0; }}
  p {{ max-width: 980px; line-height: 1.45; margin: 6px 0; }}
  .bar {{ position: sticky; top: 0; z-index: 10; background: #fff; padding: 8px 0;
    border-bottom: 1px solid #e5e5e5; display: flex; gap: 12px; align-items: center; }}
  input {{ font: inherit; padding: 6px 10px; width: 320px; border: 1px solid #bbb; border-radius: 6px; }}
  #count {{ color: #666; font-size: 14px; }}
  table.scores {{ border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
  table.scores th, table.scores td {{ padding: 3px 12px; text-align: right; border-bottom: 1px solid #eee; }}
  table.scores th:first-child, table.scores td:first-child {{ text-align: left; }}
  #tip {{ position: fixed; display: none; z-index: 20; max-width: 340px; pointer-events: none;
    background: #fff; border: 1px solid #ccc; border-radius: 6px; padding: 8px 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,.15); font-size: 12.5px; line-height: 1.35; }}
  #tip table {{ border-collapse: collapse; margin: 4px 0; }}
  #tip td {{ padding: 1px 0; vertical-align: top; }}
  #tip td:first-child {{ color: #777; padding-right: 10px; white-space: nowrap; }}
  #tip p {{ margin: 6px 0 0; padding-top: 6px; border-top: 1px solid #eee; }}
</style></head><body>
<h1>The role map with tradition erased at two grains</h1>
<p>Both panels are UMAP layouts of the same {len(corpus)} role-instructed embeddings. On the
left, LEACE erases the 38 traditions from Wikipedia's categories. On the right, it also erases
{n_fine} finer cultures within them (Hittite, Hurrian and Luwian within Anatolian; Aztec and
Maya; Yoruba, Akan and Igbo...).</p>
<table class="scores"><tr><th>Share of each point's 15 nearest neighbours on the layout</th>
<th>same tradition</th><th>same finer culture</th><th>overlapping category domain</th></tr>
{scores}</table>
<p>Hover a point to ring it in red in both panels. Click a legend entry to hide it in both
panels, double-click to show it alone. Search matches names, traditions, cultures and
domains, and rings the matches in black in every plot.</p>
<div class="bar"><input id="q" placeholder="Search: Thor, Hittite, war, Yoruba..." autocomplete="off">
<span id="count"></span></div>
{"".join(parts)}
<div id="tip"></div>
<script>
const D = {blob};
const tip = document.getElementById("tip");
const count = document.getElementById("count");
// Each figure ends with two hover rings, then two search rings (one per panel).
function rings(gd, first, idx) {{
  Plotly.restyle(gd, {{
    x: D.xy.map((l) => idx.map((i) => l[i][0])),
    y: D.xy.map((l) => idx.map((i) => l[i][1])),
  }}, [first, first + 1]);
}}
for (const id of D.figs) {{
  const gd = document.getElementById(id);
  gd.on("plotly_hover", (ev) => {{
    const i = ev.points[0].customdata;
    if (i === undefined) return;
    tip.innerHTML = D.cards[i];
    tip.style.display = "block";
    const e = ev.event || {{ clientX: 24, clientY: 120 }};
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(e.clientX + 16, innerWidth - w - 8) + "px";
    tip.style.top = (e.clientY + 16 + h > innerHeight ? e.clientY - h - 12 : e.clientY + 16) + "px";
    rings(gd, gd.data.length - 4, [i]);
  }});
  gd.on("plotly_unhover", () => {{
    tip.style.display = "none";
    rings(gd, gd.data.length - 4, []);
  }});
}}
document.getElementById("q").addEventListener("input", (e) => {{
  const q = e.target.value.trim().toLowerCase();
  const hit = q.length < 2 ? [] : D.search.flatMap((s, i) => (s.includes(q) ? [i] : []));
  for (const id of D.figs) {{
    const gd = document.getElementById(id);
    rings(gd, gd.data.length - 2, hit);
  }}
  count.textContent = q.length < 2 ? "" : `${{hit.length}} match${{hit.length === 1 ? "" : "es"}}`;
}});
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
