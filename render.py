"""Render one variant's interactive datamap under docs/, which GitHub Pages serves.

The page carries model-written text (one-line summaries, region names), the
categorical labels, Wikipedia titles and pageview counts, and hotlinks each
article's Wikipedia thumbnail. Lead sections stay out of the page: they're in the
embeddings, not the HTML.

    uv run render.py --variant pantheon          # docs/index.html
    uv run render.py --variant archetype         # docs/archetypes.html
    uv run social_preview.py --variant pantheon  # screenshot -> link-card image
"""

import argparse
import colorsys
import html
import os
import re
import tempfile
import urllib.parse
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

from common import ROOT, write_bytes_atomic
from embed import load_corpus
from pageviews import PAGEVIEWS_PARQUET
from taxonomy import (
    DOMAIN_LABELS,
    GENDER_LABELS,
    KIND_LABELS,
    resolve_traditions,
    upper_first,
)
from variants import VARIANTS, MapVariant, variant_arg

DOCS = ROOT / "docs"
PUBLIC_URL = "https://stevenfazzio.com/deity-map/"
SNAPSHOT = "October 2026"  # when the articles were fetched (candidates.py, fetch.py)

# Type: a book serif for the title, for a map that reads like a reference work on
# mythology; labels and hovercards in a sans, which holds up small, bold and outlined
# over points. DataMapPlot writes font names unquoted, so neither may contain a token
# starting with a digit.
BODY_FONT = "IBM Plex Sans"  # DataMapPlot loads it (labels, subtitle, hovercards)
TITLE_FONT_CSS = "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600&display=swap"

CUSTOM_CSS = """
#main-title {
  font-family: 'Cormorant Garamond', Georgia, serif !important;
  font-weight: 600 !important;
  /* DataMapPlot's 0.95 lets a wrapped title's descenders hit the subtitle. */
  line-height: 1.1 !important;
  text-wrap: balance;
}
#main-title + br + span {
  display: inline-block; margin-top: 4px; line-height: 1.3; text-wrap: pretty;
}
.title-pill {
  display: inline-block; margin: 8px 6px 0 0; padding: 3px 10px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
  color: #636c76; background: rgba(99, 108, 118, 0.08);
  border: 1px solid rgba(99, 108, 118, 0.15); border-radius: 12px;
}
.title-link {
  display: inline-block; margin: 8px 0 0 4px; font-size: 12px; font-weight: 500;
  color: #4b5563; text-decoration: none; border-bottom: 1px solid #c7ccd1;
}
.title-link:hover { color: #111827; border-bottom-color: #111827; }
/* Articles without a thumbnail get no image box at all. */
.card-img[src=""] { display: none; }
/* DataMapPlot's legend has no height limit, so Tradition's 38 entries ran the full
   height of a laptop screen and covered the labels on the map's east side. */
.color-legend-container { max-height: 55vh; overflow-y: auto; }
.legend-item { padding: 2px 4px; }
"""

# Non-categories are pinned to greys so the named categories carry the colour. Each
# grey is distinct: the legend selects points by colour, so a shared grey would select
# two categories at once. "God or goddess" is most of the map, so as a pale background
# it turns the Type colormap into a picture of where the other kinds are.
NEUTRAL_COLORS = {
    "God or goddess": "#d4d4d4",
    "Other": "#8c8c8c",
    "Not known": "#bdbdbd",
    "Unspecified": "#a6a6a6",
    "Mixed group": "#737373",
    "Unknown": "#e6e6e6",
}

# Cards are read in quick succession while sweeping the mouse, so every card has the
# same layout: tradition pill, name, labelled fields, then the summary and a footer.
# The fields sit above the summary because summaries vary in length, so they land at
# the same height card to card. The dots use the colormap colours, so a card reads
# against whichever legend is showing.
DOT = (
    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; "
    "margin-right: 6px; box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15); "
    "background: {color};"
)
FIELD_NAME = "color: #6b7280; font-size: 11px; padding-top: 1px;"
HOVER_TEMPLATE = f"""
<div style="max-width: 340px; white-space: normal; font-weight: 400; color: #1f2328;">
  <img src="{{thumb_url}}" class="card-img" alt="" onerror="this.style.display='none'"
    style="float: right; width: 72px; max-height: 96px; object-fit: cover;
    border-radius: 4px; margin: 0 0 6px 10px; background: #f3f4f6;"/>
  <span style="display: inline-block; padding: 1px 8px; margin-bottom: 5px;
    border-radius: 999px; font-size: 11px; font-weight: 600;
    background: {{pill_bg}}; color: {{pill_fg}};">{{tradition}}</span>
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25;">{{name}}</div>
  <div style="display: grid; grid-template-columns: 52px minmax(0, 1fr);
    gap: 3px 10px; margin-top: 6px; font-size: 12px; line-height: 1.35;">
    <div style="{FIELD_NAME}">Type</div>
    <div><span style="{DOT.format(color="{kind_dot}")}"></span>{{kind}}</div>
    <div style="{FIELD_NAME}">Domain</div>
    <div><span style="{DOT.format(color="{domain_dot}")}"></span>{{domains}}</div>
    <div style="{FIELD_NAME}">Gender</div>
    <div><span style="{DOT.format(color="{gender_dot}")}"></span>{{gender}}</div>
  </div>
  <div style="clear: both; font-size: 13px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4;">{{summary}}</div>
  <div style="color: #6b7280; font-size: 11px; margin-top: 7px;">{{footer}}</div>
</div>
"""

# Pageview percentiles are ranked within this Toponymy layer's regions (0 = finest).
# Raw views mostly re-encode which pantheon is famous; within a region, the locally
# best-known deity stands out everywhere on the map.
RANK_LAYER = 1


def categorical_palette(n_colors: int, avoid: list[str]) -> list[str]:
    """n_colors glasbey colours, each also kept distinct from the `avoid` colours."""
    # Seeding with the greys already on the map stops glasbey handing out a muted
    # colour that reads as one of them.
    palette = glasbey.extend_palette(
        avoid,
        palette_size=len(avoid) + n_colors,
        colorblind_safe=True,
        cvd_severity=50.0,  # glasbey's default, moderate red-green CVD
        # A floor of 35: near-black colours are distinct as swatches but read as the
        # same colour as small dots. The ceiling keeps swatches from washing out.
        lightness_bounds=(35, 75),
        chroma_bounds=(20, 90),
        # Always explicit: with four or more seed greys glasbey 0.3 takes the hue
        # range from the seeds, which span no hues, and crashes.
        hue_bounds=(0, 360),
    )
    return palette[len(avoid) :]


def color_mapping(values: pd.Series) -> dict[str, str]:
    """Category -> colour, most frequent first, since glasbey's earliest are most distinct."""
    order = values.value_counts().index.tolist()
    named = [c for c in order if c not in NEUTRAL_COLORS]
    neutrals = {c: NEUTRAL_COLORS[c] for c in order if c in NEUTRAL_COLORS}
    mapping = dict(zip(named, categorical_palette(len(named), list(neutrals.values()))))
    mapping.update(neutrals)
    # The legend selects points by colour, so no two categories may share one.
    assert len(set(mapping.values())) == len(mapping), f"two categories share a colour: {mapping}"
    return mapping


def categorical_colormap(field: str, description: str, values: pd.Series):
    # Explicit "Unknown" rather than dropping points, so gaps in the metadata don't
    # read as holes in the map.
    values = values.fillna("Unknown").astype(str)
    mapping = color_mapping(values)
    meta = {
        "field": field,
        "description": description,
        "kind": "categorical",
        # An explicit mapping, so the hovercard can use exactly the map's colours.
        "color_mapping": mapping,
        # DataMapPlot hides the legend above 20 categories unless asked; Tradition has
        # 38 and Main domain 23. It scrolls past 500 px rather than growing.
        "show_legend": True,
    }
    return values.to_numpy(), meta, mapping


def relative_luminance(rgb) -> float:
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b) -> float:
    la, lb = sorted([relative_luminance(a), relative_luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def pill_colors(hex_color: str) -> tuple[str, str]:
    """A pale tint for the pill background and a deep shade of the same hue for text."""
    h, _, s = colorsys.rgb_to_hls(*to_rgb(hex_color))
    bg = colorsys.hls_to_rgb(h, 0.92, min(s, 0.65))
    lightness = 0.35
    fg = colorsys.hls_to_rgb(h, lightness, s)
    while contrast(fg, bg) < 4.5:  # WCAG AA for small text
        lightness -= 0.03
        fg = colorsys.hls_to_rgb(h, lightness, s)
    return to_hex(bg), to_hex(fg)


def domains_text(row) -> str:
    primary = (
        upper_first(row.primary_domain_other)
        if row.primary_domain == "other" and row.primary_domain_other.strip()
        else DOMAIN_LABELS[row.primary_domain]
    )
    rest = [DOMAIN_LABELS[d] for d in row.secondary_domains]
    return " · ".join([primary, *rest])


def wiki_url(title: str) -> str:
    # Percent-encodes backticks too, which matters: on_click interpolates the URL
    # into a JavaScript template literal.
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_"), safe="/:(),'!*"
    )


def compact_number(n: float) -> str:
    """1540 -> '1.5k', 12345 -> '12k', 2340000 -> '2.3M'."""
    # Choose the unit after rounding, so 999,999 reads 1M rather than 1000k.
    for scale, unit in [(1e6, "M"), (1e3, "k")]:
        if round(n / scale, 1) >= 1:
            value = n / scale
            text = f"{value:.0f}" if value >= 10 else f"{value:.1f}".removesuffix(".0")
            if text == "1000" and unit == "k":
                return "1M"
            return text + unit
    return f"{n:.0f}"


def page_url(variant: MapVariant) -> str:
    return PUBLIC_URL + ("" if variant.html == "index.html" else variant.html)


def title_pills(variant: MapVariant) -> str:
    pills = [f"English Wikipedia · {SNAPSHOT}"]
    spans = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in pills)
    if variant.other:
        other, text = VARIANTS[variant.other[0]], variant.other[1]
        # Relative, so the pair of pages works from any host, local preview included.
        spans += f'<a class="title-link" href="{other.html}">{html.escape(text)}</a>'
    return f"<div>{spans}</div>"


def open_graph_tags(variant: MapVariant, n: int) -> str:
    og = {
        "og:title": variant.title,
        "og:description": variant.og_description.format(n=f"{n:,}"),
        "og:type": "website",
        "og:url": page_url(variant),
        # Declared unconditionally: social_preview.py screenshots the rendered page,
        # so the image is always produced after the HTML that names it.
        "og:image": PUBLIC_URL + variant.social_preview,
    }
    tags = [f'<meta property="{k}" content="{html.escape(v, quote=True)}">' for k, v in og.items()]
    tags.append(f'<meta name="description" content="{html.escape(og["og:description"])}">')
    tags.append('<meta name="twitter:card" content="summary_large_image">')
    return "\n".join(tags)


# Panels over the map at 1280x800, measured on the rendered page (px from top-left).
PANELS_1280 = {
    "title": (13, 624, 8, 141),
    "search": (19, 195, 160, 211),
    "dropdown": (19, 448, 717, 768),
    "legend": (1069, 1267, 13, 640),  # the tallest categorical legend, capped by CSS
}


def panel_overlap(xy: np.ndarray, w: int = 1280, h: int = 800) -> dict[str, int]:
    """Points that would start under each panel, using DataMapPlot's initial framing
    (bounds padded 1%, then zoomed out half a level)."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = (hi - lo) * 1.02
    scale = min(w / span[0], h / span[1]) * 2**-0.5
    centre = (lo + hi) / 2
    sx = w / 2 + (xy[:, 0] - centre[0]) * scale
    sy = h / 2 - (xy[:, 1] - centre[1]) * scale
    return {
        k: int(((sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)).sum())
        for k, (x0, x1, y0, y1) in PANELS_1280.items()
    }


def percentile(values: pd.Series) -> pd.Series:
    """0-100, ties averaged, so equal view counts get equal colours."""
    return values.rank(pct=True, method="average") * 100


def main():
    parser = argparse.ArgumentParser()
    variant_arg(parser)
    variant = VARIANTS[parser.parse_args().variant]
    output = DOCS / variant.html

    corpus = load_corpus()
    labels = pd.read_parquet(variant.labels)
    views = pd.read_parquet(PAGEVIEWS_PARQUET)

    df = corpus.merge(labels, on="pageid", how="left", validate="1:1")
    df = df.merge(views, on="pageid", how="left", validate="1:1")
    assert len(df) == len(corpus), "merge changed the row count"
    missing_xy = df.x.isna().sum()
    assert missing_xy == 0, f"{missing_xy} articles have no coordinates"
    missing_views = df.views_12mo.isna().sum()
    assert missing_views == 0, f"{missing_views} articles have no pageviews; run pageviews.py"
    print(f"render ({variant.name}): {len(df)} articles")

    kind_label = df.kind.map(KIND_LABELS)
    tradition_label, tradition_category, region_label = resolve_traditions(df)
    domain_label = df.primary_domain.map(DOMAIN_LABELS)
    gender_label = df.gender.map(GENDER_LABELS)
    summary = df.summary.fillna("").str.strip()
    unterminated = summary.ne("") & ~summary.str.endswith((".", "?", "!"))
    summary[unterminated] = summary[unterminated] + "."

    # Colormap values collapse rare free-text "other" answers into one "Other"
    # category; the hovercard and search keep the specific text.
    tradition_values, tradition_meta, tradition_colors = categorical_colormap(
        "tradition", "Tradition", tradition_category
    )
    region_values, region_meta, _ = categorical_colormap("region", "Broad region", region_label)
    kind_values, kind_meta, kind_colors = categorical_colormap("kind", "Type", kind_label)
    domain_values, domain_meta, domain_colors = categorical_colormap(
        "domain", "Main domain", domain_label
    )
    gender_values, gender_meta, gender_colors = categorical_colormap(
        "gender", "Gender", gender_label
    )
    pills = {cat: pill_colors(color) for cat, color in tradition_colors.items()}

    label_cols = sorted(
        (c for c in df.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),  # numeric: _10 must sort after _2
    )
    label_layers = [df[c].fillna("Unlabelled").to_numpy() for c in label_cols]
    coarsest = pd.Series(label_layers[-1])
    overrides = variant.coarsest_name_overrides
    stale = set(overrides) - set(coarsest)
    assert not stale, f"name overrides no longer match any coarsest region: {stale}"
    label_layers[-1] = coarsest.replace(overrides).to_numpy()
    n_finest = pd.Series(label_layers[0]).nunique()
    n_coarsest = pd.Series(label_layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"

    # Unlabelled points are ranked among themselves, i.e. against the map as a whole.
    rank_region = df[label_cols[min(RANK_LAYER, len(label_cols) - 1)]].fillna("Unlabelled")
    views_global = percentile(df.views_12mo)
    views_local = df.groupby(rank_region).views_12mo.transform(percentile)

    domains = pd.Series([domains_text(r) for r in df.itertuples()], index=df.index)
    footer = [f"{compact_number(v)} views in the past year · Wikipedia" for v in df.views_12mo]
    extra = pd.DataFrame(
        {
            "name": df.name.map(html.escape),
            "summary": summary.map(html.escape),
            # The API appends utm_* tracking parameters; the bare URL serves the same
            # image and saves ~80 bytes on each of ~2,400 thumbnails.
            "thumb_url": df.thumb_url.fillna("")
            .str.replace(r"\?.*$", "", regex=True)
            .map(lambda u: html.escape(u, quote=True)),
            "tradition": tradition_label.map(html.escape),
            "pill_bg": [pills[v][0] for v in tradition_values],
            "pill_fg": [pills[v][1] for v in tradition_values],
            "kind": kind_label.map(html.escape),
            "kind_dot": [kind_colors[v] for v in kind_values],
            "domains": domains.map(html.escape),
            "domain_dot": [domain_colors[v] for v in domain_values],
            "gender": gender_label.map(html.escape),
            "gender_dot": [gender_colors[v] for v in gender_values],
            "footer": [html.escape(f) for f in footer],
            "url": df.title.map(wiki_url),
            # Search the composed text, not just names: the summary, the Wikipedia
            # description and the facets are where a deity's content shows up.
            "search_text": (
                df.name
                + " "
                + df.description.fillna("")
                + " "
                + summary
                + " "
                + tradition_label
                + " "
                + kind_label
                + " "
                + domains
            ),
        }
    )

    colormaps = {
        "tradition": (tradition_values, tradition_meta),
        "region": (region_values, region_meta),
        "kind": (kind_values, kind_meta),
        "domain": (domain_values, domain_meta),
        "gender": (gender_values, gender_meta),
    }
    # The field the layout is organised around goes first: tradition on the pantheon
    # map, domain on the archetype map.
    order = [variant.lead_colormap] + [k for k in colormaps if k != variant.lead_colormap]
    colormap_rawdata = [colormaps[k][0] for k in order] + [
        views_global.to_numpy(),
        views_local.to_numpy(),
    ]
    colormap_metadata = [colormaps[k][1] for k in order] + [
        # One ramp for both, so they can be compared.
        {
            "field": "views_all",
            "description": "Pageviews (percentile, whole map)",
            "kind": "continuous",
            "cmap": "viridis",
        },
        {
            "field": "views_region",
            "description": "Pageviews (percentile within region)",
            "kind": "continuous",
            "cmap": "viridis",
        },
    ]

    # UMAP's orientation is arbitrary; each variant's is chosen so nothing starts under
    # a panel. The report shows all four mirror images for when a re-run moves things.
    xy = df[["x", "y"]].to_numpy()
    for flip in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        mark = "  <- used" if flip == variant.orientation else ""
        print(f"  orientation {flip}: points under panels {panel_overlap(xy * flip)}{mark}")

    fig = datamapplot.create_interactive_plot(
        xy * variant.orientation,
        *label_layers,
        hover_text=df.name.to_numpy(),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        colormap_rawdata=colormap_rawdata,
        colormap_metadata=colormap_metadata,
        cvd_safer=True,
        title=variant.title,
        # One line at 1280 px: a second line pushes the panel over the top labels.
        # The source is in the pill underneath, so the subtitle needn't name it, and
        # it says clicking works, which nothing else does.
        sub_title=variant.subtitle.format(n=f"{len(df):,}"),
        # The defaults (36/18) make the title panel cover the top of the map and
        # collide with the colormap legend on laptop-width screens.
        title_font_size=24,
        sub_title_font_size=13,
        font_family=BODY_FONT,
        tooltip_font_family=BODY_FONT,
        tooltip_font_weight=400,
        custom_css=CUSTOM_CSS,
        noise_label="Unlabelled",
        # Labels are collision-tested at this multiple of their size (default 3). At 3
        # the role map's "Water Deities" was hidden by "Mother Goddesses" 167 px away,
        # leaving one of its four continents unnamed at overview.
        text_collision_size_scale=2.0,
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DOCS, suffix=".html")
    os.close(fd)
    try:
        fig.save(tmp)
        text = Path(tmp).read_text(encoding="utf-8")
        # After the charset declaration, which browsers only honour in the first
        # 1024 bytes of the document.
        anchor = re.search(r"<meta[^>]*charset[^>]*>", text) or re.search("<head>", text)
        assert anchor is not None, "no <head> in the rendered page"
        at = anchor.end()
        head = (
            open_graph_tags(variant, len(df))
            + '\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            + f'\n<link rel="stylesheet" href="{TITLE_FONT_CSS}">'
        )
        text = text[:at] + "\n" + head + text[at:]
        # Pills go inside the title panel, after the subtitle; the panel holds only
        # spans, so its first closing </div> is its own.
        text, n = re.subn(
            r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
            lambda m: m.group(1) + title_pills(variant) + m.group(2),
            text,
            count=1,
            flags=re.DOTALL,
        )
        assert n == 1, "title panel not found for the pills"
        write_bytes_atomic(text.encode("utf-8"), output)
    finally:
        os.unlink(tmp)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    print(f"  next: uv run social_preview.py --variant {variant.name}")


if __name__ == "__main__":
    main()
