"""The two maps built from the same corpus, and everything that differs between them.

  pantheon   name + lead section embedded as is: the layout is organised by tradition
             (a deity's 15 nearest map neighbours share its tradition ~87% of the time).
  archetype  the same text embedded with an instruction asking for the deity's roles,
             then tradition erased with LEACE: the layout is organised by what gods do.

Chosen in experiments/erase_tradition.py (2026-10-15). With Qwen3-Embedding-8B the
archetype recipe gave the best domain structure of everything tried: 2-d domain purity
0.51 and same-domain cross-tradition neighbours 0.53 (chance 0.07), with tradition purity
down to 0.20 (chance 0.06). Erasing tradition without the instruction left far less
structure, and with the 0.6B model it left a featureless ball.

Each stage takes `--variant` and reads and writes that variant's files.
"""

from dataclasses import dataclass, field

from common import DATA

# Phrased positively: embedding models handle "ignore X" poorly.
ROLE_INSTRUCTION = (
    "Given a Wikipedia description of a deity, represent the deity's divine roles, "
    "powers and domains of influence"
)


@dataclass(frozen=True)
class MapVariant:
    name: str
    instruction: str | None
    erase_tradition: bool
    html: str  # page under docs/
    social_preview: str  # link-card image under docs/
    title: str
    subtitle: str  # "{n}" is filled with the article count
    og_description: str
    # The colormap listed first in the dropdown, after DataMapPlot's own "Clusters":
    # the field the layout is organised around.
    lead_colormap: str
    # Toponymy's corpus descriptions and any extra instruction for the region namer.
    object_description: str
    corpus_description: str
    naming_note: str
    # UMAP's orientation is arbitrary; (x, y) signs chosen by eye on the rendered map.
    orientation: tuple[int, int]
    # Hand corrections to the coarsest layer's names (see render.py).
    coarsest_name_overrides: dict = field(default_factory=dict)
    # The other map, linked from the title panel: (variant name, link text).
    other: tuple[str, str] | None = None

    @property
    def embeddings(self):
        return DATA / f"embeddings_{self.name}.npz"

    @property
    def eraser(self):
        return DATA / f"eraser_{self.name}.pt"

    @property
    def coords(self):
        return DATA / f"umap_coords_{self.name}.npz"

    @property
    def labels(self):
        return DATA / f"labels_{self.name}.parquet"

    @property
    def topic_names(self):
        return DATA / f"topic_names_{self.name}.json"


VARIANTS = {
    "pantheon": MapVariant(
        name="pantheon",
        instruction=None,
        erase_tradition=False,
        html="index.html",
        social_preview="social-preview.png",
        title="Map of the Gods",
        subtitle="{n} divine beings placed by meaning. Click one to open it.",
        og_description=(
            "{n} gods, goddesses and other divine beings from English Wikipedia, laid out "
            "by the meaning of their articles and named at several zoom levels. "
            "Pan, zoom, hover and search."
        ),
        lead_colormap="tradition",
        object_description="Wikipedia articles about gods, goddesses and other divine beings",
        corpus_description="English Wikipedia's articles on the world's deities",
        naming_note="",
        orientation=(-1, 1),
        # Toponymy names a region after its most central members. This continent's
        # are mostly Native American, African and Polynesian, but it also holds 421
        # Norse, Slavic, Baltic, Finnic and Turkic deities, so Odin and Thor sat under
        # "Indigenous Mythologies". Keyed by name, so a re-run of cluster.py that
        # renames it fails loudly in render.py instead of silently dropping this.
        coarsest_name_overrides={
            "Indigenous Mythologies": "Northern and Indigenous Gods",
            # The two Greek continents sit side by side, and their labels are too close
            # for both to show at overview: this one's six-word name hid both, and a
            # shorter one won the collision and labelled Zeus's side "Nymphs". One shared
            # name reads right whichever label wins; the next layer down still says
            # "Oceanids, Naiads, Titans and River-Gods".
            "Oceanids, Naiads, Titans and River-Gods": "Greek Deities",
        },
        other=("archetype", "See them by role →"),
    ),
    "archetype": MapVariant(
        name="archetype",
        instruction=ROLE_INSTRUCTION,
        erase_tradition=True,
        html="archetypes.html",
        social_preview="social-preview-archetypes.png",
        title="The Gods by Role",
        subtitle="{n} divine beings placed by what they do. Click one to open it.",
        og_description=(
            "{n} gods, goddesses and other divine beings from English Wikipedia, laid out "
            "by what they do, with culture factored out: storm gods beside storm gods, "
            "from every tradition. Pan, zoom, hover and search."
        ),
        lead_colormap="domain",
        object_description="Wikipedia articles about gods, goddesses and other divine beings",
        corpus_description=(
            "English Wikipedia's articles on the world's deities, arranged by their roles "
            "rather than their cultures"
        ),
        naming_note=(
            "\n\nThese regions group deities from many different cultures by what they do: "
            "cultural and regional identity was deliberately removed from the layout. Name "
            "each region for the role, power or archetype its members share (for example "
            '"Storm and Thunder Gods" or "Psychopomps and Guides of the Dead"), not for a '
            "culture or pantheon. Name a culture only if one genuinely dominates the region."
        ),
        orientation=(-1, 1),
        other=("pantheon", "See them by pantheon →"),
    ),
}


def variant_arg(parser) -> None:
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
