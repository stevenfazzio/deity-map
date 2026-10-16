# Map of the Gods

Two interactive maps of the gods, goddesses and other divine beings that have English
Wikipedia articles, named at several zoom levels:

- **By pantheon:** https://stevenfazzio.com/deity-map/ places each deity by the meaning
  of its article's lead section, so traditions form continents: Greek, Hindu, Aztec.
- **By role:** https://stevenfazzio.com/deity-map/archetypes.html places the same
  deities by what they do, with culture factored out, so storm gods sit beside storm
  gods from every tradition.

## What's on the maps

Anything treated as divine by a real religious or mythological tradition, past or
present: gods and goddesses (including monotheistic conceptions of God), minor
divinities such as nymphs and household spirits, deified people such as Guan Yu, and
named groups such as the Muses. Modern fiction (Lovecraft, comics, games), non-divine
mythological figures, and articles about deity *types* ("Sun deity") are left out.

The candidate pool is generous on purpose, because neither obvious source is complete:

- **Wikidata**: items that are an instance of deity (Q178885) or any subclass of it.
  Precise, but it misses whole groups whose classes aren't linked into that tree:
  Vodou loa, deified humans typed as `human`, theonyms such as Allah.
- **Wikipedia's Category:Deities**, crawled three levels deep. It finds those, but
  also pulls in temples, festivals, artworks and fiction.

Claude reads every candidate's lead section and decides whether it belongs; the maps
show whatever passes that filter.

## How the role map removes culture

A deity's Wikipedia lead is mostly about its pantheon: genealogy, cult sites, sources.
Embedded as is, 87% of a deity's nearest neighbours come from its own tradition. The
role map changes two things, both tested in `experiments/erase_tradition.py`:

1. **An embedding instruction.** Qwen3-Embedding accepts a task instruction; the role
   map asks it to represent "the deity's divine roles, powers and domains of influence".
2. **Concept erasure.** [LEACE](https://arxiv.org/abs/2306.03819) then removes every
   linear trace of the 38 tradition categories, changing the vectors as little as
   possible. Toponymy's keyphrases go through the same instruction and eraser, so the
   region names are computed in the same space.

Measured on the 4,618 deities, with chance in brackets:

| | Neighbours sharing a tradition | Neighbours sharing a main domain | Cross-cultural neighbours sharing a domain |
|---|---|---|---|
| Pantheon map | 0.87 (0.06) | 0.22 (0.07) | 0.21 (0.07) |
| Role map | 0.20 | 0.51 | 0.53 |

Erasure alone was not enough: without the instruction, the leads have too little role
content left once culture is gone (with a smaller embedding model the result was a
featureless ball). A centroid projection erased tradition as completely as LEACE and
scored the same on domain; LEACE was kept because it moves the vectors about half as
far and keeps more of the remaining structure.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Candidates | `candidates.py` | Wikidata deity tree ∪ Category:Deities to depth 3 |
| Fetch | `fetch.py` | Lead sections, short descriptions, thumbnails and page metadata from the MediaWiki API |
| Enrich | `enrich.py` | Claude Sonnet 5.5 (Message Batches API) classifies each article and labels its tradition, domains and gender, with a one-line summary |
| Pageviews | `pageviews.py` | Twelve months of human pageviews for the articles on the map |
| Embed | `embed.py` | Qwen3-Embedding-8B (through OpenRouter) embeds name + lead section; the role map adds the instruction and LEACE |
| Cluster | `cluster.py` | UMAP to 2-d, then [Toponymy](https://github.com/TutteInstitute/toponymy) finds and names regions with Claude Opus 5.5 |
| Render | `render.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds `docs/index.html` and `docs/archetypes.html` |
| Card | `social_preview.py` | Screenshots each rendered map in Chrome for its link-preview image |

`variants.py` holds everything that differs between the two maps. Regions are found
on the same 2-d layout that is plotted, so every named region matches something
visible on the map. Continent names Toponymy gets wrong are corrected by hand in
`variants.py`: it names a region after its most central members, which can be the
many minor figures around a few major gods.

The enrichment model was chosen on a 59-article sample run through Opus, Sonnet and
Haiku 5.5: Sonnet matched Opus on 57 of 59 scope decisions at half the price, while
Haiku broke the prompt's explicit rules. `enrich.py --model` reruns that comparison.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/), `ANTHROPIC_API_KEY` and
`OPENROUTER_API_KEY`. The social card needs a local Chrome.

```bash
uv run candidates.py
uv run fetch.py
uv run enrich.py --sample 60 --direct && uv run enrich.py
uv run pageviews.py
for v in pantheon archetype; do
  uv run embed.py --variant $v
  uv run cluster.py --variant $v --explore
  uv run cluster.py --variant $v --min-cluster-size 8
  uv run render.py --variant $v
  uv run social_preview.py --variant $v
done
```

Intermediate data lives in `data/` and is not committed. Fetches, embeddings and LLM
responses are cached on disk, so re-running a stage does not repeat network or paid
calls.

## Credits

Article titles, short descriptions, thumbnails and pageview counts come from English
Wikipedia and Wikimedia Commons; text there is available under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and each image under
the license shown on its Commons page. Clicking a point opens the article.
