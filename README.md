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
divinities such as nymphs and household spirits, people worshipped as divine such as
Guan Yu, and named groups such as the Muses. Modern fiction (Lovecraft, comics, games),
non-divine mythological figures, and articles about deity *types* ("Sun deity") are
left out.

A person worshipped as divine is someone who lived, in history or in legend, whatever
their tradition says about how they came to be divine: a man made a god, like Guan Yu,
or a god born as a man, like Swaminarayan. Wikipedia's editors decide where they can.
The 179 individual beings on the map that Wikipedia files under a "Deified …" category
(Deified Chinese women, Deified male monarchs) all count, among them Jesus, Jesus in
Christianity, Mazu, Shango and the Buddha. The categories say who is deified, never who
isn't, so for the rest Claude re-read each lead under that rule (`enrich.py --rerun`)
and decided the other 77, such as Heracles, Wong Tai Sin and the Jain tīrthaṅkaras.
God the Son, Krishna and Rama stay gods. Where the categories speak, Claude agrees on
82% (147 of 179); two thirds of the rest are Burmese nats and Chinese gods that the
categories file as people but whose leads present a god.

The candidate pool is generous on purpose, because neither obvious source is complete:

- **Wikidata**: items that are an instance of deity (Q178885) or any subclass of it.
  Precise, but it misses whole groups whose classes aren't linked into that tree:
  Vodou loa, deified humans typed as `human`, theonyms such as Allah.
- **Wikipedia's Category:Deities**, crawled three levels deep. It finds those, but
  also pulls in temples, festivals, artworks and fiction.

Claude reads every candidate's lead section and decides whether it belongs. A hand
review of the people worshipped as divine and of modern religions then removed 67
more, by rule rather than case by case (`exclusions.py`): things that are not divine
at all (Sapa Inca is an office, the Flying Spaghetti Monster is satire), biographies
whose lead never says the person was worshipped, people worshipped only by their own
movement who are living or died after 2000, and people whose deity has its own article
(Sugawara no Michizane, who is also Tenjin).

## Traditions

Both maps colour deities by tradition, and the role map erases it, so the list of
traditions matters. A rule in `taxonomy.py` defines it, and `taxonomy_audit.py` fails
if the list and the rule disagree. A tradition gets a category of its own when
Wikipedia files its gods in a deity category of their own under *Deities by continent*
or *Deities by religion*, and Wikipedia's categories put at least 15 of the map's
articles in it. Any floor from 10 to 22 picks the same 33 traditions, so the list does
not hang on the number chosen. The traditions below it (Proto-Indo-European, Bon,
Guanche, Nuristani and a few others) share an "Other" colour but keep their own name
on the card. "Other European" is a leftover by construction: Thracian, Basque,
Albanian and the rest of Europe's smaller traditions.

Two independent raters assign each article's tradition: Wikipedia's own categories,
mapped to traditions by hand-written patterns (`category_labels.py`), and Claude's
reading of the lead (`enrich.py`). Claude's label is among the categories' for 97% of
articles. The categories decide, because they are editors' labels that anyone can
check, and they can name several traditions for a syncretic figure (521 deities have
two or more, such as Hindu and Buddhist). Claude's label picks the colour among several
and fills in for the 19 articles whose categories name no tradition; otherwise it is
the check on the categories.

## How the role map removes culture

A deity's Wikipedia lead is mostly about its pantheon: genealogy, cult sites, sources.
Embedded as is, 88% of a deity's nearest neighbours come from its own tradition. The
role map changes two things, both tested in `experiments/erase_tradition.py`:

1. **An embedding instruction.** Qwen3-Embedding accepts a task instruction; the role
   map asks it to represent "the deity's divine roles, powers and domains of influence".
2. **Concept erasure.** [LEACE](https://arxiv.org/abs/2306.03819) then removes every
   linear trace of the 34 tradition categories that Wikipedia's categories assign,
   changing the vectors as little as possible. Toponymy's keyphrases go through the
   same instruction and eraser, so the region names are computed in the same space.

Claude's tradition labels are kept out of the eraser so they can check it. With the
eraser fit on a training split, a linear probe finds Claude's tradition for 20.0% of
held-out deities, against 16.4% for always guessing the largest tradition.

Measured on the 4,551 deities, as the share of each deity's 15 nearest neighbours on
the map, with chance in brackets:

| | Same tradition | Same main domain (Claude) | Shared function category (Wikipedia) | Cross-cultural neighbours with the same main domain |
|---|---|---|---|---|
| Pantheon map | 0.88 (0.06) | 0.23 (0.07) | 0.38 (0.16) | 0.22 (0.07) |
| Role map | 0.20 | 0.56 | 0.71 | 0.55 |

The function categories are Wikipedia's *Deities by function* tree (storm gods, war
goddesses, psychopomps), so the role structure is measured by a rater that had no part
in building the map. What tradition clustering remains, 0.20, is close to the 0.16 a
map arranged purely by role would show, since traditions differ in their mix of roles:
Greek myth has a great many water nymphs.

Erasure alone was not enough: without the instruction, the leads have too little role
content left once culture is gone (with a smaller embedding model the result was a
featureless ball). A centroid projection erased tradition as completely as LEACE and
scored the same on domain; LEACE was kept because it moves the vectors about half as
far and keeps more of the remaining structure. `experiments/erase_by_categories.py`
found that an eraser fit on Claude's labels gives the same map as one fit on
Wikipedia's, and that erasing 86 finer cultures as well (Hittite and Hurrian, Aztec and
Maya) halved the clustering within traditions but cost some role structure, so it was
not adopted.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Candidates | `candidates.py` | Wikidata deity tree ∪ Category:Deities to depth 3 |
| Fetch | `fetch.py` | Lead sections, short descriptions, thumbnails and page metadata from the MediaWiki API |
| Enrich | `enrich.py` | Claude Sonnet 5.5 (Message Batches API) classifies each article and labels its tradition, domains and gender, with a one-line summary |
| Pageviews | `pageviews.py` | Twelve months of human pageviews for the articles on the map |
| Categories | `categories.py` | Each article's Wikipedia categories, the tree two levels above them, and Wikipedia's deities-by-culture tree |
| Embed | `embed.py` | Qwen3-Embedding-8B (through OpenRouter) embeds name + lead section; the role map adds the instruction and LEACE, fit on the categories' traditions |
| Cluster | `cluster.py` | UMAP to 2-d, then [Toponymy](https://github.com/TutteInstitute/toponymy) finds and names regions with Claude Opus 5.5 |
| Render | `render.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds `docs/index.html` and `docs/archetypes.html` |
| Card | `social_preview.py` | Screenshots each rendered map in Chrome for its link-preview image |

`category_labels.py` maps Wikipedia's categories to traditions and domains and decides
who is a person worshipped as divine, and
`taxonomy.py` holds the tradition list. `variants.py` holds everything that differs
between the two maps. Regions are found
on the same 2-d layout that is plotted, so every named region matches something
visible on the map. Continent names Toponymy gets wrong can be corrected by hand in
`variants.py` (none need it at present): it names a region after its most central
members, which can be the many minor figures around a few major gods.

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
uv run enrich.py --rerun  # the map's articles again, for who is a person
uv run pageviews.py
uv run categories.py
for v in pantheon archetype; do
  uv run embed.py --variant $v
  uv run cluster.py --variant $v --explore
  uv run cluster.py --variant $v --min-cluster-size 8
  uv run render.py --variant $v
  uv run social_preview.py --variant $v
done
uv run taxonomy_audit.py  # checks the tradition list against its rule
```

Intermediate data lives in `data/` and is not committed. Fetches, embeddings and LLM
responses are cached on disk, so re-running a stage does not repeat network or paid
calls.

## Credits

Article titles, short descriptions, thumbnails and pageview counts come from English
Wikipedia and Wikimedia Commons; text there is available under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and each image under
the license shown on its Commons page. Clicking a point opens the article.
