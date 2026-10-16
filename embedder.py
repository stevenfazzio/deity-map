"""Text embedders with an on-disk cache: Qwen3-Embedding locally or through OpenRouter.

Both implement Toponymy's TextEmbedderProtocol (`encode(texts, show_progress_bar)`), so
the same object embeds the articles and, later, Toponymy's keyphrases and topic
names. Every vector is cached on disk by (model, text) hash: re-running any stage
never re-embeds a string, and the two backends never collide in the cache.

An optional `instruction` is prepended to every text in Qwen3-Embedding's format
(`Instruct: ...\\nQuery:...`), steering what the vectors emphasise. It applies to every
text the embedder sees, Toponymy's keyphrases included, and it is part of the cached
text, so instructed and plain vectors never collide either.

  LocalEmbedder       Qwen3-Embedding-0.6B on Apple-silicon MPS (or CPU). Texts longer
                      than MAX_SEQ_LENGTH tokens are truncated, which only affects the
                      tail of the longest leads.
  OpenRouterEmbedder  Qwen3-Embedding-8B (the maps' model) or another Qwen3 size through
                      OpenRouter's embeddings API, pinned to one provider so every
                      vector comes from the same deployment. Needs OPENROUTER_API_KEY.
                      Texts are not truncated.
  ErasedEmbedder      wraps either, applying a fitted concept eraser to every vector, so
                      Toponymy's keyphrases land in the same erased space as the articles.
"""

import hashlib
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from tqdm import tqdm

from common import DATA

MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LENGTH = 1024
CACHE_PATH = DATA / "embedding_cache.sqlite"
BATCH_SIZE = 32
# Rows per cache commit: a crash loses at most this much work.
CHUNK = 256

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"
# Pinned so a fallback can't mix deployments. DeepInfra serves both the 4B and 8B;
# the 8B's other providers are Nebius and SiliconFlow (fp8-quantised), 2026-10-15.
OPENROUTER_PROVIDER = "DeepInfra"
OPENROUTER_BATCH = 64
OPENROUTER_WORKERS = 4


class CachedEmbedder:
    """Cache lookup, dedupe and chunked writes; subclasses embed one chunk of new texts."""

    model_name: str
    chunk_size = CHUNK

    def __init__(self, cache_path=CACHE_PATH, instruction=None):
        self.instruction = instruction
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, model TEXT, vec BLOB)"
        )

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()

    def _embed_chunk(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts, show_progress_bar=None, verbose=None, **kwargs) -> np.ndarray:
        texts = list(texts)
        if self.instruction:
            texts = [f"Instruct: {self.instruction}\nQuery:{t}" for t in texts]
        keys = [self._key(t) for t in texts]
        cached = {}
        for i in range(0, len(keys), 900):  # stay under SQLite's variable limit
            chunk = keys[i : i + 900]
            rows = self.db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})", chunk
            )
            cached.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})

        # Dedupe so repeated strings (common among keyphrases) are embedded once.
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cached))
        if todo:
            # Longest first, so a memory or size problem shows up in the first chunk.
            todo.sort(key=len, reverse=True)
            chunks = range(0, len(todo), self.chunk_size)
            for i in tqdm(chunks, desc="embedding", disable=not (show_progress_bar or verbose)):
                batch = todo[i : i + self.chunk_size]
                vecs = self._embed_chunk(batch).astype(np.float32)
                assert vecs.shape[0] == len(batch), (vecs.shape, len(batch))
                rows = [(self._key(t), self.model_name, v.tobytes()) for t, v in zip(batch, vecs)]
                with self.db:
                    self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
                cached.update({row[0]: vec for row, vec in zip(rows, vecs)})

        return np.vstack([cached[k] for k in keys])


class LocalEmbedder(CachedEmbedder):
    def __init__(
        self, model_name=MODEL, cache_path=CACHE_PATH, batch_size=BATCH_SIZE, instruction=None
    ):
        super().__init__(cache_path, instruction)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # loaded on first cache miss, so cached runs stay fast

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"loading {self.model_name} on {device}")
            self._model = SentenceTransformer(
                self.model_name, device=device, model_kwargs={"dtype": torch.float16}
            )
            self._model.max_seq_length = MAX_SEQ_LENGTH
        return self._model

    def _embed_chunk(self, texts: list[str]) -> np.ndarray:
        import torch

        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return vecs


class OpenRouterEmbedder(CachedEmbedder):
    def __init__(
        self,
        model=OPENROUTER_MODEL,
        provider=OPENROUTER_PROVIDER,
        cache_path=CACHE_PATH,
        instruction=None,
    ):
        super().__init__(cache_path, instruction)
        self.model = model
        self.provider = provider
        # Provider in the key: another deployment of the same weights is not the same vectors.
        self.model_name = f"openrouter:{model}@{provider}"
        self.chunk_size = OPENROUTER_BATCH * OPENROUTER_WORKERS
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "HTTP-Referer": "https://github.com/stevenfazzio/deity-map",
                "X-Title": "deity-map",
            }
        )
        self.cost_usd = 0.0

    def _post(self, texts: list[str], max_retries: int = 6) -> np.ndarray:
        body = {
            "model": self.model,
            "input": texts,
            "provider": {"order": [self.provider], "allow_fallbacks": False},
        }
        for attempt in range(max_retries):
            try:
                r = self.session.post(OPENROUTER_URL, json=body, timeout=180)
                if r.status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    wait = min(2**attempt * 5, 120) + random.uniform(0, 3)
                    print(f"  {r.status_code}: {r.text[:200]!r}; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                d = r.json()
                if "error" in d:
                    raise RuntimeError(f"OpenRouter error: {d['error']}")
                self.cost_usd += float(d.get("usage", {}).get("cost") or 0)
                data = sorted(d["data"], key=lambda e: e["index"])
                return np.asarray([e["embedding"] for e in data], dtype=np.float32)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = min(2**attempt * 5, 120)
                print(f"  attempt {attempt + 1}: {e!r}; retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("exhausted retries")

    def _embed_chunk(self, texts: list[str]) -> np.ndarray:
        batches = [texts[i : i + OPENROUTER_BATCH] for i in range(0, len(texts), OPENROUTER_BATCH)]
        with ThreadPoolExecutor(OPENROUTER_WORKERS) as pool:
            return np.vstack(list(pool.map(self._post, batches)))


class ErasedEmbedder:
    """Toponymy's embedder protocol, with a concept eraser applied to every vector."""

    def __init__(self, base: CachedEmbedder, eraser):
        self.base = base
        self.eraser = eraser

    def encode(self, texts, show_progress_bar=None, verbose=None, **kwargs) -> np.ndarray:
        import torch

        vecs = self.base.encode(texts, show_progress_bar=show_progress_bar, verbose=verbose)
        with torch.no_grad():
            return self.eraser(torch.from_numpy(vecs)).numpy().astype(np.float32)
