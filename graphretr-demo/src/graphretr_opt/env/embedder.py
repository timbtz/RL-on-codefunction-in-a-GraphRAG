"""QueryEmbedder -- must match the model the nodes were embedded/indexed with.

Two implementations behind the same encode() interface, selected by
cfg.embedder via make_embedder():
  minilm       -- all-MiniLM-L6-v2, 384-d (the original ETL model)
  openai_small -- text-embedding-3-small, 1536-d, metered through OpenAIBudget
                  (valid only after reembed_openai.py has re-indexed the graph)
Both memoize text -> vector so each query embeds exactly once per process.
"""
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384


class QueryEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, max_seq_length: int = 128):
        self._model_name = model_name
        self._max_seq_length = max_seq_length
        self._encoder = None
        self._cache = {}

    def encode(self, text: str) -> np.ndarray:
        vec = self._cache.get(text)
        if vec is None:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._model_name, device="cpu")
                self._encoder.max_seq_length = self._max_seq_length
            vec = self._encoder.encode(
                [text], normalize_embeddings=True, convert_to_numpy=True,
            )[0].astype(np.float32)
            self._cache[text] = vec
        return vec


class OpenAIQueryEmbedder:
    def __init__(self, budget):
        self._budget = budget
        self._cache = {}

    def encode(self, text: str) -> np.ndarray:
        vec = self._cache.get(text)
        if vec is None:
            vec = np.asarray(self._budget.embed([text])[0], dtype=np.float32)
            self._cache[text] = vec
        return vec


def make_embedder(cfg, budget=None):
    if cfg.embedder == "openai_small":
        if budget is None:
            raise ValueError("embedder=openai_small needs an OpenAIBudget")
        return OpenAIQueryEmbedder(budget)
    if cfg.embedder == "minilm":
        return QueryEmbedder()
    raise ValueError(f"unknown embedder {cfg.embedder!r}")
