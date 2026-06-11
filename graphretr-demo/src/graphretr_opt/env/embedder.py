"""QueryEmbedder -- the same model/settings the ETL embedded nodes with
(all-MiniLM-L6-v2, 384-d, normalized, max_seq_length 128), loaded lazily,
with a text -> vector memo so each query embeds exactly once per process.
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
