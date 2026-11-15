"""Shared configuration and the embedding model wrapper (Group-12, Assignment-2)."""
import os, shutil, pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).parent
DATA_CSV = ROOT / "data" / "festival_feedback.csv"
CHROMA_DIR = ROOT / "chroma_store"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
COLLECTION = "festival_feedback"
EMBED_MODEL_NAME = "WordLlama l2_supercat (256-d)"
EMBED_DIM = 256


class Embedder:
    """WordLlama static-embedding model (Llama-2 token embeddings distilled for
    sentence similarity). Weights ship inside the pip wheel, so the model loads
    fully offline on Proyogshala / a laptop with no GPU.

    NOTE: WordLlama returns *un-normalised* vectors; we keep them raw so that the
    Part-B metric study (L2 vs cosine vs dot product) can show the effect of
    normalisation explicitly.
    """

    def __init__(self, cache_dir="~/.cache/wordllama_offline"):
        import wordllama
        from wordllama import WordLlama
        cache = pathlib.Path(cache_dir).expanduser()
        pkg = pathlib.Path(wordllama.__file__).parent
        for sub in ("tokenizers", "weights"):  # copy bundled files so no download is attempted
            shutil.copytree(pkg / sub, cache / sub, dirs_exist_ok=True)
        self.model = WordLlama.load(cache_dir=cache, dim=EMBED_DIM, disable_download=True)

    def encode(self, texts, batch_size=256):
        out = [self.model.embed(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(out).astype(np.float32)


class ChromaEF:
    """Adapter so ChromaDB can call our embedder for query_texts."""

    def __init__(self, emb):
        self.emb = emb

    def __call__(self, input):
        return self.emb.encode(list(input)).tolist()

    def name(self):
        return "wordllama_l2_supercat_256"


def get_collection(emb=None, create=False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        return client, client.create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine", "embedding_model": EMBED_MODEL_NAME,
                                  "description": "FestiVox attendee reviews, incident reports and staff debriefs"})
    return client, client.get_collection(COLLECTION)


def dump(name, obj):
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)
