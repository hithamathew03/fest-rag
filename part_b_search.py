"""Part B - semantic search, metadata filtering, and exhaustive NumPy search with L2 / cosine / dot."""
import sys
import numpy as np, pandas as pd
from common import *
import re
from queries import QUERIES, KEY_FACTS

K_DEFAULT = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 5  # configurable K

# Relevance proxy used only for analysis: a hit is "relevant" if its aspect is one
# the question is about (and, where the question is festival-specific, the festival matches).
RELEVANCE = {
    "Q1": dict(aspects={"cashless_payment"}), "Q2": dict(aspects={"weather", "crowd_safety", "performance", "connectivity"}, fest={2}),
    "Q3": dict(aspects={"accessibility"}), "Q4": dict(aspects={"sound"}),
    "Q5": dict(aspects={"crowd_safety"}, fest={7}), "Q6": dict(aspects={"vip_experience"}),
    "Q7": dict(aspects={"transport"}), "Q8": dict(aspects={"performance"}),
    "Q9": dict(aspects={"weather"}, fest={3}), "Q10": dict(aspects={"food_beverage"}),
}


def is_rel(qid, m):
    r = RELEVANCE[qid]
    return (m["aspect"] in r["aspects"] or m.get("aspect2") in r["aspects"]) and ("fest" not in r or int(m["festival_id"]) in r["fest"])


def fact_recall(qid, texts):
    """Share of the query's ground-truth key facts that appear somewhere in the retrieved texts."""
    blob = " ".join(texts).lower()
    return round(sum(bool(re.search(f, blob)) for f in KEY_FACTS[qid]) / len(KEY_FACTS[qid]), 2)


def fmt_hits(res, qid):
    out = []
    for rank, (i, d, m, doc) in enumerate(zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]), 1):
        out.append(dict(rank=rank, id=i, distance=round(float(d), 4), festival=m["festival_name"], aspect=m["aspect"],
                        doc_type=m["doc_type"], rating=m["rating"], ticket=m["ticket_category"], genre=m["genre"],
                        relevant=is_rel(qid, m), text=doc))
    return out


def semantic_search(col, emb, text, k=K_DEFAULT, where=None):
    """Top-K semantic search in ChromaDB (HNSW, cosine space); K is configurable."""
    qv = emb.encode([text]).tolist()
    return col.query(query_embeddings=qv, n_results=k, where=where, include=["documents", "metadatas", "distances"])


def exhaustive(X, q, metric, k):
    """Brute-force search over all stored vectors. Returns (indices, scores, kind)."""
    if metric == "l2":
        s = np.sqrt(((X - q) ** 2).sum(1)); idx = np.argsort(s)[:k]; kind = "distance (lower=closer)"
    elif metric == "cosine":
        sim = (X @ q) / (np.linalg.norm(X, axis=1) * np.linalg.norm(q))
        s = 1 - sim; idx = np.argsort(s)[:k]; kind = "cosine distance = 1 - cos (lower=closer)"
    elif metric == "dot":
        s = X @ q; idx = np.argsort(-s)[:k]; kind = "similarity (higher=closer)"
    return idx, s[idx], kind


def main():
    emb = Embedder()
    _, col = get_collection()
    meta = col.get(include=["metadatas", "documents"])
    id2m = dict(zip(meta["ids"], meta["metadatas"]))
    id2d = dict(zip(meta["ids"], meta["documents"]))
    X = np.load(RESULTS / "embeddings.npy")
    ids = pd.read_csv(RESULTS / "embedding_ids.csv").doc_id.tolist()
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

    # keyword baseline (TF-IDF, exact token match) to justify vector retrieval
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs_in_order = [id2d[i] for i in ids]
    tfv = TfidfVectorizer(stop_words="english"); T = tfv.fit_transform(docs_in_order)

    out = {"k_default": K_DEFAULT, "queries": []}
    for Q in QUERIES:
        qid = Q["qid"]
        r_plain = fmt_hits(semantic_search(col, emb, Q["text"], K_DEFAULT), qid)
        r_filt = fmt_hits(semantic_search(col, emb, Q["text"], K_DEFAULT, Q["where"]), qid)
        n_match = len(col.get(where=Q["where"], include=[])["ids"])
        qv = emb.encode([Q["text"]])[0]
        qn = qv / np.linalg.norm(qv)
        metrics = {}
        for norm_label, XX, qq in (("raw", X, qv), ("normalized", Xn, qn)):
            for k in (5, 10):
                for m in ("l2", "cosine", "dot"):
                    idx, sc, kind = exhaustive(XX, qq, m, k)
                    hits = [dict(rank=r + 1, id=ids[i], score=round(float(s), 4), aspect=id2m[ids[i]]["aspect"],
                                 festival=id2m[ids[i]]["festival_name"], relevant=is_rel(qid, id2m[ids[i]]),
                                 chars=len(id2d[ids[i]]), norm=round(float(np.linalg.norm(X[i])), 3))
                            for r, (i, s) in enumerate(zip(idx, sc))]
                    metrics[f"{norm_label}|k{k}|{m}"] = dict(kind=kind, hits=hits,
                                                            fact_recall=fact_recall(qid, [id2d[ids[i]] for i in idx]),
                                                            precision=round(np.mean([h["relevant"] for h in hits]), 2))
        # overlap between metrics
        ov = {}
        for nl in ("raw", "normalized"):
            for k in (5, 10):
                S = {m: [h["id"] for h in metrics[f"{nl}|k{k}|{m}"]["hits"]] for m in ("l2", "cosine", "dot")}
                ov[f"{nl}|k{k}"] = {
                    "l2_vs_cos": len(set(S["l2"]) & set(S["cosine"])), "cos_vs_dot": len(set(S["cosine"]) & set(S["dot"])),
                    "l2_vs_dot": len(set(S["l2"]) & set(S["dot"])),
                    "identical_order_all3": S["l2"] == S["cosine"] == S["dot"]}
        tf_scores = (T @ tfv.transform([Q["text"]]).T).toarray().ravel()
        tf_idx = np.argsort(-tf_scores)[:5]
        keyword = [dict(rank=r + 1, id=ids[i], score=round(float(tf_scores[i]), 4), aspect=id2m[ids[i]]["aspect"],
                        relevant=is_rel(qid, id2m[ids[i]]), text=id2d[ids[i]]) for r, i in enumerate(tf_idx)]
        # ChromaDB HNSW (cosine) vs exact cosine
        chroma10 = semantic_search(col, emb, Q["text"], 10)["ids"][0]
        exact10 = [h["id"] for h in metrics["raw|k10|cosine"]["hits"]]
        out["queries"].append(dict(
            **{k: Q[k] for k in ("qid", "student", "bits", "text", "why")}, where=Q["where"],
            filter_matches=n_match, semantic=r_plain, filtered=r_filt,
            precision_semantic=round(np.mean([h["relevant"] for h in r_plain]), 2),
            precision_filtered=round(np.mean([h["relevant"] for h in r_filt]), 2) if r_filt else None,
            fact_recall_keyword=fact_recall(qid, [h["text"] for h in keyword]),
            fact_recall_semantic=fact_recall(qid, [h["text"] for h in r_plain]),
            fact_recall_filtered=fact_recall(qid, [h["text"] for h in r_filt]),
            keyword=keyword, precision_keyword=round(np.mean([h["relevant"] for h in keyword]), 2),
            metrics=metrics, overlap=ov, query_norm=round(float(np.linalg.norm(qv)), 3),
            hnsw_recall_at10=len(set(chroma10) & set(exact10)) / 10))
        print(qid, "P@5 semantic", out["queries"][-1]["precision_semantic"], "filtered", out["queries"][-1]["precision_filtered"],
              "| raw k10 overlap", ov["raw|k10"], "| norm identical", ov["normalized|k10"]["identical_order_all3"])

    # corpus-level stats used in the normalisation discussion
    nrm = np.linalg.norm(X, axis=1)
    lens = np.array([len(id2d[i]) for i in ids])
    out["norm_vs_length_corr"] = round(float(np.corrcoef(nrm, lens)[0, 1]), 3)
    out["norm_stats"] = dict(min=float(nrm.min()), max=float(nrm.max()), mean=float(nrm.mean()), std=float(nrm.std()))
    dump("part_b.json", out)


if __name__ == "__main__":
    main()
