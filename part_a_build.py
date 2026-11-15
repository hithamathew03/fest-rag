"""Part A - preprocessing, embedding generation, persistent ChromaDB storage and CRUD demo."""
import time, re
import pandas as pd, numpy as np
from common import *


def preprocess(df):
    n0 = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    df = df[df["text"].str.len() >= 30]                      # drop too-short / empty records
    df = df.drop_duplicates(subset=["text", "festival_id"])  # exact duplicate removal
    for c in ["festival_id", "month", "rating"]:
        df[c] = df[c].astype(int)
    for c in [c for c in df.columns if df[c].dtype == object]:
        df[c] = df[c].fillna("NA").astype(str)
    return df.reset_index(drop=True), n0


def main():
    log = {}
    df, n0 = preprocess(pd.read_csv(DATA_CSV, keep_default_na=False))
    log["records_raw"], log["records_clean"] = n0, len(df)
    log["doc_type_counts"] = df.doc_type.value_counts().to_dict()
    log["festival_counts"] = df.festival_name.value_counts().to_dict()
    log["aspect_counts"] = df.aspect.value_counts().to_dict()
    log["rating_counts"] = df[df.doc_type == "attendee_review"].rating.value_counts().sort_index().to_dict()
    log["avg_chars"] = float(df.text.str.len().mean())
    log["sample_records"] = df.sample(5, random_state=3).to_dict("records")

    emb = Embedder()
    t = time.time()
    X = emb.encode(df.text.tolist())
    log["embed_seconds"] = round(time.time() - t, 2)
    log["embedding_shape"] = list(X.shape)
    norms = np.linalg.norm(X, axis=1)
    log["norm_stats"] = dict(min=float(norms.min()), mean=float(norms.mean()), max=float(norms.max()))
    np.save(RESULTS / "embeddings.npy", X)
    df[["doc_id"]].to_csv(RESULTS / "embedding_ids.csv", index=False)

    client, col = get_collection(create=True)
    meta_cols = [c for c in df.columns if c not in ("doc_id", "text")]
    t = time.time()
    B = 1000
    for i in range(0, len(df), B):
        part = df.iloc[i:i + B]
        col.add(ids=part.doc_id.tolist(), documents=part.text.tolist(),
                embeddings=X[i:i + B].tolist(), metadatas=part[meta_cols].to_dict("records"))
    log["insert_seconds"] = round(time.time() - t, 2)
    log["collection_count"] = col.count()
    log["collection_metadata"] = col.metadata

    # ------------------------------------------------------------ CRUD demonstration
    crud = {}
    new_id = "FB-05-NEW001"
    new_doc = ("NimbusPay wristband refund finally arrived 3 weeks after Riverside Fest, but only after I raised a "
               "complaint on social media.")
    new_meta = dict(doc_type="attendee_review", festival_id=5, festival_name="Vibe Collective Riverside Fest",
                    organizer="Vibe Collective", city="Delhi", venue="Riverside Grounds", stage="River Stage",
                    artist="Night Owl Collective", genre="Indie", event_date="2026-02-02", month=2,
                    ticket_category="General", rating=2, sentiment="negative", aspect="cashless_payment",
                    severity="NA", author_role="attendee")
    # CREATE
    col.add(ids=[new_id], documents=[new_doc], metadatas=[new_meta], embeddings=emb.encode([new_doc]).tolist())
    crud["count_after_insert"] = col.count()
    # READ (by id, and by metadata)
    crud["read_by_id"] = col.get(ids=[new_id], include=["documents", "metadatas"])
    r = col.get(where={"$and": [{"festival_id": 5}, {"aspect": "cashless_payment"}]}, include=["metadatas"])
    crud["read_by_metadata_count"] = len(r["ids"])
    # UPDATE (document text + metadata; embedding recomputed)
    upd_doc = new_doc + " Update: the organiser has now also given a free pass for next year as an apology."
    col.update(ids=[new_id], documents=[upd_doc], embeddings=emb.encode([upd_doc]).tolist(),
               metadatas=[{**new_meta, "rating": 3, "sentiment": "mixed"}])
    crud["after_update"] = col.get(ids=[new_id], include=["documents", "metadatas"])
    # UPSERT (insert-or-update semantics)
    col.upsert(ids=[new_id], documents=[upd_doc], embeddings=emb.encode([upd_doc]).tolist(),
               metadatas=[{**new_meta, "rating": 3, "sentiment": "mixed", "severity": "NA"}])
    # DELETE
    col.delete(ids=[new_id])
    crud["after_delete"] = col.get(ids=[new_id])
    crud["count_after_delete"] = col.count()
    # bulk delete by metadata, then restore (shows where-based delete)
    victims = col.get(where={"$and": [{"festival_id": 10}, {"doc_type": "staff_debrief"}]},
                      include=["documents", "metadatas", "embeddings"])
    col.delete(where={"$and": [{"festival_id": 10}, {"doc_type": "staff_debrief"}]})
    crud["bulk_delete_removed"] = len(victims["ids"])
    crud["count_after_bulk_delete"] = col.count()
    col.add(ids=victims["ids"], documents=victims["documents"], metadatas=victims["metadatas"],
            embeddings=[list(map(float, e)) for e in victims["embeddings"]])
    crud["count_after_restore"] = col.count()
    log["crud"] = crud
    dump("part_a.json", log)
    print({k: v for k, v in log.items() if k not in ("sample_records", "crud")})
    print("CRUD:", {k: v for k, v in crud.items() if "count" in k or "removed" in k})


if __name__ == "__main__":
    main()
