"""Part C - Retrieval-Augmented Generation over the ChromaDB festival-feedback store.

Pipeline: query -> query embedding (same WordLlama model) -> top-K ChromaDB retrieval
(with the query's metadata filter) -> context preparation -> grounded prompt -> LLM answer.

LLM backends
  ollama (default) : local Ollama server, model llama3.2:3b   ->  ollama pull llama3.2:3b
  hf               : Hugging Face transformers, Qwen/Qwen2.5-1.5B-Instruct (CPU OK, slower)
  dummy            : echoes the prompt - ONLY for testing the plumbing; never used in the report

Usage:  python part_c_rag.py [ollama|hf|dummy] [model-name]
"""
import sys, re, time, json, urllib.request
from common import *
from queries import QUERIES, KEY_FACTS

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "ollama"
MODEL = sys.argv[2] if len(sys.argv) > 2 else {"ollama": "llama3.2:3b", "hf": "Qwen/Qwen2.5-1.5B-Instruct", "dummy": "dummy"}[BACKEND]
TOP_K = 6                 # passages placed in the prompt
N_CANDIDATES = 30         # passages fetched from ChromaDB before diversification
MMR_LAMBDA = 0.5          # 1.0 = pure relevance, 0.0 = pure diversity
MIN_FILTERED = 3          # if a filter leaves fewer hits than this, fall back to unfiltered retrieval
MAX_CTX_CHARS = 3000

SYSTEM_RAG = (
    "You are FestiVox, an analyst assistant for music-festival organisers. Answer the question using ONLY the "
    "feedback records in the CONTEXT. Rules:\n"
    "1. Every factual statement must cite its source record id in square brackets, e.g. [FB-05-00123].\n"
    "2. Do not use outside knowledge and do not invent numbers, names or dates that are not in the context.\n"
    "3. If the context does not contain enough information to answer, say exactly: "
    "\"The retrieved records do not contain enough information to answer this.\" and state what is missing.\n"
    "4. Mention which festival(s) each point comes from. Keep the answer under 150 words, as short bullet points.")
SYSTEM_PLAIN = ("You are a helpful assistant for music-festival organisers in India. Answer the question concisely "
                "in under 150 words.")

EXTRA_PROBE = dict(qid="X1", student="(test)", text="How many tickets were sold for Crescendo Classical and what was the total revenue?",
                   where={"festival_name": "Crescendo Classical"})


def llm(system, user):
    t = time.time()
    if BACKEND == "ollama":
        body = json.dumps({"model": MODEL, "stream": False, "options": {"temperature": 0, "seed": 12, "num_ctx": 4096},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
        text = json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"]
    elif BACKEND == "hf":
        global _pipe
        if "_pipe" not in globals():
            from transformers import pipeline
            _pipe = pipeline("text-generation", model=MODEL, torch_dtype="auto", device_map="auto")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = _pipe(msgs, max_new_tokens=300, do_sample=False)[0]["generated_text"][-1]["content"]
    else:
        text = "[DUMMY BACKEND - plumbing test only] " + user[-200:]
    return text.strip(), round(time.time() - t, 1)


def build_context(res):
    blocks, used = [], 0
    for i, d, m, doc in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]):
        b = (f"[{i}] ({m['doc_type']}; {m['festival_name']}, {m['city']}; stage: {m['stage']}; date: {m['event_date']}; "
             f"rating: {m['rating'] or 'n/a'}; ticket: {m['ticket_category']})\n{doc}")
        if used + len(b) > MAX_CTX_CHARS:
            break
        blocks.append(b); used += len(b)
    return "\n\n".join(blocks)


def fact_coverage(qid, ans):
    if qid not in KEY_FACTS:
        return None
    a = ans.lower()
    return round(sum(bool(re.search(f, a)) for f in KEY_FACTS[qid]) / len(KEY_FACTS[qid]), 2)


def mmr_select(qv, res, k, lam=MMR_LAMBDA):
    """Maximal Marginal Relevance: many reviews repeat the same complaint, so we pick
    passages that are relevant to the query but not near-duplicates of each other."""
    import numpy as np
    E = np.array(res["embeddings"][0], dtype=np.float32)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    q = qv / np.linalg.norm(qv)
    rel = E @ q
    chosen = [int(np.argmax(rel))]
    while len(chosen) < min(k, len(E)):
        red = (E @ E[chosen].T).max(1)
        score = lam * rel - (1 - lam) * red
        score[chosen] = -1e9
        chosen.append(int(np.argmax(score)))
    return {key: [[res[key][0][i] for i in chosen]] for key in ("ids", "distances", "metadatas", "documents")}


def retrieve(col, emb, text, where):
    qv = emb.encode([text])[0]
    res = col.query(query_embeddings=[qv.tolist()], n_results=N_CANDIDATES, where=where,
                    include=["documents", "metadatas", "distances", "embeddings"])
    return res, qv


def rag(col, emb, Q):
    res, qv = retrieve(col, emb, Q["text"], Q.get("where"))
    used_filter = True
    if len(res["ids"][0]) < MIN_FILTERED:
        res, qv = retrieve(col, emb, Q["text"], None); used_filter = False
    res = mmr_select(qv, res, TOP_K)
    ctx = build_context(res)
    prompt = f"CONTEXT:\n{ctx}\n\nQUESTION: {Q['text']}\n\nANSWER (with [record-id] citations):"
    ans, secs = llm(SYSTEM_RAG, prompt)
    retrieved = res["ids"][0]
    cited = sorted(set(re.findall(r"FB-\d{2}-[A-Z0-9]+", ans)))
    return dict(retrieved=[dict(id=i, distance=round(d, 4), festival=m["festival_name"], excerpt=doc[:160])
                           for i, d, m, doc in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0])],
                used_filter=used_filter, prompt=SYSTEM_RAG + "\n\n" + prompt, answer=ans, seconds=secs, cited=cited,
                cited_valid=[c for c in cited if c in retrieved], cited_invalid=[c for c in cited if c not in retrieved],
                fact_coverage=fact_coverage(Q["qid"], ans),
                says_insufficient="not contain enough information" in ans.lower())


def main():
    emb = Embedder()
    _, col = get_collection()
    out = dict(backend=BACKEND, model=MODEL, top_k=TOP_K, n_candidates=N_CANDIDATES, mmr_lambda=MMR_LAMBDA, system_prompt_rag=SYSTEM_RAG, system_prompt_plain=SYSTEM_PLAIN, queries=[])
    for Q in QUERIES + [EXTRA_PROBE]:
        r = rag(col, emb, Q)
        plain, secs = llm(SYSTEM_PLAIN, Q["text"])
        r.update(qid=Q["qid"], student=Q["student"], text=Q["text"], no_context_answer=plain, no_context_seconds=secs,
                 no_context_fact_coverage=fact_coverage(Q["qid"], plain))
        out["queries"].append(r)
        print(f"{Q['qid']}: RAG {r['seconds']}s cov={r['fact_coverage']} cited={len(r['cited_valid'])}/{len(r['cited'])} | "
              f"plain cov={r['no_context_fact_coverage']}")
    dump(f"part_c_{BACKEND}.json", out)


if __name__ == "__main__":
    main()
