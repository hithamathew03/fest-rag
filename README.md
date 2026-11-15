# FestiVox — Group-12, Assignment-2 (Vector DB + RAG)

Attendee-feedback intelligence for the Assignment-1 Music Festival & Concert Management System.
Python + ChromaDB (persistent) + WordLlama embeddings + NumPy exhaustive search + Ollama LLM.

## Run (≈10–15 min, CPU is fine)
```bash
pip install -r requirements.txt
ollama pull llama3.2:3b        # Ollama must be running (ollama serve)
python run_all.py              # or: python run_all.py hf
```
Produces `data/festival_feedback.csv` (5,200 records), `chroma_store/`, `results/*.json`, and **Group-12.pdf**.

| File | Part |
|---|---|
| generate_dataset.py | A – dataset (grounded in Assignment-1 entities) |
| part_a_build.py | A – preprocessing, embeddings, ChromaDB load, CRUD |
| queries.py | 10 queries (2 per student) + filters |
| part_b_search.py [K] | B – semantic/filtered search, TF-IDF baseline, L2/cosine/dot for K=5,10 |
| part_c_rag.py [ollama\|hf] | C – RAG + with/without-context comparison |
| build_report.py | rebuilds Group-12.pdf from results |
| FestiVox_Group12.ipynb | notebook version |
