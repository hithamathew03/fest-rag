# FestiVox — Group-12, Modern Database Systems Assignment-2

FestiVox is an attendee-feedback intelligence system for the Music Festival & Concert Management application. It uses Python, ChromaDB, WordLlama embeddings, NumPy-based exhaustive similarity search, and an Ollama LLM for Retrieval-Augmented Generation (RAG).

## Project contents

- `generate_dataset.py` — generates the 5,200-record festival feedback dataset.
- `part_a_build.py` — preprocesses the dataset, generates embeddings, creates the persistent ChromaDB collection, and demonstrates CRUD operations.
- `queries.py` — defines the 10 information-seeking queries and their metadata filters.
- `part_b_search.py` — performs semantic and metadata-filtered retrieval, keyword baseline comparison, and exhaustive L2, cosine, and dot-product experiments for K=5 and K=10.
- `part_c_rag.py` — runs RAG and compares answers generated with and without retrieved context.
- `common.py` — shared paths, embedding model wrapper, and ChromaDB configuration.
- `run_all.py` — runs the complete workflow.
- `build_report.py` — generates `Group-12.pdf` from the experiment results.
- `FestiVox_Group12.ipynb` — notebook version of the project.
- `data/festival_feedback.csv` — generated dataset (if included in the submitted ZIP).
- `results/` — saved experiment outputs and supporting figures (if included).

## Requirements

- Python 3.12 (the project was prepared using Python 3.12.3).
- Internet access may be needed during initial Python package installation.
- Ollama installed for the local LLM workflow.
- Sufficient disk space for the generated ChromaDB database, embeddings, and experiment outputs.

## Setup

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

## Run the complete project

The submitted lightweight project package may omit the generated `chroma_store/` directory and `results/embeddings.npy` to remain within the upload-size limit. These are generated files, not source files. The full pipeline recreates them from the dataset.

For the Ollama workflow, install/start Ollama and download the model:

```bash
ollama pull llama3.2:3b
ollama serve
```

Keep Ollama running, open another terminal in the project directory, activate the virtual environment, and execute:

```bash
python run_all.py
```

The script runs the following stages in order:

1. Generates the dataset using `generate_dataset.py`.
2. Builds embeddings and the persistent ChromaDB collection using `part_a_build.py`.
3. Runs Part B retrieval and similarity-metric experiments.
4. Runs the dummy-backend RAG demonstration.
5. Runs the Ollama RAG demonstration and the with/without-context comparison.
6. Builds the final `Group-12.pdf` report.

The complete run can take several minutes depending on the computer and whether the embedding model and Ollama model are already available.

## Optional Hugging Face LLM backend

The code also supports the Hugging Face backend. Install the optional dependencies:

```bash
pip install transformers torch accelerate
```

Then run:

```bash
python run_all.py hf
```

This uses the Hugging Face backend configured in `part_c_rag.py` instead of Ollama for the final RAG run. The Hugging Face model may require additional downloads, memory, and processing time.

## Generated files

After a successful full run, the project creates or refreshes:

- `data/festival_feedback.csv` — generated dataset.
- `chroma_store/` — persistent ChromaDB database.
- `results/embeddings.npy` — document embedding matrix used by the exhaustive NumPy experiments.
- `results/embedding_ids.csv` — mapping between embedding rows and document IDs.
- `results/part_a.json` — Part A dataset, embedding, database, and CRUD results.
- `results/part_b.json` — Part B retrieval and metric-comparison results.
- `results/part_c_dummy.json` — dummy-backend RAG output.
- `results/part_c_ollama.json` — Ollama RAG output (or `part_c_hf.json` when using the Hugging Face backend).
- `Group-12.pdf` — generated report.

**Important:** Do not delete `chroma_store/` or `results/embeddings.npy` before running Part B or Part C individually. If these generated files are absent, run the complete pipeline with `python run_all.py` first so that Part A recreates them.

## Running individual parts

Run these commands from the project directory, after completing the required earlier stages:

```bash
python generate_dataset.py
python part_a_build.py
python part_b_search.py
python part_c_rag.py ollama
python build_report.py
```

`part_b_search.py` requires the ChromaDB collection and `results/embeddings.npy` created by Part A. `part_c_rag.py` requires the ChromaDB collection. The Ollama backend additionally requires the configured Ollama model to be installed and the Ollama service to be running.

## Notes

- The ChromaDB database and embedding matrix are intentionally omitted from a size-optimized source submission when necessary; they can be regenerated by running the complete pipeline.
- The experiment JSON files and figures included in the submission are saved outputs from the recorded experiments. Running the pipeline again regenerates the corresponding outputs.
- The report should be reviewed after regeneration to ensure that its results correspond to the latest execution.
