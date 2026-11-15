"""End-to-end run: dataset -> ChromaDB -> Part B experiments -> RAG -> Group-12.pdf
Usage:  python run_all.py            (Ollama llama3.2:3b - run `ollama pull llama3.2:3b` first)
        python run_all.py hf         (Hugging Face Qwen2.5-1.5B-Instruct instead of Ollama)
"""
import subprocess, sys
backend = sys.argv[1] if len(sys.argv) > 1 else "ollama"
steps = [["generate_dataset.py"], ["part_a_build.py"], ["part_b_search.py", "5"], ["part_c_rag.py", "dummy"],
         ["part_c_rag.py", backend], ["build_report.py"]]
for s in steps:
    print("\n=== python", " ".join(s), flush=True)
    subprocess.run([sys.executable] + s, check=True)
