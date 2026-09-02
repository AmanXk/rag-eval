# RAG-Eval

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![DeepEval](https://img.shields.io/badge/evaluation-DeepEval-red)](https://github.com/confident-ai/deepeval)
[![ChromaDB](https://img.shields.io/badge/vector--store-ChromaDB-green)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/judge-Ollama-gray?logo=ollama&logoColor=white)](https://ollama.com/)

> RAG retriever evaluation using LLM-as-a-judge — ingest YouTube transcripts, build a vector store, and measure retrieval quality with DeepEval.

---

### Feature Highlights

- **Transcript Ingestion** — Parse VTT captions from YouTube into clean documents
- **Semantic Chunking** — Split text with configurable size/overlap via LangChain
- **BGE Embeddings** — `BAAI/bge-small-en-v1.5` sentence-transformer (384-dim normalized)
- **ChromaDB Vector Store** — Persistent local vector database with cosine similarity search
- **Golden Dataset** — 15 curated Q&A pairs covering LLM evaluation topics
- **DeepEval Evaluation** — Contextual Precision & Recall metrics with LLM judge (`gemma:2b` via Ollama)

---

### Quick Start

```bash
# 1. Clone
git clone https://github.com/AmanXk/rag-eval.git
cd rag-eval

# 2. Install dependencies
uv sync

# 3. Pull the judge model for Ollama
ollama pull gemma:2b

# 4. Run the evaluation
python evals/retriever_evals.py
```

First run auto-downloads the embedding model and builds the ChromaDB store. Results are saved to `.deepeval/`.

---

### How It Works

```
 data/*.vtt           RecursiveTextSplitter       ChromaDB
 (8 transcripts)  ──▶  (size=1000, overlap=150)  ──▶  (persisted)
                                                    │
                                                    ▼
 .deepeval/            DeepEval                     retriever.invoke()
 (scores, results) ◀──  ContextualPrecision   ◀──  top-5 chunks per query
                        + ContextualRecall
                        (judge: gemma:2b via Ollama)
```

1. **Ingest** — VTT files are parsed, timestamps stripped, session numbers extracted
2. **Chunk** — Documents split into 1000-char chunks (150 overlap)
3. **Embed** — BGE model produces normalized 384-dim vectors
4. **Store** — ChromaDB persists embeddings to `chroma_store/`
5. **Retrieve** — For each golden query, fetch top-5 nearest chunks
6. **Evaluate** — DeepEval computes contextual precision & recall with LLM judge

---

### Key Components

| File | Purpose |
| ------ | --------- |
| `src/embeddings.py` | BGE embedding wrapper (LangChain-compatible) |
| `src/retriever.py` | Transcript loading, chunking, ChromaDB store, retriever builder |
| `evals/retriever_evals.py` | DeepEval evaluation pipeline |
| `goldens/retriever_golden.json` | 15 golden Q&A pairs for evaluation |
| `main.py` | Entry point (planned) |

<details>
<summary><strong>Embeddings — src/embeddings.py</strong></summary>

Wraps `BAAI/bge-small-en-v1.5` in a LangChain `Embeddings` interface. Cached singleton via `@lru_cache`.

```python
from src.embeddings import get_embedding_function

embedding_fn = get_embedding_function()
doc_vectors = embedding_fn.embed_documents(["chunk 1", "chunk 2"])
query_vector = embedding_fn.embed_query("what is regression testing?")
```

</details>

<details>
<summary><strong>Retriever — src/retriever.py</strong></summary>

- `load_transcripts()` — Reads `*.vtt`, strips timestamps, extracts session metadata
- `load_store()` — Creates or loads ChromaDB (auto-builds on first run)
- `build_retriever()` — Returns LangChain retriever with `k=5`

```python
from src.retriever import build_retriever

retriever = build_retriever()
results = retriever.invoke("what is regression testing?")
for doc in results:
    print(f"[Session {doc.metadata['session']}] {doc.page_content[:150]}...")
```

</details>

<details>
<summary><strong>Evaluation — evals/retriever_evals.py</strong></summary>

Loads golden Q&A pairs, retrieves context, and evaluates with two metrics:

- **Contextual Precision** — Are relevant docs ranked highest?
- **Contextual Recall** — Were all relevant docs found?

```bash
python evals/retriever_evals.py
```

Output: `.deepeval/.latest_test_run.json`

</details>

---

<details>
<summary><strong>Configuration</strong></summary>

| Setting | Value | File:Line |
| --------- | ------- | ----------- |
| Embedding model | `BAAI/bge-small-en-v1.5` | `src/embeddings.py:6` |
| Chunk size | 1000 chars | `src/retriever.py:66` |
| Chunk overlap | 150 chars | `src/retriever.py:67` |
| Retriever top-k | 5 | `src/retriever.py:82` |
| Judge model | `gemma:2b` (Ollama) | `evals/retriever_evals.py:15` |
| Judge base URL | `http://localhost:11434` | `evals/retriever_evals.py:16` |
| Judge temperature | 0 | `evals/retriever_evals.py:17` |
| Metric threshold | 0.7 | `evals/retriever_evals.py:19` |
| Async concurrency | 1 | `evals/retriever_evals.py:46` |

</details>

<details>
<summary><strong>Dependencies</strong></summary>

| Package | Purpose |
| --------- | --------- |
| `chromadb` | Vector database |
| `deepeval` | LLM evaluation framework |
| `langchain-chroma` | LangChain ChromaDB integration |
| `langchain-core` | Core abstractions (Document, Embeddings) |
| `langchain-text-splitters` | Text chunking |
| `sentence-transformers` | BGE embedding inference |
| `ollama` | Ollama client for local LLM judge |
| `python-dotenv` | `.env` loading |
| `google-genai`, `langchain-groq`, `langchain-openai`, `openai` | Reserved for future RAG chain |

</details>

---

### Roadmap

- [x] Transcript ingestion and preprocessing
- [x] Text chunking and embedding
- [x] ChromaDB vector store with persistence
- [x] Retriever with top-5 semantic search
- [x] Golden dataset (15 Q&A pairs)
- [x] DeepEval evaluation pipeline (contextual precision & recall)
- [ ] Generator component (LLM-based answer generation)
- [ ] Full RAG triad evaluation (context relevance, faithfulness, answer relevance)
- [ ] `main.py` wired to full pipeline
- [ ] Automated test suite (`pytest`)

---

### Contributing

Contributions welcome! Please open an issue or submit a PR.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

### License

This project is for educational purposes.
