# RAG-Eval

Evaluation framework for RAG (Retrieval-Augmented Generation) pipelines using [DeepEval](https://github.com/confident-ai/deepeval). Measures retriever quality with LLM-as-a-judge metrics against a curated golden dataset of YouTube transcript Q&A pairs.

---

## Project Structure

```
rag-eval/
├── data/                          # Raw YouTube transcript captions (VTT)
│   ├── YT Sandbox  LLM Evals Session 1.vtt
│   ├── YT Sandbox  LLM Evals Session 2.vtt
│   └── ... (8 sessions)
│
├── goldens/                       # Golden evaluation dataset
│   └── retriever_golden.json      # 15 curated Q&A pairs with ideal answers
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── embeddings.py              # BGE sentence-transformer embedding wrapper
│   └── retriever.py               # Transcript ingestion, chunking, ChromaDB store, retriever
│
├── evals/                         # Evaluation scripts
│   ├── __init__.py
│   └── retriever_evals.py         # DeepEval evaluation pipeline (contextual precision & recall)
│
├── chroma_store/                  # Persisted ChromaDB vector database (auto-generated)
│
├── .deepeval/                     # DeepEval framework state (auto-generated)
│   ├── .deepeval-cache.json
│   ├── .latest_run_full.json
│   └── .latest_test_run.json
│
├── main.py                        # Entry point (stub)
├── pyproject.toml                 # Project metadata & dependencies
├── uv.lock                        # uv lockfile
├── .python-version                # Python 3.13
├── .env                           # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
└── .gitignore
```

---

## How It Works

The pipeline ingests YouTube video transcripts, chunks and embeds them into a vector store, then evaluates retrieval quality against a golden dataset using LLM-as-a-judge metrics.

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   1. INGEST      │     │   2. CHUNK        │     │   3. EMBED &     │
│                  │     │                   │     │   STORE           │
│  data/*.vtt      │────▶│  RecursiveCharac- │────▶│  ChromaDB         │
│  (8 YouTube      │     │  terTextSplitter  │     │  (persisted at   │
│   transcripts)   │     │  size=1000        │     │   chroma_store/) │
│                  │     │  overlap=150      │     │                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   6. RESULTS     │     │   5. EVALUATE     │     │   4. RETRIEVE     │
│                  │     │                   │     │                   │
│  .deepeval/      │◀────│  DeepEval          │◀────│  For each query:  │
│  (scores,        │     │  Contextual       │     │  top-5 nearest    │
│   pass/fail)     │     │  Precision +      │     │  chunks via       │
│                  │     │  Contextual       │     │  cosine similarity│
│                  │     │  Recall           │     │                   │
│                  │     │  (judge: gemma:2b │     │                   │
│                  │     │   via Ollama)     │     │                   │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Key Components

### Embeddings — `src/embeddings.py`

Wraps the `BAAI/bge-small-en-v1.5` sentence-transformer model in a LangChain-compatible `Embeddings` interface. Produces 384-dimensional normalized vectors. The `get_embedding_function()` factory is cached via `@lru_cache` to ensure a single model instance.

```python
from src.embeddings import get_embedding_function

embedding_fn = get_embedding_function()

# Embed documents (batch)
doc_vectors = embedding_fn.embed_documents(["chunk 1", "chunk 2"])

# Embed a single query
query_vector = embedding_fn.embed_query("what is regression testing?")
```

### Retriever — `src/retriever.py`

Handles the full ingestion-to-retrieval pipeline:

1. **`load_transcripts()`** — Reads all `*.vtt` files from `data/`, strips VTT timestamps and headers, joins text, and extracts session numbers from filenames. Returns a list of LangChain `Document` objects with `session` and `source` metadata.

2. **`load_store()`** — If a ChromaDB exists at `chroma_store/`, loads it directly. Otherwise, loads transcripts, splits them into 1000-character chunks (150-character overlap) using `RecursiveCharacterTextSplitter`, and creates the Chroma vector store.

3. **`build_retriever()`** — Wraps the store as a LangChain retriever with `k=5` (top-5 results).

```python
from src.retriever import build_retriever

retriever = build_retriever()
results = retriever.invoke("what is regression testing?")

for doc in results:
    print(f"[Session {doc.metadata['session']}] {doc.page_content[:150]}...")
```

### Evaluation — `evals/retriever_evals.py`

Runs the full evaluation pipeline:

1. Loads 15 Q&A pairs from `goldens/retriever_golden.json`
2. For each query, retrieves top-5 documents from the vector store
3. Constructs DeepEval `LLMTestCase` objects (query, expected answer, retrieved context)
4. Evaluates using two metrics with `gemma:2b` (via Ollama) as the LLM judge:
   - **Contextual Precision** — Are the most relevant documents ranked highest?
   - **Contextual Recall** — Were all relevant documents retrieved?

```python
# Run the evaluation
python evals/retriever_evals.py
```

Output is stored in `.deepeval/` as JSON with per-test-case scores, reasons, and pass/fail verdicts.

### Golden Dataset — `goldens/retriever_golden.json`

15 hand-crafted Q&A pairs derived from the "YT Sandbox - LLM Evals" course. Each entry contains:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (g001–g015) |
| `query` | User question (mix of English and Hindi/Hinglish) |
| `ideal_answer` | Detailed expected answer (gold reference) |
| `source` | Session number the answer is derived from |

Topics covered: online vs offline eval, faithfulness/groundedness, reference-based vs reference-free eval, LLM testing challenges, SQL comparison, benchmark selection, eval execution modes, RAG failure modes, recall@k, RAG triad, regression testing, eval suites, and retriever metrics.

---

## Prerequisites

- **Python 3.13** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Ollama](https://ollama.com/)** running locally with the `gemma:2b` model pulled:
  ```bash
  ollama pull gemma:2b
  ```
- A `.env` file with your API keys:
  ```
  GROQ_API_KEY=your_key_here
  GOOGLE_API_KEY=your_key_here
  ```

## Setup

```bash
# Clone and install
git clone <repo-url>
cd rag-eval
uv sync

# Verify Ollama is running
ollama list
```

## Usage

### Test the Retriever

Runs a sample query against the vector store. Auto-builds the ChromaDB on first run.

```bash
python src/retriever.py
```

### Run the Full Evaluation

Runs all 15 golden Q&A pairs through the retriever and evaluates with DeepEval. Requires Ollama with `gemma:2b`.

```bash
python evals/retriever_evals.py
```

Results are saved to `.deepeval/`. Review with:

```bash
cat .deepeval/.latest_test_run.json
```

---

## Configuration

| Setting | Value | Location |
|---------|-------|----------|
| Embedding model | `BAAI/bge-small-en-v1.5` | `src/embeddings.py:6` |
| Chunk size | 1000 characters | `src/retriever.py:66` |
| Chunk overlap | 150 characters | `src/retriever.py:67` |
| Retriever top-k | 5 | `src/retriever.py:82` |
| Judge model | `gemma:2b` (Ollama) | `evals/retriever_evals.py:15` |
| Judge base URL | `http://localhost:11434` | `evals/retriever_evals.py:16` |
| Judge temperature | 0 (deterministic) | `evals/retriever_evals.py:17` |
| Metric threshold | 0.7 | `evals/retriever_evals.py:19` |
| Async concurrency | 1 (sequential) | `evals/retriever_evals.py:46` |
| Data directory | `./data/` | `src/retriever.py:15` |
| DB directory | `./chroma_store/` | `src/retriever.py:16` |
| Golden dataset | `goldens/retriever_golden.json` | `evals/retriever_evals.py:12` |

---

## Evaluation Metrics

### Contextual Precision

Measures whether the most relevant documents are ranked higher in the retrieval results. A score of 1.0 means every relevant document appears before any irrelevant one.

### Contextual Recall

Measures whether all relevant documents for a given query were retrieved. A score of 1.0 means the retriever found every document needed to fully answer the question.

Both metrics use an LLM judge (`gemma:2b` via Ollama) to compare the retrieved context against the expected answer, with a **0.7 threshold** for pass/fail.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `chromadb` | Vector database for storing and querying embeddings |
| `deepeval` | LLM evaluation framework (pytest-based) |
| `langchain-chroma` | LangChain integration for ChromaDB |
| `langchain-core` | Core LangChain abstractions (Document, Embeddings) |
| `langchain-text-splitters` | Text chunking utilities |
| `sentence-transformers` | BGE embedding model inference |
| `ollama` | Ollama client for local LLM judge |
| `python-dotenv` | `.env` file loading |
| `pytest` | Testing framework |
| `google-genai`, `langchain-groq`, `langchain-openai`, `openai` | Reserved for future generator/RAG-chain components |

---

## Project Status

**Current state:** Retriever evaluation only — no generator/LLM component yet.

- [x] Transcript ingestion and preprocessing
- [x] Text chunking and embedding
- [x] ChromaDB vector store with persistence
- [x] Retriever with top-5 semantic search
- [x] Golden dataset (15 Q&A pairs)
- [x] DeepEval evaluation pipeline (contextual precision & recall)
- [ ] Generator component (LLM-based answer generation)
- [ ] Full RAG triad evaluation (context relevance, faithfulness, answer relevance)
- [ ] `main.py` entry point wired to full pipeline
- [ ] Automated test suite (`pytest`)
