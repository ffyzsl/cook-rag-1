# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run the system

```bash
python main.py
```

No CLI arguments. The system loads a pre-built FAISS index from `vector_index/` if available, otherwise builds a new one from `data/C8/cook/dishes/`.

## Conda environment

The project uses the `cook-rag-1` conda environment. Key dependencies (no `requirements.txt` exists):

- `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-core`
- `faiss-cpu` (or `faiss` with AVX2 support)
- `dashscope` (Alibaba Cloud Bailian SDK)
- `python-dotenv`

## Architecture

Four-module RAG pipeline orchestrated by `RecipeRAGSystem` in `main.py`:

### Parent-child document pattern

Each recipe `.md` file is a **parent** document. `MarkdownHeaderTextSplitter` splits it at `#`/`##`/`###` headers into **child** chunks. Retrieval searches child chunks (fine-grained matching), then `get_parent_documents()` reconstructs the full parent recipes for LLM context. A `parent_child_map` (child_id → parent_id) is rebuilt every run from the chunk list.

### Module responsibilities

| Module | File | Role |
|---|---|---|
| Data Preparation | `rag_modules/data_preparation.py` | Load `.md` files, extract metadata (category, difficulty, dish name), chunk by Markdown headers |
| Index Construction | `rag_modules/index_construction.py` | Build/load/save FAISS vector index using DashScope embeddings |
| Retrieval Optimization | `rag_modules/retrieval_optimization.py` | Hybrid search: FAISS dense + BM25 sparse, fused via Reciprocal Rank Fusion (RRF, k=60) |
| Generation Integration | `rag_modules/generation_integration.py` | Query routing (list/detail/general), query rewriting, LLM answer generation via ChatTongyi (Qwen) |

### Query pipeline (`ask_question`)

1. `query_router()` classifies query as `list`, `detail`, or `general`
2. `query_rewrite()` optionally refines vague queries via LLM
3. `_extract_filters_from_query()` detects category/difficulty keywords in the query text
4. `hybrid_search()` or `metadata_filtered_search()` retrieves child chunks
5. `get_parent_documents()` maps chunks back to full recipe documents
6. Type-specific answer generation (`generate_list_answer`, `generate_step_by_step_answer`, `generate_basic_answer`)

### Configuration (`config.py`)

`RAGConfig` dataclass with `DEFAULT_CONFIG` global instance:

- `data_path="data/C8/cook"` — recipe Markdown root
- `index_save_path="./vector_index"` — FAISS persistence directory
- `embedding_model="text-embedding-v2"` — DashScope embedding model
- `llm_model="qwen-plus"` — Tongyi Qwen LLM
- `top_k=3`, `temperature=0.1`, `max_tokens=2048`

### Data structure

323 Chinese recipe `.md` files under `data/C8/cook/dishes/` organized into 11 category subdirectories (e.g., `meat_dish/`, `vegetable_dish/`, `soup/`). Category mapping from English dir names to Chinese labels is in `CATEGORY_MAPPING` dict in `data_preparation.py`. Difficulty is parsed from `★` characters in the text.

## Environment

Requires `DASHSCOPE_API_KEY` in `.env` for both embeddings and LLM. The `MOONSHOT_API_KEY` check in `main.py` is vestigial — no module actually uses it.

## Known issues

- **No dependency file** — no `requirements.txt` or `pyproject.toml`. Dependencies must be installed manually.
- **`generate_basic_answer()` and its streaming variant** build a LangChain chain but call `prompt.invoke(query)` directly with a raw string instead of using the chain, which may produce errors since `ChatPromptTemplate` expects a dict with `{question, context}` keys.
- **`from click import prompt`** in `generation_integration.py` is unused.
- **Emoji encoding errors on Windows** — the interactive prompts use emoji (`🍽️`) that will fail on GBK-configured Windows consoles. Use a UTF-8 terminal (VS Code, PyCharm, Windows Terminal with UTF-8 enabled).
