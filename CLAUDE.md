# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install from source (development)
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_faithfulness.py

# Run a single test by name
uv run pytest tests/test_faithfulness.py::test_name

# Run the CLI
python main.py --input data/sample_eval_set.jsonl --output results/
python main.py --input data/sample_eval_set.jsonl --output results/ --faithfulness-method nli
OPENAI_API_KEY=sk-... python main.py --input data/sample_eval_set.jsonl --output results/ --llm-judge

# Installed entry point (after pip install)
rag-eval --input data/sample_eval_set.jsonl --output results/
```

## Architecture

### Data flow

```
input JSONL  →  pipeline.py (run_evaluation)
                  ├── faithfulness.py   (score_faithfulness_detailed)
                  ├── relevance.py      (score_answer_relevance)
                  ├── precision.py      (score_context_precision)
                  ├── recall.py         (score_context_recall)
                  └── context_relevance.py (score_context_relevance)
                ↓
              clustering.py (cluster_failures)
                ↓
              results/scores.jsonl + report.md + summary.json
```

The pipeline scores records in parallel using `ThreadPoolExecutor`. `pipeline.py:run_evaluation` is the single entry point; `cli.py:main` is the CLI wrapper.

### Key types (`rag_eval_harness/types.py`)

- `EvalRecord` — input: question, answer, contexts, optional ground_truth
- `ScoredRecord` — input + scores
- `Scores` — five metric floats (recall is Optional) + faithfulness_claims
- `ClaimVerdict` — per-claim supported bool + optional confidence

### Faithfulness backends (`faithfulness.py`)

Three backends selected via `faithfulness_method` config key:

1. **cosine** (default) — max cosine similarity of each claim against context sentences; fast, offline, misses contradictions
2. **nli** — `NLIScorer` (CrossEncoder `nli-deberta-v3-small`, ~85 MB); detects negations and contradictions; no API key; misses magnitude/unit errors
3. **llm** — gpt-4o-mini via `AsyncOpenAI`; concurrent per-claim async calls; highest accuracy; falls back to cosine on API failure

Priority in `score_faithfulness_detailed`: NLI > LLM > cosine.

### Embedder (`embeddings.py`)

Lazy-loading `SentenceTransformer` wrapper with a thread-safe per-text embedding cache. Cache avoids re-encoding identical strings when multiple scorers process the same record. Double-checked locking pattern for model init.

### NLIScorer (`nli.py`)

Lazy-loading `CrossEncoder` wrapper. `entailment_idx` is resolved from the model's own `id2label` config rather than hardcoded — different NLI checkpoints use different label orderings. `entailment_matrix()` batches all pairs into a single `model.predict` call for efficiency.

### Clustering (`clustering.py`)

k-means over embedded `"question answer"` strings. Searches `min_k..max_k` and picks k with the best silhouette score. Each cluster is labelled by its lowest-scoring metric (faithfulness → "hallucination-dominant", recall → "retrieval-gap-dominant", etc.).

### Text splitting (`text_utils.py`)

`split_into_claims` splits on `.?!;` and on `, and ` only when followed by a subject pronoun. This avoids splitting Oxford-comma enumerations while correctly splitting compound independent clauses.

## Config

CLI `--faithfulness-method` and `--llm-judge` flags override `config.yaml`. Config keys: `embedder_model`, `faithfulness_method`, `nli_model`, `similarity_thresholds.faithfulness`, `similarity_thresholds.precision`, `clustering.min_k/max_k`, `llm_as_judge.enabled`, `quality_gates.*`.

## Tests

The `conftest.py` session-scoped `embedder` fixture is shared across all test modules to avoid reloading the SentenceTransformer model per file. Pass it to any test that needs embedding.
