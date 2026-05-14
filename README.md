# rag-eval-harness

A standalone evaluation framework that scores RAG pipeline outputs across four quality dimensions, clusters failure modes, and produces an aggregate report — implemented from scratch using sentence-transformer embeddings with an optional OpenAI judge path.

---

## Why RAG evaluation matters

A RAG pipeline retrieves context chunks from a document store and passes them to an LLM to generate an answer. Two things can go wrong independently:

- **The retriever** pulls the wrong chunks (or not enough of them)
- **The LLM** ignores the chunks and generates from its parametric memory (hallucination)

Both failures look identical to the user: a confident, fluent, wrong answer. Standard LLM benchmarks don't catch this because they don't separate retrieval quality from generation quality.

Without a dedicated eval harness you have no signal on:

- whether a new embedding model improves retrieval without hurting answer quality
- whether a prompt change causes the LLM to start ignoring retrieved context
- which query types systematically fail and why

This harness gives you per-record and aggregate scores for each failure mode so you can act on them.

---

## How it fits into a production workflow

```
                        ┌─────────────────────────────┐
  User questions ──────►│   Your RAG pipeline         │
  (from logs/tickets)   │   retriever + LLM           │
                        └────────────┬────────────────┘
                                     │ (question, answer, contexts)
                                     ▼
                         rag-eval-harness
                        ┌─────────────────────────────┐
                        │  faithfulness   (no GT)     │
                        │  answer relevance (no GT)   │
                        │  context precision (no GT)  │
                        │  context recall  (GT req.)  │
                        │  failure-mode clustering    │
                        └────────────┬────────────────┘
                                     │
                       ┌─────────────┴──────────────┐
                       ▼                            ▼
               scores.jsonl                    report.md
               (per-record)                  (aggregate)
```

**Pre-launch gate** — before shipping a new model version or retriever config, run the harness and require, e.g., faithfulness ≥ 0.85. Block the deploy if scores drop.

**Regression testing** — wire into CI. A faithfulness dip after a vector DB schema change or prompt edit surfaces before users see it.

**Continuous monitoring** — log a sample of live (question, answer, contexts) tuples nightly and track score trends over time.

**Retriever A/B testing** — compare context precision and recall across chunk sizes (512 vs 1024 tokens), embedding models, or top-k values.

---

## What each score measures (and the business cost of failing it)

| Metric | Failure description | Business impact |
| --- | --- | --- |
| **Faithfulness** | Answer makes claims not supported by retrieved chunks | Legal liability, customer trust erosion |
| **Answer Relevance** | Answer is coherent but doesn't address the question asked | User drops off, support ticket volume rises |
| **Context Precision** | Retrieved chunks are mostly noise — low signal-to-token ratio | LLM answer quality degrades; token cost inflates with k |
| **Context Recall** | Retrieved chunks are missing key information | Answer is incomplete; user calls support anyway |
| **Context Relevance** | Retriever pulls semantically distant chunks regardless of answer utility | Silent topic drift; hard to diagnose without this metric |

---

## Architecture

```
input JSONL
     │
     ▼
evaluator/pipeline.py
┌──────────────────────────────────────────────┐
│  for each record:                            │
│    faithfulness.py  ── NLI-style claim check │
│    relevance.py     ── synthetic-Q cosine    │
│    precision.py     ── chunk contribution    │
│    recall.py        ── GT claim coverage     │
└──────────────┬───────────────────────────────┘
               │ scored records
               ▼
        clustering.py
        k-means + silhouette score selection
               │
       ┌───────┴────────┐
       ▼                ▼
results/scores.jsonl  results/report.md
```

---

## Scoring method

**Faithfulness** — the answer is decomposed into atomic claims (sentence-split). `score = supported_claims / total_claims`. Three backends available; choose based on your accuracy/speed trade-off:

| Backend | How it works | When to use |
| --- | --- | --- |
| `cosine` (default) | Max cosine similarity of each claim against context sentences | Fast prototyping, offline CI without a GPU |
| `nli` (**recommended offline**) | CrossEncoder NLI model (`nli-deberta-v3-small`, ~85 MB) classifies each (context, claim) pair as entailment/neutral/contradiction | Catch contradictions cosine misses — "Jupiter is the largest planet" and "Jupiter is the smallest planet" have near-identical embeddings but opposite entailment |
| `llm` | gpt-4o-mini judges each claim; structured JSON response | Highest accuracy; requires `OPENAI_API_KEY` and network access |

**Limitation of cosine-only faithfulness:** embedding similarity measures topic proximity, not logical support. A claim that contradicts the context can still score highly if it shares the same vocabulary. Use NLI or LLM judge for contradiction-sensitive evaluation.

### Faithfulness method — known blind spots

Verified against a 17-question adversarial set (`data/adversarial_eval_set.jsonl`) covering negation, unit errors, magnitude errors, wrong attribution, date contradictions, and paraphrases:

| Failure type | Cosine | NLI | LLM |
| --- | --- | --- | --- |
| Negation ("NOT visible" vs "clearly visible") | ❌ | ✅ | ✅ |
| Date contradiction (1989 vs 1991) | ❌ | ✅ | ✅ |
| Percentage contradiction (71% vs 50%) | ❌ | ✅ | ✅ |
| Wrong attribution (Newton vs Einstein) | ✅ | ✅ | ✅ |
| Explicit correction in context (Sydney → Canberra) | ❌ | ✅ | ✅ |
| Magnitude error (3,844 km vs 384,400 km, 100×) | ❌ | ❌ | ✅ |
| Unit confusion (37°C vs 37°F, same number) | ❌ | ❌ | ✅ |
| Vague paraphrase (concise subset of context) | ❌ | ✅ | ✅ |
| Synonym paraphrase ("biggest" = "largest") | ✅ | ✅ | ✅ |
| Speculative addition beyond context | ✅ | ✅ | ✅ |
| Multi-chunk synthesis (claims span two contexts) | ✅ | ✅ | ✅ |
| Completely irrelevant context | ✅ | ✅ | ✅ |

**Why cosine misses negation and numbers:** embeddings encode topic proximity, not logical polarity. "The wall is visible" and "the wall is NOT visible" are a single token apart — their vectors are nearly identical.

**Why NLI misses magnitude and unit errors:** NLI models are trained on textual entailment patterns, not arithmetic. "3,844 km" and "384,400 km" have the same sentence structure and unit word; the model has no mechanism to compare numeric magnitudes. Similarly, "37°C" and "37°F" look like the same number with a different letter.

**Practical guidance:** cosine is sufficient if your RAG answers are unlikely to contradict context using the same vocabulary. Switch to NLI when contradiction detection matters and you need an offline solution. Use LLM judge when numeric facts, units, or measurements are critical — it is the only method that reliably catches magnitude and unit errors.

### Threshold tuning

The default thresholds are conservative starting points, not universal truths. Each threshold is a tradeoff between false positives (flagging correct answers) and false negatives (missing real hallucinations):

| Threshold | Default | Lower if… | Raise if… |
| --- | --- | --- | --- |
| `faithfulness` (cosine) | 0.7 | Short paraphrases are being missed (false negatives) | Too many correct answers flagged as hallucinations |
| `precision` (cosine) | 0.5 | Good context chunks are being excluded | Noisy chunks are being counted as useful |
| NLI entailment (internal, 0.5) | 0.5 | Valid paraphrases with different phrasing are failing | Weak entailments are accepted too liberally |

Run `data/adversarial_eval_set.jsonl` after any threshold change to verify you haven't introduced new regressions.

### Diagnosing failures from metric combinations

A single low score points to the retriever or the LLM; the combination tells you which:

| Pattern | Probable cause | Where to look |
| --- | --- | --- |
| `faithfulness` ↓, others normal | LLM generating beyond its context | Tighten prompt; switch to NLI or LLM judge |
| `ctx_rel` ↓ + `precision` ↓ | Retriever pulling wrong documents | Review embedding model, re-chunk, check top-k |
| `recall` ↓, others normal | Retriever missing key chunks | Increase top-k; check document coverage |
| `faithfulness` ↓ + `ctx_rel` ↓ | Both retriever and generator failing | Fix retriever first, then re-evaluate faithfulness |
| `relevance` ↓, faithfulness OK | Answer is on-topic but doesn't address the question | Review answer generation prompt |
| All metrics normal except `ctx_rel` near 0 | Specific query type returns junk (topic drift) | Add adversarial examples for that query type |

`ctx_rel` is the most direct retrieval health signal — it measures question-to-context semantic distance before any generation happens. A `ctx_rel` below 0.3 almost always means the retriever fetched the wrong document, regardless of how the LLM handled it.

**Answer Relevance** — three synthetic questions are generated from the answer text (heuristic offline, or via gpt-4o-mini when enabled). All four questions (original + synthetic) are embedded and the score is the mean cosine similarity to the original question. A good answer generates questions that closely resemble the real question.

**Context Precision** — for each retrieved chunk, compute max cosine similarity to any sentence in the answer. `precision = chunks_above_threshold / total_chunks`. A low score means your retriever is pulling irrelevant documents.

> **Note on definition:** RAGAS context precision is a rank-weighted precision@k that asks "were the relevant chunks ranked first?" This harness measures chunk-to-answer cosine contribution instead — a chunk counts as useful if any sentence in the final answer is semantically close to it. The RAGAS definition requires a ground truth relevance label per chunk; this definition is reference-free and computable offline.

**Context Recall** — the ground truth is decomposed into atomic claims. Each claim is checked against the union of retrieved context sentences. `recall = covered_claims / total_claims`. Requires `ground_truth`; skipped when absent.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

**Offline mode (no API key):**

```bash
git clone https://github.com/paulyonghaoli/rag-eval-harness
cd rag-eval-harness
uv pip install -e .
```

**LLM judge mode (optional):**

```bash
uv pip install -e ".[llm]"
```

**Development (includes pytest):**

```bash
uv pip install -e ".[dev]"
```

---

## Run

**Offline (cosine, no API key needed):**

```bash
python main.py --input data/sample_eval_set.jsonl --output results/
```

**Offline NLI (recommended for contradiction-sensitive evaluation):**

```bash
python main.py --input data/sample_eval_set.jsonl --output results/ --faithfulness-method nli
```

Downloads `cross-encoder/nli-deberta-v3-small` (~85 MB) on first run. No API key required.

**LLM judge (highest accuracy, requires OpenAI key):**

```bash
OPENAI_API_KEY=sk-... python main.py --input data/sample_eval_set.jsonl --output results/ --llm-judge
```

CLI flags override `config.yaml` when provided.

---

## Outputs

Every run writes three files to `--output`:

| File | Contents |
| --- | --- |
| `scores.jsonl` | Per-record scores with per-claim faithfulness verdicts |
| `report.md` | Aggregate mean ± std per metric + failure-mode cluster summaries |
| `summary.json` | Machine-readable metrics + quality gate results (for CI) |

`summary.json` format:

```json
{
  "records_evaluated": 25,
  "metrics": {
    "faithfulness": {"mean": 0.62, "std": 0.47, "n": 25}
  },
  "quality_gates": null,
  "passed": null
}
```

---

## CI quality gates

Add to `config.yaml` to fail the run when any metric mean falls below a threshold:

```yaml
quality_gates:
  faithfulness: 0.85
  relevance: 0.70
  precision: 0.60
  context_relevance: 0.60
  recall: 0.70
```

Exit code is 1 if any gate fails — wire directly into GitHub Actions:

```yaml
- name: RAG regression check
  run: python main.py --input data/eval_set.jsonl --output results/
```

---

## Baseline regression comparison

Save a known-good `scores.jsonl` and fail future runs if faithfulness drops:

```bash
# Save baseline
python main.py --input data/eval_set.jsonl --output baseline/

# Later — fail if faithfulness drops more than 2 percentage points
python main.py --input data/eval_set.jsonl --output results/ \
  --baseline-scores baseline/scores.jsonl \
  --min-delta-faithfulness -0.02
```

---

## Config

```yaml
llm_as_judge:
  enabled: false            # set true to use gpt-4o-mini for faithfulness + relevance
  openai_api_key_env: OPENAI_API_KEY   # name of the env var holding your key
faithfulness_method: cosine # cosine | nli  — override per-run with --faithfulness-method
nli_model: cross-encoder/nli-deberta-v3-small
similarity_thresholds:
  faithfulness: 0.7         # cosine threshold for claim support
  precision: 0.5            # cosine threshold for chunk contribution
clustering:
  min_k: 3                  # minimum k for k-means search
  max_k: 5                  # maximum k for k-means search
embedder_model: all-MiniLM-L6-v2
# quality_gates:            # uncomment to enable CI gate
#   faithfulness: 0.85
#   relevance: 0.70
```

---

## Tests

```bash
uv run pytest
```

---

## Building a real eval set

The sample data in `data/sample_eval_set.jsonl` uses synthetic trivia to exercise the scoring machinery. For a real deployment, collect data from three sources:

### 1. Mine your existing logs

Pull real user questions from your support tickets, chat logs, or analytics. For each question, re-run your retriever to capture the contexts it actually returned, then run your LLM to get the current answer. A human (or GPT-4o) writes the `ground_truth`.

```text
real_queries  ──► retriever ──► contexts
                                    │
real_queries  ──► LLM  ────────► answer
                                    │
human / GPT-4o ─────────────► ground_truth
```

100 real queries is a practical starting point — enough to detect regressions without being expensive to label.

### 2. Construct targeted failure examples

For each failure mode you care about, write examples that should trigger it. This gives you guaranteed coverage of edge cases your log sample may not contain.

```jsonl
// Hallucination target — LLM contradicts the retrieved context
{"question": "What is the cancellation fee?",
 "answer": "There is no cancellation fee.",
 "contexts": ["A $25 fee applies to orders cancelled after 24 hours."],
 "ground_truth": "The cancellation fee is $25 for orders cancelled after 24 hours."}

// Low context precision — retriever pulled an irrelevant chunk
{"question": "How do I reset my password?",
 "answer": "Click Forgot Password on the login page.",
 "contexts": ["Click Forgot Password on the login page.",
              "Our servers use AES-256 encryption.",
              "The French office is open 9am–6pm CET."],
 "ground_truth": "Click Forgot Password on the login page and follow the email link."}

// Low context recall — retriever missed a key chunk
{"question": "What payment methods do you accept?",
 "answer": "We accept Visa and Mastercard.",
 "contexts": ["We accept Visa and Mastercard."],
 "ground_truth": "We accept Visa, Mastercard, PayPal, and cryptocurrency."}

// Off-topic answer — low relevance
{"question": "How do I cancel my subscription?",
 "answer": "Our platform uses bank-grade encryption to protect your data.",
 "contexts": ["To cancel, go to Account > Subscription > Cancel Plan."],
 "ground_truth": "Go to Account > Subscription > Cancel Plan."}
```

### 3. Domain-specific adversarial cases

- **Multi-hop questions** — the correct answer requires combining information from two separate chunks; tests whether low precision (too few chunks) causes failures
- **Temporally sensitive questions** — the answer changed between document versions; tests whether stale chunks pollute recall
- **Ambiguous questions** — multiple valid interpretations; tests whether relevance scores are robust

### Recommended eval set composition

| Type | Count | Purpose |
| --- | --- | --- |
| Real user queries (correctly answered) | 50 | Baseline / regression guard |
| Real user queries (known failures) | 20 | Catch regressions on hard cases |
| Constructed hallucination examples | 10 | Faithfulness sensitivity check |
| Constructed precision failures | 10 | Retriever noise detection |
| Constructed recall failures | 10 | Retriever coverage detection |
| Off-topic / low relevance | 5 | Prompt drift detection |
| Empty context | 5 | Graceful degradation check |

---

## Example output

### Console summary

```text
==================================================
RAG Evaluation Summary
==================================================
Records evaluated : 25

  faithfulness       0.620 ± 0.475
  relevance          0.736 ± 0.127
  precision          0.760 ± 0.427
  recall             0.913 ± 0.282

Scores  -> results/scores.jsonl
Report  -> results/report.md
==================================================
```

### Per-record scores (`results/scores.jsonl`)

```jsonl
{"question": "Where is the Eiffel Tower located?", "answer": "The Eiffel Tower is in Paris, France.", "scores": {"faithfulness": 1.0, "relevance": 0.87, "precision": 1.0, "recall": 1.0}}
{"question": "Who invented the telephone?", "answer": "The telephone was invented by Thomas Edison in 1867...", "scores": {"faithfulness": 0.0, "relevance": 0.54, "precision": 0.0, "recall": 0.0}}
```

### Aggregate report (`results/report.md`)

Includes mean ± std per dimension, failure-mode cluster summaries (size, mean scores, centroid example), and the top-5 worst examples per cluster.

---

## Input format

```json
{
  "question": "What causes tides?",
  "answer": "Tides are caused by the Moon and Sun's gravity.",
  "contexts": ["The Moon's gravity creates ocean tides.", "..."],
  "ground_truth": "Tides are caused by the gravitational pull of the Moon and Sun."
}
```

`ground_truth` is optional. Context recall is skipped for any record that omits it.
