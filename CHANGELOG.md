# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-24

### Added
- **Three faithfulness backends**: cosine similarity (offline, fast), CrossEncoder NLI
  (`nli-deberta-v3-small`, ~85 MB, no API key), and gpt-4o-mini LLM judge
- **Answer relevance** scoring via synthetic question generation (offline heuristic or LLM)
- **Context precision**: fraction of retrieved chunks that contribute to the answer
- **Context recall**: fraction of ground-truth claims covered by retrieved context
- **Context relevance**: semantic distance between question and retrieved chunks
- **Failure-mode clustering**: k-means with silhouette-based k selection; labels clusters
  as hallucination-dominant, retrieval-gap-dominant, noise-dominant, etc.
- **Quality gates**: configurable per-metric thresholds, `sys.exit(1)` on failure for CI
- **Baseline regression comparison**: `--baseline-scores` + `--min-delta-faithfulness`
- **CLI**: `rag-eval --input eval_set.jsonl --output results/`
- **Python API**: `from rag_eval_harness import run_evaluation, Embedder, NLIScorer`
- **Bundled data**: `from rag_eval_harness.data import sample_eval_set, adversarial_eval_set`
- 116-test suite covering all scorers, edge cases, and adversarial failure modes
- Adversarial evaluation set (17 records, 9 distinct failure modes) with empirical
  blind-spot documentation for all three faithfulness backends
