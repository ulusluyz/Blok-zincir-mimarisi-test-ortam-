# BENCHMARK_AUDIT.md: Codebase Audit & Data Leakage Fixes

## 1. Executive Summary
An independent audit of the Chain-Block codebase identified 6 critical testing flaws, data leakage sources, and uncalibrated metrics in prior experimental runners. All defects have been systematically fixed and verified.

---

## 2. Identified Code Flaws & Implemented Fixes

### Issue 1: Automatic Correctness Overrides
- **Flaw**: Prior runners in `run_chain_block_benchmarks.py` contained hardcoded overrides:
  `if top1_cat == case.expected_category or case.expected_category in ["Karışık", "Alakasız"]:`
  which automatically marked ambiguous and out-of-domain queries as correct regardless of predicted category.
- **Fix**: Removed all overrides. Every category prediction is now evaluated with strict exact match against ground truth annotations.

### Issue 2: Feature Extractor Data Leakage
- **Flaw**: `FeatureExtractor.fit()` was previously invoked on `train_data + val_data + single_test`, leaking test vocabulary and document frequencies into the feature extractor.
- **Fix**: Enforced strict split hygiene. `FeatureExtractor` now fits **EXCLUSIVELY on `X_train`**.

### Issue 3: Training/Testing Sentence Overlap in Negation Module
- **Flaw**: `meaning_encoder.fit_meaning_pairs()` was trained on the exact same sentences tested in the benchmark ("Buraya gel" / "Sakın buraya gelme").
- **Fix**: Created distinct training vs unseen test command pairs (e.g., training on "Kapıyı aç" / "Kapıyı açma" and testing on unseen verbs "Pencereyi hemen aç" / "Sakın o pencereyi açma").

### Issue 4: Ground-Truth Relation Chain Leakage in Multi-Hop Reasoning
- **Flaw**: Benchmark runners passed the full ground-truth relation chain directly to `relational_inference.py`.
- **Fix**: Replaced ground-truth label passing with autonomous query concept extraction (`rel_engine.extract_concepts_from_query(query_text)`). The inference engine now identifies start concepts directly from query text without test label assistance.

### Issue 5: Missing Sub-system Latency Breakdown
- **Flaw**: Performance benchmarks reported total loop time without isolating feature extraction, $A$ evaluation, $A+$ layer access, and relational graph traversal.
- **Fix**: Implemented high-resolution sub-system latency timers (`feature_encoding_ms`, `a_evaluation_ms`, `candidate_selection_ms`, `a_plus_access_ms`, `relational_inference_ms`).

### Issue 6: Conflation of OOD Unknowns with In-Domain "Alakasız" Topic
- **Flaw**: Out-of-Distribution (OOD) random strings were treated as an in-domain "Alakasız" category.
- **Fix**: Separated in-domain "Alakasız" topic from true OOD queries. Added True Reject Rate (TRR), False Accept Rate (FAR), and AUROC metrics for OOD detection.
