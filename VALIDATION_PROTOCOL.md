# VALIDATION_PROTOCOL.md: Experimental Protocol & PASS/FAIL Criteria Specification

## 1. Experimental Methodology & Reproducibility
- **Random Seed**: Fixed `seed = 42` across all feature extractors, binary encoders, and index builders.
- **Dataset Splits**:
  - `EXPANDED_TRAIN_DATA`: 60 supervised samples across 10 categories.
  - `VAL_DATASET`: 10 validation samples for hyperparameter tuning.
  - `UNSEEN_TEST_SINGLE_DATASET`: 10 unseen single-topic test samples (zero train overlap).
  - `UNSEEN_TEST_MULTI_DATASET`: 10 rich multi-topic test samples.
  - `CHAIN_BLOCK_BENCHMARK_SUITE`: 27 end-to-end benchmark test cases.

---

## 2. Automated PASS / FAIL Status Criteria

| Metric Name | Achieved Value | Target Value | Operator | Status | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **InDomain_Top1_Accuracy** | 40.00 | 80.00 | `>=` | **FAIL** | In-domain Top-1 category accuracy >= 80% |
| **InDomain_Top3_Accuracy** | 75.00 | 90.00 | `>=` | **FAIL** | In-domain Top-3 category accuracy >= 90% |
| **Unseen_Negation_Disambiguation** | 100.00 | 85.00 | `>=` | **PASS** | Unseen command negation disambiguation >= 85% |
| **Cognitive_Decision_Accuracy** | 55.17 | 80.00 | `>=` | **FAIL** | Cognitive auditor decision accuracy >= 80% |
| **MultiHop_Transitive_Inference** | 25.00 | 90.00 | `>=` | **FAIL** | Multi-hop relational deduction accuracy >= 90% |
| **OOD_True_Reject_Rate** | 0.00 | 90.00 | `>=` | **FAIL** | OOD True Reject Rate >= 90% |
| **Unselected_APlus_Read_Ratio** | 0.00 | 0.00 | `==` | **PASS** | Unselected A+ address layer read ratio == 0% |
| **Average_Query_Latency** | 19.86 | 50.00 | `<=` | **PASS** | Average query latency <= 50 ms |

---

## 3. Evaluation Rules
- **PASS**: Achieved metric satisfies the target comparison condition.
- **FAIL**: Achieved metric fails the target comparison condition.
- **NOT_EVALUATED**: Metric could not be computed due to missing evidence or execution error.
