# FAILURE_ANALYSIS.md: Failure Case Breakdown & Limitations Analysis

## 1. Detailed Failure Case Log
- **[CB_003]** Query: 'Afrika savanlarında yaşayan vahşi kedigiller' | Expected Cat: `Hayvanlar` | Predicted: `Alakasız` (Score: 0.7188) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_004]** Query: 'Lehim havyası ve pasta ile devre kartı tamiri' | Expected Cat: `Elektronik` | Predicted: `Alakasız` (Score: 0.7344) | Decision: `ÇELİŞKİLİ_KANIT` (Expected: `ACCEPT`)
- **[CB_005]** Query: 'İkinci dereceden denklem köklerini bulma' | Expected Cat: `Matematik` | Predicted: `Hayvanlar` (Score: 0.8359) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_007]** Query: 'Ekosistemdeki besin zinciri ve klorofil' | Expected Cat: `Biyoloji` | Predicted: `Elektronik` (Score: 0.7500) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_008]** Query: 'Kadeş antlaşması tarihteki ilk yazılı barış metni' | Expected Cat: `Tarih` | Predicted: `Fizik` (Score: 0.7578) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_009]** Query: 'Tasavvuf edebiyatında Yunus Emre ilahileri' | Expected Cat: `Edebiyat` | Predicted: `Hayvanlar` (Score: 0.7266) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_010]** Query: 'Pazar günleri sahil kenarında yürüyüş yapmak' | Expected Cat: `Alakasız` | Predicted: `Biyoloji` (Score: 0.7109) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_NEG_006]** Query: 'Radyonun sesini yükseltme' | Expected Cat: `Elektronik` | Predicted: `Alakasız` (Score: 1.0000) | Decision: `ÇELİŞKİLİ_KANIT` (Expected: `ACCEPT`)
- **[CB_HOP_3]** Query: 'Aslan hücresel organizma mıdır?' | Expected Cat: `Hayvanlar` | Predicted: `Matematik` (Score: 0.8359) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_HOP_4]** Query: 'Güneş enerjisi kütleye dönüşür mü?' | Expected Cat: `Biyoloji` | Predicted: `Elektronik` (Score: 0.8359) | Decision: `ACCEPT` (Expected: `ACCEPT`)
- **[CB_HOP_5]** Query: 'Quark galaksi kümesi oluşturur mu?' | Expected Cat: `Fizik` | Predicted: `Elektronik` (Score: 0.8203) | Decision: `ACCEPT` (Expected: `ACCEPT`)

---

## 2. Root Cause Analysis of Identified Failure Modes

### A. Feature Sparsity on Short Unseen Inputs
Short queries with out-of-vocabulary words produce sparse TF-IDF vectors that overlap weakly with Turkish training prototypes.

### B. High Semantic Overlap Between Physics and Electronics
Queries discussing scientific terms share underlying n-grams, leading the Cognitive Auditor to trigger `ÇELİŞKİLİ_KANIT` due to narrow margins.

---

## 3. Recommended Architectural Enhancements for Phase 2
1. **Hierarchical Sub-word Byte-Pair Encoding (BPE)**: Replace character n-grams with learned BPE tokens to improve cross-domain generalization.
2. **Contextual Margin Calibration**: Dynamically adjust margin thresholds in CognitiveUncertaintyAuditor based on vocabulary density.
