# Chain-Block Architecture Specification & Technical Design

## 1. Overview
The **Chain-Block Architecture** is a non-Transformer, ultra-low latency, CPU-native cognitive information routing and reasoning paradigm. It completely eliminates matrix multiplications ($Q \times K^T$), positional encodings, self-attention mechanisms, and heavy float32/bfloat16 neural parameter trees.

Instead, knowledge is structured into modular, fixed-size **$A$ Knowledge Blocks** coupled with decoupled, dynamically expandable **$A+$ Address Layers**.

---

## 2. Core Architectural Components

### A. $A$ Knowledge Block (Fixed-Size Bitwise Representation)
- **Size**: Fixed bit length ($N \in \{64, 128, 256\}$ bits). Default is **128 bits**.
- **Role**: Learns a compact binary hash/prototype representation of an entire knowledge domain or command primitive.
- **Compute Overhead**: Evaluates queries using **bitwise XOR and POPCOUNT**:
  $$S(A, Q) = 1 - \frac{\text{POPCOUNT}(A \oplus Q)}{N}$$
- **Isolation Principle**: An $A$ Knowledge Block never executes downstream logic or reads $A+$ addresses unless $S(A, Q) \ge \tau_{\text{accept}}$.

### B. $A+$ Address Layer (Decoupled Relational Pointer Layer)
- **Structure**: Variable-length relational adjacency table containing structured links $(T_{\text{target}}, R_{\text{relation}}, w_{\text{weight}})$.
- **Relation Types**:
  - `IS_A`: Subclass / Superclass inheritance (e.g., Aslan $\rightarrow$ Memeli)
  - `PART_OF`: Meronymy (e.g., Motor $\rightarrow$ Araba)
  - `CAUSE_OF`: Causality (e.g., Yağmur $\rightarrow$ Islaklık)
  - `PREREQUISITE_OF`: Dependency (e.g., Un $\rightarrow$ Ekmek)
  - `TRANSITIVE_GENERIC`: Multi-hop deduction chains ($A > B \land B > C \implies A > C$)
- **Memory Decoupling**: Adding new connections to $A+$ does NOT alter the fixed 128-bit vector size of $A$.
- **Software Traps**: Enforced by properties and stack frame inspection to guarantee **0 unselected reads**.

---

## 3. Structural Routing & Search Index Integration

For large-scale block registries ($N_B \ge 100,000$), full linear scans of $A$ blocks are replaced by:
1. **$G$ Representative Routing Trees**: Hierarchical sub-linear cluster pruning ($O(\log K)$).
2. **CPU-Native LSH / IVF Bitwise Indexes**: Multi-table locality-sensitive hashing providing $>83.33\%$ search space pruning at zero accuracy loss.

---

## 4. Intent, Polarity & Negation Learning Module (`src/meaning_encoder.py`)

Standard TF-IDF feature extractors collapse negated command pairs ("Gel" vs "Gelme", "5 çarp" vs "5 çarpma") to $>86\%$ bitwise similarity.

The **Meaning Encoder Module** solves this by:
1. Extracting explicit Turkish morphological polarity markers (`-me`, `-ma`, `-mıyor`, `-sız`).
2. Training a Straight-Through Estimator (STE) binary encoder with Supervised Contrastive Loss ($\mathcal{L}_{\text{contrastive}}$), pushing positive and negated command vectors to low Hamming similarity ($S < 0.50$).

---

## 5. Cognitive Uncertainty Auditor & Calibrated Decision Control (`src/cognitive_control.py`)

Rather than forcing every input into a category, the Cognitive Auditor enforces 3 distinct outcomes:
1. `ACCEPT`: Single top candidate dominates with sufficient confidence and margin ($S_1 \ge 0.55, S_1 - S_2 \ge 0.05$).
2. `BELİRSİZ_VEYA_BİLİNMİYOR`: Out-of-domain or unlearned query ($S_1 < 0.55$).
3. `ÇELİŞKİLİ_KANIT`: High overlapping similarity between multiple conflicting categories ($S_1 \ge 0.55, S_1 - S_2 < 0.05$).

---

## 6. Multi-Hop Relational Deduction Engine (`src/relational_inference.py`)

Performs multi-hop transitive reasoning (2 to 5 hops) by traversing $A+$ address layers with confidence decay:
$$w_{\text{accumulated}} = \prod_{i=1}^H w_i \cdot \gamma^H, \quad \gamma = 0.90$$
This enables explicit symbolic deduction across blocks without Transformer generation.
