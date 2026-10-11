# EXPERIMENT_REPORT.md: Verified Empirical Evaluation Report

## Executive Summary
- **Overall Empirical Status**: **KISMEN BAŞARILI**
- **Architecture**: Non-Transformer Bitwise $A$ Knowledge Block / $A+$ Address Layer Paradigm
- **Evaluation Date**: March 2025

---

## Key Benchmark Metrics Summary

| Metric | Target Criteria | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **In-Domain Top-1 Accuracy** | $\ge 80.0\%$ | **45.00%** | FAIL |
| **In-Domain Top-3 Accuracy** | $\ge 90.0\%$ | **80.00%** | FAIL |
| **Unseen Negation Disambiguation** | $\ge 85.0\%$ | **100.00%** | PASS |
| **Cognitive Decision Accuracy** | $\ge 80.0\%$ | **62.07%** | FAIL |
| **Multi-Hop Deduction Accuracy** | $\ge 90.0\%$ | **50.00%** | FAIL |
| **OOD True Reject Rate (TRR)** | $\ge 90.0\%$ | **0.00%** | FAIL |
| **Unselected A+ Read Ratio** | **0.00%** | **0.00%** | PASS |
| **Average Query Latency** | $\le 50.0$ ms | **22.616 ms** | PASS |

---

## Subsystem Latency Breakdown (ms)
- **Feature Extraction & Encoding**: **14.283 ms**
- **A Mathematical Evaluation**: **0.059 ms**
- **Candidate Ranking & Selection**: **0.016 ms**
- **Actual A+ Layer Access**: **7.697 ms**
- **Relational Multi-Hop Inference**: **0.096 ms**
- **Total Query Latency**: **22.616 ms**

---

## Answers to Mandatory Research Questions

### 1. Bloklar uygunluğu gerçekten öğrenebildi mi?
**KISMEN.** Supervised Contrastive STE Kayıp Fonksiyonu ile eğitilen $A$ Bilgi Blokları, sızıntısız test setinde **%45.00** Top-1 kategori doğruluğuna ulaşmıştır.

### 2. XOR + POPCOUNT yeterli mi?
**EVET.** XOR + POPCOUNT işlemi CPU üzerinde ortalama **0.059 ms** sürede tamamlanmakta ve matris çarpımına ihtiyaç duymamaktadır.

### 3. Görülmemiş Türkçe sorguları ve olumsuz ifadeleri ayırt edebiliyor mu?
**EVET.** Eğitilmeyen yeni fiil ve cümle yapılarında olumsuzluk ayrıştırma başarısı **%100.00** olarak ölçülmüştür.

### 4. A+ katmanları gereksiz yere okunuyor mu?
**HAYIR.** Seçilmeyen A+ katmanlarının okunma oranı kesin olarak **%0.00** olarak doğrulanmıştır.

### 5. Bellek ve işlem maliyetleri ne?
Ortalama sorgu gecikmesi **22.616 ms** (P95: **24.213 ms**) ve tepe bellek kullanımı **3.00 MB** olarak ölçülmüştür.
