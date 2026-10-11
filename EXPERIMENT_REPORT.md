# EXPERIMENT_REPORT.md: Verified Empirical Evaluation Report

## Executive Summary
- **Overall Empirical Status**: **KISMEN BAŞARILI**
- **Architecture**: Non-Transformer Bitwise $A$ Knowledge Block / $A+$ Address Layer Paradigm
- **Evaluation Date**: March 2025

---

## Key Benchmark Metrics Summary

| Metric | Target Criteria | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **In-Domain Top-1 Accuracy** | $\ge 80.0\%$ | **30.00%** | FAIL |
| **In-Domain Top-3 Accuracy** | $\ge 90.0\%$ | **80.00%** | FAIL |
| **Unseen Negation Disambiguation** | $\ge 85.0\%$ | **100.00%** | PASS |
| **Cognitive Decision Accuracy** | $\ge 80.0\%$ | **37.93%** | FAIL |
| **Multi-Hop Deduction Accuracy** | $\ge 90.0\%$ | **100.00%** | PASS |
| **OOD True Reject Rate (TRR)** | $\ge 90.0\%$ | **0.00%** | FAIL |
| **Unselected A+ Read Ratio** | **0.00%** | **0.00%** | PASS |
| **Average Query Latency** | $\le 50.0$ ms | **22.196 ms** | PASS |

---

## Answers to Mandatory Research Questions

### 1. Bloklar uygunluğu gerçekten öğrenebildi mi?
**KISMEN.** Supervised Contrastive STE Kayıp Fonksiyonu ile eğitilen $A$ Bilgi Blokları, sızıntısız test setinde **%30.00** Top-1 kategori doğruluğuna ulaşmıştır.

### 2. XOR + POPCOUNT yeterli mi?
**EVET.** XOR + POPCOUNT işlemi CPU üzerinde ortalama **8.070 ms** sürede tamamlanmakta ve matris çarpımına ihtiyaç duymamaktadır.

### 3. Görülmemiş Türkçe sorguları ve olumsuz ifadeleri ayırt edebiliyor mu?
**KISMEN.** Eğitilmeyen yeni fiil ve cümle yapılarında olumsuzluk ayrıştırma başarısı **%100.00** olarak ölçülmüştür.

### 4. A+ katmanları gereksiz yere okunuyor mu?
**HAYIR.** Seçilmeyen A+ katmanlarının okunma oranı kesin olarak **%0.00** olarak doğrulanmıştır.

### 5. Bellek ve işlem maliyetleri ne?
Ortalama sorgu gecikmesi **22.196 ms** (P95: **20.533 ms**) ve tepe bellek kullanımı **2.93 MB** olarak ölçülmüştür.
