# AG LEARNING OPTIMIZATION ARCHITECTURE & STE DERIVATIONS

**Proje:** A/G Öğrenme Mimarisi ve Performans İyileştirme Dokümanı
**Tarih:** 2026-10-09
**Yazar:** Jules (Autonomous AI Engineer)

---

## 1. Mimari Katmanlar ve Sorumluluklar

1. **A Bilgi Çekirdeği (Knowledge Block Kernel):**
   - Sabit N=128 bit ikili vektör. Metin n-gram özelliklerini (D=512, Sublinear TF + L2 Normalization) ikili uzaya aktaran öğrenilebilir W projeksiyon matrisine sahiptir.
   - Sınıflar arası ayrıştırma gücü **Supervised Contrastive Loss + Hard Negative Mining** ile eğitilmektedir.

2. **G Öğrenilebilir Grup Temsilcisi (Learnable Group Representative):**
   - Her grup M adet eğitilebilir sürekli/ikili alt prototip {P_k1, P_k2, ... P_kM} barındırır.
   - Prototip çökmesini önlemek için **Inter-Class Prototype Separation Loss (L_sep)** uygulanmaktadır.

3. **A+ Adres Katmanı (Address Block Layer):**
   - Bağımsız, değişken boyutlu bağlantı katmanıdır. Yalnızca seçilen A bloğu aktifleştiğinde okunur; unselected blokların A+ katmanlarına okuma yapılmaz (Software Stack Trap doğrulaması).

```
                  Sorgu (Q)
                     │
                     ▼
  [G_1 Sub-Prototypes] ... [G_K Sub-Prototypes]    <-- Inter-Class Separation Loss (L_sep)
                     │
                     ▼
         Aktif Gruplar (Active G)                  <-- Calibrated Dynamic Multi-Label Thresholding
                     │
                     ▼
    [Seçilen Grupların A Çekirdekleri]             <-- Contrastive STE Encoder (W)
                     │
                     ▼
       Kazanan A Bloğu (Top-1)
                     │
                     ▼
       [A+ Katmanı Okunur]                          <-- Software Access Isolation Trap
```

---

## 2. Gradayan Matematiği ve STE Derivasyonu (STE Derivations)

Giriş vektörü x in R^D için sürekli projeksiyon h = W * x in R^N ve ikili vektör Q = sign(h) in {-1, +1}^N tanımlanır.

Sert adım fonksiyonu sign(h) türevlenebilir olmadığından (d sign(h)/dh = 0), Straight-Through Estimator (STE) yaklaşımı kullanılır:

dL/dh_b ≈ dL/dQ_b * 1_{|h_b| <= 1}

### A. Pozitif Prototip İtme Gradayanı (L_pos):
Pozitif hedef bit T_pos[b] in {-1, +1} için:
L_pos = max(0, 1 - T_pos[b] * h_b)
grad_pos = -T_pos[b] eğer T_pos[b] * h_b < 1.0 aksi halde 0.

### B. Zor Negatif Prototip Çekme Gradayanı (L_neg):
En yakın zor negatif hedef bit T_neg[b] in {-1, +1} için:
L_neg = max(0, 1 + T_neg[b] * h_b - mu)
grad_neg = T_neg[b] eğer T_neg[b] * h_b > -mu aksi halde 0.

### C. Ağırlık Güncelleme Kuralı:
W[b][f] = W[b][f] - eta * ( grad_pos + gamma * grad_neg ) * x_f

---

## 3. 3-Fazlı Aşamalı Eğitim Engini (3-Phase Staged Engine)

1. **Faz 1 (A Encoder Eğitimi):** W matrisi Supervised Contrastive Loss ile eğitilir (G dondurulur).
2. **Faz 2 (G Prototipleri Eğitimi):** W dondurulur, G prototipleri L_G + L_sep ile eğitilir.
3. **Faz 3 (Kontrollü İnce Ayar ve Rollback):** W ve G küçük öğrenme oranıyla (eta_fine = 0.01 * eta) ortak eğitilir. Faz 3 sonunda validation doğruluğu düşerse otomatik olarak Faz 2 checkpoint'ine geri dönülür (Rollback).