# AG CO-OPTIMIZATION & LEARNING ARCHITECTURE

**Proje:** A/G Ortak Öğrenme ve Yönlendirme İyileştirme Mimari Dokümanı
**Tarih:** 2026-10-09
**Yazar:** Jules (Autonomous AI Engineer)

---

## 1. Mimari Katmanlar ve Sorumluluklar

1. **A Bilgi Çekirdeği (Knowledge Block Kernel):**
   - Sabit boyutlu N-bit ikili matematiksel vektör (N = 128).
   - Metin özelliklerini N-bit ikili uzaya aktaran öğrenilebilir W projeksiyon matrisine sahiptir.
   - Sınıflar arası ayrıştırma gücünü artırmak için **Supervised Contrastive STE Loss + Hard Negative Mining** ile eğitilir.

2. **G Öğrenilebilir Grup Temsilcisi (Learnable Group Representative):**
   - Her bilgi kategorisi için M adet eğitilebilir sürekli/ikili alt prototip {P_k1, P_k2, ... P_kM} barındırır.
   - Sorgu Q ile benzerliği S(G_k, Q) = max_m S(P_km, Q) formülüyle hesaplanır.
   - Çoklu prototipler, grubun alt konu çeşitliliğini (örneğin Elektronik altında Transistör vs Entegre Devre) bağımsız temsil eder.

3. **A+ Adres Katmanı (Address Block Layer):**
   - Bağımsız, değişken boyutlu bağlantı katmanıdır. Yalnızca seçilen A bloğu aktifleştiğinde okunur; unselected blokların A+ katmanlarına okuma yapılmaz.

```
                  Sorgu (Q)
                     │
                     ▼
  [G_1 Sub-Prototypes] ... [G_K Sub-Prototypes]    <-- Inter-Class Separation Loss (L_sep)
                     │
                     ▼
         Aktif Gruplar (Active G)                  <-- Dynamic Multi-Label Thresholding
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

## 2. Birleşik Kayıp Fonksiyonu ve Gradayan Matematiği (Unified Loss Formulation)

Encoder W matrisi (A çekirdekleri) ve G prototipleri {P_km} eşzamanlı olarak eğitilir:

L_total = L_A(W) + lambda_G * L_G({P_km}) + lambda_sep * L_sep({P_km})

### A. L_A (Contrastive STE Loss with Hard Negative Mining):
For query x_i with target prototype T_pos and hardest negative prototype T_neg:
L_A = sum_{b=1}^N [ max(0, 1 - T_pos[b] * h_i[b]) + gamma * max(0, 1 + T_neg[b] * h_i[b] - margin) ]
where h_i = W * x_i. Gradient update via STE:
dW[b][f] = -eta * ( grad_pos + gamma * grad_neg ) * x_i[f]

### B. L_sep (Inter-Class Prototype Separation Loss):
Enforces minimum Hamming separation d_min between distinct group prototypes:
L_sep = sum_{k != j} max(0, S(P_k, P_j) - margin_sep)
If similarity S(P_k, P_j) > 0.60, repulsion gradients force prototypes P_k and P_j apart.