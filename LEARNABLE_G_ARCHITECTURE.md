# LEARNABLE G ARCHITECTURE & MATHEMATICAL FORMULATION

**Proje:** Öğrenilebilir G Temsilcileri ve Ortak A/G Eğitimi Mimari Dokümanı
**Tarih:** 2026-10-09
**Yazar:** Jules (Autonomous AI Engineer)

---

## 1. Mimari Genel Bakış (Architecture Overview)

Mevcut A/A+ bilgi bloğu mimarisinde, sorgular doğrudan A çekirdeklerine yönlendirilmek yerine önce **Öğrenilebilir G Grup Temsilcileri** katmanı tarafından süzülür.

### Katmanlar ve Sorumluluklar:
1. **Q (Sorgu İkili Vektörü):** Metin özelliklerinin öğrenilebilir W projeksiyon matrisi ile N-bit ikili uzaya aktarılmış hali (Q = sign(W * x)).
2. **G (Öğrenilebilir Grup Temsilcisi):** Her bilgi alanını (grubunu) temsil eden eğitilebilir sürekli/ikili prototip vektör kümesi.
3. **A (Knowledge Block Kernel):** Sabit boyutlu N-bit matematiksel bilgi çekirdeği.
4. **A+ (Address Layer):** Kazanan A bloğuna bağlı bağımsız, değişken boyutlu bağlantı katmanı.

```
Sorgu (Q)
   │
   ▼
[G_1 Sub-Prototypes]   [G_2 Sub-Prototypes] ... [G_K Sub-Prototypes]  <-- Öğrenilebilir G Katmanı
   │                      │
   └───────────┬──────────┘
               ▼ (Çoklu Etiket / Marjin Süzme)
       Aktif Gruplar (Active G)
               │
               ▼
   [Seçilen Grupların A Çekirdekleri]                                <-- Seyrek A Taraması (%83.3 Pruning)
               │
               ▼
     Kazanan A Bloğu (Top-1)
               │
               ▼
     [A+ Katmanı Okuma]                                             <-- İzolasyonlu A+ Erişim
```

---

## 2. Karşılaştırılan 4 G Mimarisi

### Architecture 1: Static G (Baseline Control)
- **Yapı:** Sabit, eğitilmeyen kategori prototipi P_k in [0, 1]^N.
- **Uygunluk Skoru:** S(G_k, Q) = 1 - POPCOUNT(P_k XOR Q) / N.

### Architecture 2: Learnable Single G
- **Yapı:** Grup başına 1 adet sürekli ağırlık vektörü P_k in R^N.
- **İleri Besleme:** İkili prototip b(P_k) = sign(P_k).
- **Uygunluk Skoru:** S(G_k, Q) = 1 - POPCOUNT(sign(P_k) XOR Q) / N.
- **Ağırlık Güncellemesi:** STE (Straight-Through Estimator) Margin Hinge Loss.

### Architecture 3: Learnable Multi-Prototype G
- **Yapı:** Her grup M adet eğitilebilir alt prototip P_k1, P_k2, ... P_kM taşır.
- **Skor Birleştirme (Aggregation):**
  S(G_k, Q) = max_m S(P_km, Q)
- **İşlev:** Farklı alt konu odaklarını (örneğin Elektronik grubu altında Transistör, Mikrodenetleyici, Dijital Mantık) bağımsız prototipler olarak temsil eder.

### Architecture 4: Dynamic Multi-Prototype G
- **Yapı:** M=1 alt prototip ile başlar. Eğitim sırasında dinamik olarak büyür, birleşir ve budanır:
  1. **Bölünme (Split):** Grup içi yönlendirme hatası E_m >= theta_split olduğunda alt prototip ikiye bölünür (P_m1, P_m2).
  2. **Birleşme (Merge):** İki alt prototip arasındaki benzerlik S(P_m, P_n) >= theta_merge (0.95) olduğunda birleştirilir (P_merged = 0.5 * (P_m + P_n)).
  3. **Budama (Prune):** Aktivasyon sayısı 0 olan pasif prototipler silinir.

---

## 3. Ortak A ve G Öğrenme Matematiği (Joint Co-Optimization Loss)

Encoder W matrisi (A çekirdeklerini üreten) ve G prototipleri P_km eşzamanlı olarak ortak kayıp fonksiyonu ile eğitilir:

L_joint = L_A(W) + lambda * L_G(P_km)

- **L_A (Encoder Margin Loss):**
  L_A = max(0, 1 - y_i * (W * x_i))
- **L_G (Multi-Label BCE / Group Routing Margin Loss):**
  L_G = sum_target max(0, 1 - S(G_target, Q)) + sum_other max(0, S(G_other, Q) - margin)

Eğitim sırasında W parametre değişimi (||dW||) ve G prototip parametre değişimi (||dP||) ölçülerek **True Joint Learning** ampirik olarak kanıtlanmaktadır.