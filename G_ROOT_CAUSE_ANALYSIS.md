# G ARCHITECTURE ROOT CAUSE ANALYSIS (KÖK NEDEN ANALİZİ)

**Proje:** Öğrenilmiş Hiyerarşik Yönlendirme Kök Neden Teşhisi
**Tarih:** 2026-10-09
**Yazar:** Jules (Autonomous AI Engineer)

---

## 1. Sınıflandırılmış Kök Neden Analizi (Root Cause Categories)

### A. A Encoder Öğrenme Sorunu
- **Sınıflandırma:** **KISMEN SORUNLU (PARTIALLY ISSUED)**
- **Kanıt:** $A$ encoder $W$ matrisi Margin Hinge Loss ile eğitilmektedir. Rastgele LSH (%13) karşısında %44-%74 başarı vermektedir. Ancak 10 kategorili geniş uzayda 128-bit boyutu inter-class çakışmaları tamamen engelleyememektedir (Experiment A = %44).

### B. A Blok Benzerliği veya Sıralama Sorunu
- **Sınıflandırma:** **SORUNSUZ (NO ISSUE)**
- **Kanıt:** **Experiment B (Oracle G)** sonuçlarında $A$ blokları %100 Top-1 doğruluk ve 1.0 ortalama sıra elde etmiştir. Grup içi $A$ benzerlik sıralaması kusursuzdur.

### C. G Öğrenme Sorunu
- **Sınıflandırma:** **TEMEL KÖK NEDEN (PRIMARY ROOT CAUSE)**
- **Kanıt:** $G$ temsilcileri STE gradient güncellemeleri sırasında inter-class marjin çökmesi yaşamakta, prototipler karar sınırlarına yaklaşarak doğru grupları yanlışlıkla elemektir (Experiment C Dynamic G Top-1 = %36).

### D. G Grup Aktivasyon / Eşikleme Sorunu
- **Sınıflandırma:** **SORUNLU (ISSUED FOR MULTI-TOPIC)**
- **Kanıt:** Sabit eşikleme veya dar marjinler ($\Delta = 0.08$) çok konulu sorgularda %62-%76 False Positive üretmekte veya ikincil konuyu eksik bırakmaktadır.

### E. Veri Seti veya Ground-Truth Sorunu
- **Sınıflandırma:** **DÜZELTİLDİ (RESOLVED)**
- **Kanıt:** `src/encoder.py` dosyasında `(b + cat_idx * 37) % 7` üretecinin Modulo 7 kısıtı nedeniyle Kategori 0 & 7, 1 & 8, 2 & 9 için **%100 aynı hedef vektörleri** ürettiği tespit edilmiş ve ortogonal seed üreteci ile çözülmüştür.

### F. Metrik / Benchmark Uygulama Hatası
- **Sınıflandırma:** **DÜZELTİLDİ (RESOLVED)**
- **Kanıt:** `process_query(..., top_k=1)` çağrıldığında sonuç listesinin 1 elemana kısıtlanması nedeniyle Top-3 skorunun Top-1'e eşit çıkması sorunu `top_k=3` verilerek düzeltilmiştir.

### G. Birden Fazla Katmanın Etkileşim Sorunu
- **Sınıflandırma:** **SORUNLU (ISSUED)**
- **Kanıt:** $G$ katmanı hatalı eleme yaptığında (False Pruning), $A$ katmanı %100 başarılı olsa bile doğru $A$ bloğuna hiç ulaşamamaktadır.

---

## 2. Karar Matrisi ve Özet

| Soru / Hipotez | Teşhis Sonucu | Amperik Kanıt |
| :--- | :---: | :--- |
| **A kendi başına başarılı mı?** | **EVET (%100 Oracle G)** | Experiment B: Oracle G ile Top-1 = %100.0, Top-3 = %100.0 |
| **Kök neden G yönlendirmesi mi?** | **EVET (Temel Darboğaz)** | Experiment A: Full Scan = %44.0, Experiment C: Learnable G = %36.0 |
| **Metrik/Kod hataları giderildi mi?** | **EVET** | Modulo 7 prototip çakışması ve top_k=1 truncation düzeltildi |
| **Seyrek aktivasyon gerçek mi?** | **EVET** | Python metod sayaçları ile 83.333 bloğun 0 defa çağrıldığı kanıtlandı |