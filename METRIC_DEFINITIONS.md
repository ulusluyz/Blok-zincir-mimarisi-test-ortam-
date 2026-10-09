# METRIC DEFINITIONS AND MATHEMATICAL FORMULAS (METRİK TANIMLARI)

**Proje:** Öğrenilmiş Blok Uygunluğu ve Hiyerarşik Yönlendirme Denetimi
**Dosya:** `METRIC_DEFINITIONS.md`

---

## 1. Metriklerin Matematiksel Tanımları ve Kod Konumları

### 1.1 Gerçek Etiket Doğruluğu (True Label Accuracy / Top-1 Accuracy)
* **Tanım:** Model tarafından birinci sırada ($Top-1$) seçilen $A$ bloğunun kategorisinin, sorgunun gerçek veri etiketine ($y_i$) eşit olma oranıdır.
* **Matematiksel Formül:**
  $$\text{Top-1 Accuracy} = \frac{\sum_{i=1}^{M} \mathbb{I}(\hat{y}_{i,1} = y_i)}{M}$$
  * Pay: Doğru kategoriye yönlendirilen sorgu sayısı.
  * Payda: Toplam test sorgusu sayısı ($M$).
* **Kod Konumu:** `src/benchmark.py` -> `calculate_metrics()` -> satır 35-43.

---

### 1.2 Top-3 Blok Doğruluğu (Top-3 Block Accuracy)
* **Tanım:** Sorgunun gerçek kategorisinin ($y_i$), model tarafından ilk 3 sırada ($Top-3$) önerilen aday kategoriler arasında bulunma oranıdır.
* **Matematiksel Formül:**
  $$\text{Top-3 Accuracy} = \frac{\sum_{i=1}^{M} \mathbb{I}(y_i \in \{\hat{y}_{i,1}, \hat{y}_{i,2}, \hat{y}_{i,3}\})}{M}$$
  * Pay: İlk 3 aday içinde doğru kategorisi bulunan sorgu sayısı.
  * Payda: Toplam test sorgusu sayısı ($M$).
* **Açıklama (Top-1 ile Top-3 Eşitliği Çözümü):**
  `process_query(..., top_k=1)` çağrıldığında sistem yalnızca $Top-1$ kazanan bloğu aktif edip döndürmektedir (`active_blocks = candidate_list[:1]`). Sonuç dizisinin uzunluğu 1 olduğu için `ranked_preds[:3]` ifadesi yalnızca ilk elemanı almakta ve dolayısıyla $Top-3$ skoru matematiksel olarak $Top-1$ skoruna eşit çıkmaktadır. `top_k=3` veya daha büyük değerler verildiğinde $Top-3$ skoru $Top-1$'den yüksek olmaktadır (%85.2 vs %74.1).
* **Kod Konumu:** `src/benchmark.py` -> `calculate_metrics()` -> satır 46-47.

---

### 1.3 Doğru Grubu Koruma Oranı (Group Retention Rate / Group Recall)
* **Tanım:** Hiyerarşik yönlendirme aşamasında $G$ grup temsilcilerinin, sorgunun ait olduğu gerçek grubu ($y_i$) aktif edilen aday gruplar kümesine dahil etme oranıdır.
* **Matematiksel Formül:**
  $$\text{Group Recall} = \frac{\sum_{i=1}^{M} \mathbb{I}(y_i \in \mathcal{G}_{\text{active}, i})}{M}$$
  * Pay: Gerçek grubu aktifleşen $G$ grupları arasında yer alan sorgu sayısı.
  * Payda: Toplam test sorgusu sayısı ($M$).
* **Kod Konumu:** `src/benchmark.py` -> `run_hierarchical_benchmark()` -> satır 216-218.

---

### 1.4 Yanlış Eleme Oranı (False Pruning Rate / False Negative Rate)
* **Tanım:** $G$ temsilcilerinin hatalı karar vererek doğru $A$ bloklarını içeren grubu süzgeç aşamasında yanlışlıkla eleme oranıdır.
* **Matematiksel Formül:**
  $$\text{False Pruning Rate} = 1 - \text{Group Recall} = \frac{\sum_{i=1}^{M} \mathbb{I}(y_i \notin \mathcal{G}_{\text{active}, i})}{M}$$
  * Pay: Doğru grubu yanlışlıkla elenen sorgu sayısı.
  * Payda: Toplam test sorgusu sayısı ($M$).
* **Kod Konumu:** `src/benchmark.py` -> satır 228.

---

### 1.5 Tam Taramayı Koruma Oranı (Full Scan Retention Rate)
* **Tanım:** Hiyerarşik $G$ yönlendirmesinin seçtiği $Top-1$ kazanan bloğun, hiçbir eleme yapılmayan Tam $A$ Taramasının (Full Scan) seçtiği $Top-1$ blok ile aynı olma oranıdır.
* **Matematiksel Formül:**
  $$\text{Full Scan Retention} = \frac{\sum_{i=1}^{M} \mathbb{I}(\hat{b}_{G, i} = \hat{b}_{\text{Full}, i})}{M}$$
  * Pay: $G$ yönlendirmesi ile Tam Taramada aynı kazanan bloğu üreten sorgu sayısı.
  * Payda: Toplam test sorgusu sayısı ($M$).
* **Kod Konumu:** `src/benchmark.py` -> satır 220-225.

---

### 1.6 Çok Konulu Grup Recall (Multi-Topic Group Recall)
* **Tanım:** Çok konulu sorgularda beklenen tüm gerekli konuların ($\mathcal{T}_{\text{expected}, i}$) aktif edilen gruplar ($\mathcal{G}_{\text{active}, i}$) tarafından kapsanma oranıdır.
* **Matematiksel Formül:**
  $$\text{Multi-Topic Recall} = \frac{\sum_{i=1}^{K} |\mathcal{T}_{\text{expected}, i} \cap \mathcal{G}_{\text{active}, i}|}{\sum_{i=1}^{K} |\mathcal{T}_{\text{expected}, i}|}$$
  * Pay: Aktifleşen gruplar tarafından yakalanan beklenen konu sayısı.
  * Payda: Tüm çok konulu sorgulardaki toplam beklenen konu sayısı.
* **Kod Konumu:** `src/benchmark.py` -> satır 122-124 & `audit_multi_topic.py` -> satır 82-84.
