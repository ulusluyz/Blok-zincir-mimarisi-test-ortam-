# Aşama 1 V17 — G kanıt ön inceleme deneyi

- Yeni opsiyonel yönlendirme stratejisi: `evidence_probe`.
- En fazla `probe_group_count` G grubunun A blokları puanlanır; A+ verileri bu aşamada okunmaz.
- `probe_weight` ile G ve en iyi A benzerlik puanı harmanlanır; `max_active_groups` kadar grup seçilir.
- Mevcut varsayılan `dynamic_multi_label` değiştirilmedi.
- Yeni 5 test: kontrollü yanıltıcı G puanında doğru grup seçimi, A+ izolasyonu, parametre doğrulama.
- Tam regresyon: 121/121 başarılı (28,66 saniye).
- Sınırlama: Kanıt ön incelemesi ek A karşılaştırmaları gerektirir; düşük gecikme veya gerçek çok konulu geri çağırma iyileşmesi henüz kanıtlanmadı. Gerçek veri üzerinde bağımsız karşılaştırma zorunludur.
- Gerçek güç kesintisi/disk arızası testleri henüz gerçekleştirilmedi. Aşama 1 nihai onay almadı.
