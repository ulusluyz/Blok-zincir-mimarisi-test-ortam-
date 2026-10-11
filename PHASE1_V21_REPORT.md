# Aşama 1 V21 — süreçler arası deterministiklik doğrulaması

- Yeni test: `tests/test_encoder_process_determinism_v21.py`.
- `PYTHONHASHSEED=1`, `42` ve `random` altında ayrı Python süreçlerinde aynı sabit verilerle TF-IDF, `TrainableBinaryEncoder.fit`, iki `LearnableMultiPrototypeGRepresentative` eğitimi ve yönlendirme sonuçları karşılaştırıldı.
- Karşılaştırılanlar: sözlük sırası, encoder ağırlıklarının SHA-256 özeti, ikili kodlar, G alt prototipleri ve yönlendirme kararları.
- Yeni testler: **2/2 başarılı**.
- Hedefli paket (`test_learnable_g.py`, `test_feature_determinism_v20.py`, yeni testler): **12/12 başarılı, 8.37 saniye**.
- Tam paket: çalışma süresi sınırına ulaştı; geçiş onayı yok. Tam paketin neden yavaşladığı bu turda kesinleştirilmedi.
- Sınır: Bu test, küçük sabit bir örneklemde ve mevcut Python/NumPy ortamında deterministiklik gösterir; tüm donanımlar veya tüm eğitim senaryoları için garanti değildir.
- GitHub değiştirilmedi.
