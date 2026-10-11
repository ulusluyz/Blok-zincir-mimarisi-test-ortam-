# Aşama 1 V25 — veri ayrımı denetimini giriş noktasına bağlama

## Gerçekleştirilen
- `src/dataset.py` içindeki `get_full_split_datasets()` artık yalnızca train–single-test küçük harf karşılaştırması yapmıyor; `assert_no_split_leakage` üzerinden dört bölümün tamamını normalleştirerek kontrol ediyor.
- Denetim hata bulursa veri kümesi döndürülmeden `ValueError` fırlatılıyor (fail-closed).
- `tests/test_split_integration_v25.py` eklendi: validation ve multi-test içine eğitim örneği enjekte edilince durdurma, ayrıca validation–single-test çakışması.

## Doğrulama
- Hedefli `test_split_integration_v25.py + test_split_audit_v24.py + test_dataset.py`: 10/10 başarılı (0.07 sn).
- `test_split_audit_v24.py + test_dataset.py + test_full_pipeline.py`: 11/11 başarılı (17.06 sn).
- Tam pytest paketi birden fazla denemede zaman sınırına takıldı; tüm regresyonlar geçti iddiası yok.

## Sınırlar
- Normalleştirilmiş birebir çakışma denetimi semantik benzerliği tespit etmez.
- G çok konulu başarı kabul eşiği ve fiziksel güç/disk arızası testleri açık.
- Bu sürümde GitHub üzerinde işlem yapılmadı; V24'ün GitHub'a kısmen yüklendiği önceki sonuç geçerlidir.
