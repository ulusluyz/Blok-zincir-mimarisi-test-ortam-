# Aşama 1 V20 — deterministik özellik sıralaması

## Tespit
`FeatureExtractor.fit()` belge frekansı eşit olan n-gramları yalnızca frekansa göre sıralıyordu. Belge frekansı sözlüğü `set` üzerinden doldurulduğundan eşit frekanslı öğelerin eklenme sırası Python hash seed'e bağlıydı. Bu, ilk `max_features` öğesini, özellik indekslerini ve encoder eğitim girdisini etkileyebilir.

## Düzeltme
Sıralama `(-document_frequency, gram)` anahtarına bağlandı. Frekans eşitliğinde alfabetik sıra sabittir.

## Test
- `tests/test_feature_determinism_v20.py`: iki test geçti; farklı PYTHONHASHSEED süreçleri aynı sözlüğü üretti.
- `tests/test_probe_cost_v18.py` ile birlikte hedefli paket: 3/3 geçti, 2.93 saniye.
- Tam pytest denemesi 140 saniyelik süre sınırına ulaştı. Tam paketin başarılı olduğu iddia edilmiyor.

## Açık işler
- Aynı seed ile tam encoder eğitimi ve G yönlendirmesi çıktılarının süreçler arasında birebir aynı olduğunu doğrulamak.
- Bağımsız çok konulu sorgu setinde probe-5 ve sabit top-3'ü aynı grup ve işlem bütçesiyle kıyaslamak.
- Tam regresyon paketinin neden süre sınırına ulaştığını izole etmek.
- Gerçek disk/güç kesintisi testleri.

GitHub deposu değiştirilmedi.
