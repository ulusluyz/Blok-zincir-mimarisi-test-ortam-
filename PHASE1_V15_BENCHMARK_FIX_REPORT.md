# Aşama 1 V15 — Benchmark onarımı ve doğrulama

## Düzeltilen gerçek hatalar
1. `src/benchmark.py` eski `MULTI_TOPIC_TEST_DATA` ismini içe aktarıyordu; güncel veri kümesindeki `UNSEEN_TEST_MULTI_DATASET` sözlük yapısına uyarlandı.
2. Çok konulu ölçümde temel ağın `flat_a_blocks` kaydı eksikti. A+ durum makinesi ikinci sorguda `READ -> SELECTED` hatası veriyordu; ağın blokları kayıt edilerek sorgu sonunda doğru temizleme sağlandı.
3. Benchmark'a `--scales`, `--max-queries`, `--skip-indexes` seçenekleri eklendi. Tam indeks karşılaştırması atlanırsa sonuçlar LSH/IVF/HNSW ölçümü içermez.

## Gerçek çalıştırmalar
- `python -m pytest -q`: **113 passed in 31.19s**.
- `python -m src.benchmark --scales 10 100 1000 --max-queries 3 --skip-indexes`: başarılı; 10 hedefi asgari 50 gerçek A bloğu nedeniyle 50 blok ölçtü.
- `python -m src.benchmark --scales 10000 100000 --max-queries 3 --skip-indexes`: başarılı.
- `python -m src.benchmark --scales 1000 --max-queries 3`: indeks oluşturma ve LSH/IVF ölçümleri dahil başarılı.

| Gerçek blok | Tam tarama ms | G yönlendirme ms | Hızlanma | Bellek MB | İndeksler |
|---:|---:|---:|---:|---:|---|
| 50 | 4.138 | 2.168 | 1.9x | 0.01 | atlandı |
| 100 | 2.402 | 2.129 | 1.1x | 0.05 | atlandı |
| 1000 | 3.941 | 2.528 | 1.6x | 0.79 | atlandı |
| 10000 | 19.514 | 7.376 | 2.6x | 8.26 | atlandı |
| 100000 | 391.498 | 71.283 | 5.5x | 83.05 | atlandı |
| 1000 | 5.238 | 3.263 | 1.6x | 1.19 | dahil |

**Sınırlamalar:** Ölçümler tek çalıştırma, üç sorgu, sentetik bloklar ve bu ortam ile sınırlıdır. 100.000 blokta LSH/IVF/HNSW karşılaştırması yapılmadı; sonuçlar geniş ölçekli doğruluk/recall garantisi veya gerçek disk/elektrik kesintisi testi değildir. Önceki raporlardaki koşulsuz `%100 recall` ve `5.2x` iddiaları bu deneyle genel olarak kanıtlanmış sayılmaz.
