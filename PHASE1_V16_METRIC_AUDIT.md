# Aşama 1 V16 — Doğruluk metriği denetimi

## Kod değişiklikleri
- Top-3, bloklar arasından ilk üç **benzersiz kategori** ile hesaplanır. Önceki ölçüm ilk üç blok üzerinden yapılıyordu.
- Çok sınıflı sınıflandırmadan türetilen sözde FP/FN oranları kaldırıldı (JSON `null`). Bunlar geçerli ikili hata oranları değildi.
- G yönlendirme için değerlendirilen/elenen A blokları artık bütün sorguların ortalamasıdır; yalnızca son sorgu kullanılmaz.
- Üç yeni regresyon testi eklendi.

## Doğrulama
- `PYTHONPATH=. python -m pytest -q`: **116 passed in 31.25s**.
- `PYTHONPATH=. python -m src.benchmark --scales 1000 --max-queries 10 --skip-indexes`: tamamlandı.

## 1000 blok / 10 sorgu (bu çalıştırmaya özgü)
| Yöntem | Top-1 | Top-3 farklı kategori | Ortalama ms |
|---|---:|---:|---:|
| Tam tarama | 0.40 | 0.90 | 3.178 |
| İki seviyeli G | 0.40 | 0.70 | 2.508 |
| Çok seviyeli G | 0.40 | 0.40 | 2.677 |

Çok konulu sorgularda ortalama grup geri çağırma: **0.5167**. Bu değer, mimarinin çok konulu doğruluk hedefinin henüz karşılanmadığını gösterir. İki seviyeli yönlendirme burada yaklaşık 1.27x hızlanırken Top-3 doğruluğu tam taramadan düşüktür. Sonuçlar küçük sorgu örneği ve sentetik veri nedeniyle genellenemez. LSH/IVF/HNSW bu çalıştırmada ölçülmedi.

## Açık konular
- Çok konulu geri çağırma ve çok seviyeli doğruluk düşük; algoritmik iyileştirme ve daha geniş bağımsız test gerekiyor.
- Gerçek güç kesintisi/disk arızası testi yapılmadı.
- Nihai Aşama 1 onayı verilmedi.
