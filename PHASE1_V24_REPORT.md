# Aşama 1 V24 — veri ayrımı güvenlik denetimi

## Değişiklik
- `src/split_audit.py`: train/validation/test_single/test_multi arasında NFKC + casefold + sözcük bazlı normalize edilmiş metin çakışmalarını saptar.
- `tests/test_split_audit_v24.py`: dört yeni test. 4/4 başarılı (0,05 saniye).
- Mevcut veri kümesinde normalleştirilmiş birebir çakışma yok.

## Sınırlar
- Bu denetim semantik benzerliği veya paraphrase veri sızıntısını saptamaz.
- Tam pytest denemesi çalışma süresi sınırında durdu; V24 için 130/130 iddia edilmiyor.
- G çok konulu kalitesi, gerçek elektrik kesintisi ve fiziksel disk arızası doğrulanmadı.
- GitHub değiştirilmedi.
