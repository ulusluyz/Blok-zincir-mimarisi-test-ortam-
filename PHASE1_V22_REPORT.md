# Aşama 1 V22 — tekrarlanabilir çok konulu yönlendirme karşılaştırması

## Deney

V21 kaynak kodu değiştirilmeden `compare_probe_v19.py` iki bağımsız süreçte, `PYTHONHASHSEED=1` ve `PYTHONHASHSEED=777` ile çalıştırıldı. Her çalıştırmada aynı 5 encoder tohumu (11, 23, 42, 67, 101) ve aynı 10 çok konulu sorgu değerlendirildi. 50 eşleştirilmiş gözlem, 50 bağımsız sorgu değildir. İki çalıştırmada tüm yöntemlerin recall, joint ve grup seçim/hit sonuçları birebir aynı çıktı. G statiktir; sonuçlar öğrenilebilir G'ye genellenemez.

| Yöntem | Grup recall | Tüm konuları bulma | Ortalama A puanlama | Seçilen grup |
|---|---:|---:|---:|---:|
| Dinamik G | %50,43 | %12 | 5,58 | 1,86 |
| Sabit Top-3 | %66,09 | %36 | 9 | 3 |
| Probe-3 | %66,09 | %36 | 18 | 3 |
| Probe-5 | %67,83 | %42 | 24 | 3 |
| Probe-10 | %68,70 | %44 | 39 | 3 |

Probe-5, sabit Top-3'e karşı 50 eşleştirilmiş örneğin 6'sında daha iyi, 4'ünde daha kötü, 40'ında eşit; toplam +2 kategori isabeti sağladı. Probe-10 için 7 daha iyi, 4 daha kötü, 39 eşit; toplam +3 isabet. Probe-5 yaklaşık 2,67 kat, Probe-10 4,33 kat A puanlama işlemi gerektiriyor. Bu sınırlı veri kümesinde ek maliyete karşılık doğruluk kazancı küçük; yöntemi varsayılan yapmaya yeterli kanıt yok.

## Doğrulama

- Hedefli testler: 10/10 başarılı, 8,44 saniye.
- Tam test paketi: 110 saniyelik çalışma sınırına takıldı; tamamlandı/başarılı denilemez.
- İki çalıştırmanın milisaniye gecikmeleri farklı olabileceğinden deterministiklik kıyasına dahil edilmedi.
- Gerçek donanım kesintisi testleri yapılmadı.

## Karar

Aşama 1 tamamlanmadı. Probe-3 gereksiz hesaplama yapıyor; Probe-5/10 varsayılan yapılmamalı. Daha büyük ve bağımsız sorgu kümesi, eşit bütçeli deney, istatistiksel belirsizlik ve öğrenilebilir G için ayrı değerlendirme gerekiyor. GitHub deposu değiştirilmedi.
