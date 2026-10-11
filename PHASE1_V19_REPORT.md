# Aşama 1 V19 — eşit bütçeli çok konulu yönlendirme denetimi

## Yöntem
Beş encoder başlangıç tohumu (11, 23, 42, 67, 101); her tohumda aynı 10 ayrılmış çok konulu sorgu. Statik G, kategori başına üç A bloğu. Toplam 50 eşleştirilmiş değerlendirme; **50 bağımsız sorgu değil**. Kod `compare_probe_v19.py`, ham sonuçlar `PHASE1_V19_REPEATED_SEED_COMPARISON.json`.

## Eşit seçilen grup sayısında sonuçlar

| Yöntem | Grup recall | Tüm kategorileri yakalama | Ortalama A puanlama | Ortalama seçilen grup |
|---|---:|---:|---:|---:|
| Dinamik G | %48,7 | %4 | 5,46 | 1,82 |
| Sabit top-3 | %59,1 | %24 | 9 | 3 |
| Probe-3 | %59,1 | %24 | 18 | 3 |
| Probe-5 | %66,1 | %38 | 24 | 3 |
| Probe-10 | %68,7 | %44 | 39 | 3 |

Probe-3'ün sabit top-3'e karşı **hiç recall kazancı yok**; aynı ilk üç grubu iki kez puanlıyor. Probe-5'in sabit top-3'e karşı artışı 7,0 yüzde puanı; probe-10'un artışı 9,6 yüzde puanı. Probe-5 yaklaşık 2,67 kat, probe-10 yaklaşık 4,33 kat daha fazla A puanlama yapıyor. Bunlar sınırlı deney sonuçlarıdır, üretim başarım garantisi değildir.

## Güvenilirlik ve sınırlamalar
Aynı tohumlarla iki ayrı çalıştırmanın recall değerleri tam aynı çıkmadı. Bu, eğitim/işlem sırası kaynaklı determinizm sorununun araştırılmasını gerektiriyor. Milisaniyelik gecikmeler gürültülü olduğu için karar ölçütü olarak kullanılmamalı. Tam pytest paketi bu turda çalışma süresi sınırına takıldı; başarılı ilan edilmedi. Yeni karşılaştırma yalnızca deney betiği ve rapor ekler, üretim yönlendirme kodunu değiştirmez.

## Karar
Probe yöntemini varsayılan yapma. İlk üç grubu aynı tutan probe-3 gereksiz maliyet yaratıyor. Daha büyük, bağımsız sorgu kümesi ve deterministik eğitim sağlandıktan sonra probe-5 ile top-3 karşılaştırmasını yinele.
