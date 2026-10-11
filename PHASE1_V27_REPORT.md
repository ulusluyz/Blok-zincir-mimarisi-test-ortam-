# V27 — G prototipleri için bağımsız karşılaştırma

5 encoder başlangıç tohumu (11,23,42,67,101), 10 sabit çok konulu test sorgusu, Top-3 G seçimi. 50 eşleştirilmiş ölçüm, 50 bağımsız sorgu değil. Prototipler sadece eğitim verisinden oluşturuldu.

| G temsilcisi | Kategori recall | Tüm kategorileri bulma | G benzerlik işlemi/sorgu |
|---|---:|---:|---:|
| İlk eğitim örneği (mevcut) | %66,09 | %36 | 10 |
| Eğitim medoid'i | %65,22 | %38 | 10 |
| Bit çoğunluk prototipi | %65,22 | %38 | 10 |
| En iyi eğitim üyesi (6/ kategori) | %68,70 | %42 | 60 |

Medoid ve çoğunluk temsili recall'u düşürdü; varsayılan G değiştirilmedi. Üye maksimumu yalnızca +2,61 yüzde puan recall sağlıyor fakat G karşılaştırma sayısını 6 katına çıkarıyor. Bu nedenle varsayılan olarak uygulanmadı. Bu sonuçlar çok küçük 10 sorguluk sabit veri kümesine bağlıdır; üretim performansını kanıtlamaz.

Tam regresyon testi bu çalıştırmada zaman sınırında tamamlanamadı; PASS iddiası yok. Fiziksel kesinti ve disk arızası testleri de yapılmadı. GitHub değiştirilmedi.
