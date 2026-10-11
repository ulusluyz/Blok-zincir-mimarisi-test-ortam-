# V28 — G prototip karşılaştırmasının bağımsız örnek belirsizliği

V27'de 5 encoder tohumu × 10 sorgu, 50 bağımsız sorgu değildir. V28, eşleştirilmiş farklar için sorgu kümeleri üzerinde 10.000 tekrarlı bootstrap uygular. Aynı sorguya ait bütün tohumlar beraber örneklenir; deterministik sabit tohum kullanılır.

- En iyi eğitim üyesi vs mevcut ilk örnek: recall farkı +2,61 yüzde puan.
- Sorgu-küme bootstrap %95 betimsel aralığı: yaklaşık -5,83 ila +12,38 yüzde puan.
- Ortak kategori başarısı farkı +6 yüzde puan, aralık -12 ila +24 yüzde puan.
- 10 bağımsız sorgunun 3'ünde kazanç, diğerlerinde kayıp/beraberlik; küçük test kümesinde üstünlük kanıtlanmadı.
- Adayın aynı test seti üzerinde seçilmiş olması nedeniyle aralıklar doğrulayıcı istatistiksel kanıt değildir.

**Karar:** Daha pahalı prototip varsayılan yapılmadı. Yeni bağımsız ve daha büyük çok konulu veri kümesi, eğitim/veri sızıntısı ayrımı ve önceden tanımlanmış kabul eşiği gerekli. Fiziksel kesinti/disk testleri yapılmadı. GitHub değiştirilmedi.

V28 hedefli test günlüğü `V28_TARGETED_PYTEST.log`; tam test denemesi `V28_FULL_PYTEST.log` dosyasındadır. Tam test tamamlanmadıysa başarı iddiası yoktur.
