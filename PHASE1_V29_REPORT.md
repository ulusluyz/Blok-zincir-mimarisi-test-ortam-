# V29 — tam regresyonun doğrulanması ve güvenilir test çalıştırıcısı

- Tam test paketi bağımsız yeni süreçte çalıştırıldı: **142 passed in 40.10s**; süreç çıkış kodu 0; doğrulayıcı PASS.
- Önceki V28 sürümünün kodu, V29 test çalıştırıcısı eklenmeden önce ayrıca **139 passed in 45.48s** sonucu verdi.
- `verify_phase1.py`: BLAS/OMP/MKL iş parçacıklarını tekleştirir; gerçek test dönüş kodunu kontrol eder; TIMEOUT/FAIL/PASS durumlarını ayrı kaydeder; JSON üretir. Zaman aşımı başarı olarak raporlanmaz.
- Üç test: başarılı süreç, başarısız süreç ve zaman aşımı ayrımı.
- Bu sonuçlar yalnızca yazılım regresyonunu doğrular; bağımsız büyük holdout üzerinde G doğruluğu ve gerçek fiziksel güç/disk kesintisi doğrulanmadığından Aşama 1 nihai kabulü verilmemiştir.
- GitHub'a bu dosyalar aktarılmıştır; **yerel V29 çalışma dizininin tamamı ile depo eşitliği doğrulanmamıştır.**
