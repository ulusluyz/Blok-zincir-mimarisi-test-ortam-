# Aşama 1 — V23 bağımsız karar raporu

## Doğrulama

- Komut: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`
- Sonuç: **126 passed in 39.13s** (bkz. `V23_FULL_PYTEST.log`).
- V22 kaynak kodu değiştirilmedi. V23 doğrulama ve karar paketidir.

## G yönlendirmesi: mevcut deliller

V22'nin 5 encoder tohumu x 10 sabit çok konulu sorgu deneyinde, eşit grup bütçeli sabit Top-3 geri çağırma %66,09; evidence_probe-5 %67,83; evidence_probe-10 %68,70 idi. Probe-5, sabit Top-3'e göre yalnızca 1,74 yüzde puanı iyileşti ve daha fazla A puanlama işlemi gerektirdi. Bu, varsayılanı değiştirmek için güçlü bir gerekçe değildir. 50 değerlendirme, 50 bağımsız sorgu değildir; aynı 10 sorgu farklı encoder başlangıçlarıyla tekrar edilmiştir.

## Kabul kararı

- **Kod regresyonu: GEÇTİ** (126/126).
- **Deterministik küçük deneyler: GEÇTİ** (önceki V21–V22 raporları kapsamında).
- **G çok konulu kalite eşiği: ONAYLANMADI** (önceden tanımlı bağımsız kabul eşiği ve daha büyük ayrık test kümesi yok).
- **Gerçek elektrik kesintisi / fiziksel disk arızası: TEST EDİLMEDİ**.
- **Anahtar kurtarma / üretim operasyonu: UÇTAN UCA ONAYLANMADI**.

Sonuç: Aşama 1'i %100 tamamlandı diye etiketlemek doğru değildir. Aşama 2'ye geçiş için önceden tanımlanmış başarı eşiği, bağımsız çok konulu holdout veri, maliyet/doğruluk karşılaştırması ve operasyonel hata senaryolarının tamamlanması gerekir.

## En kısa güvenilir devam planı

1. Bağımsız çok konulu test kümesi oluştur (önceden belirlenmiş, eğitimden ayrık); veri sızıntısı kontrol et.
2. Sabit Top-3 ile öğrenilebilir G'yi **eşit grup ve A puanlama bütçesinde** karşılaştır; seed başına sonuçları ve belirsizliği raporla.
3. Varsayılan yönlendirmeyi yalnızca önceden belirlenmiş kalite/maliyet eşiğini geçerse değiştir.
4. Kesinti/kurtarma ve anahtar prosedürlerini gerçek ortamda kontrollü doğrula; fiziksel test yapılmadıysa açıkça belirt.
