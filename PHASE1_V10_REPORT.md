# Aşama 1 V10 — bağımsız yerel doğrulama

- Tam test paketi: **107 passed**, 30.26 saniye (`PYTHONPATH=. python -m pytest -vv --disable-warnings`).
- V9 üzerine 3 yeni fault-injection testi eklendi: yanlış anahtar ile yeniden açma, anchor yazma hatası, SQLite yazma hatası.
- GitHub deposunda değişiklik yapılmadı.

## Kritik sınırlamalar

1. Anchor yazımı SQLite commit'inden ayrı olduğundan işlem atomik değil. Hata durumunda sistem fail-closed davranıyor; manuel kurtarma gerekebilir.
2. Gerçek disk/elektrik kesintisi testi yapılmadı. Yazılımsal fault injection, donanım sertifikasyonu değildir.
3. Kurtarma mekanizması için operatörün olay kanıtını bağımsız doğrulaması gerekir. Meşru tek kayıtlık kesinti ile yetkili anahtarla yapılmış kötü niyetli ekleme tek başına ayırt edilemez.
4. Anahtar döndürme ve harici anahtar kasası entegrasyonu yok.
5. Tam model benchmark'ı bu turda çalıştırılmadı.

**Karar:** 107/107 test geçmesine rağmen Aşama 1 için koşulsuz %100 tamamlandı onayı verilmedi.
