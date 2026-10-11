# Aşama 1 V11 — Yerel anahtar döndürme denemesi

- `src/audit_rotation.py`: Eski HMAC imzalı denetim dönemini koruyarak yeni anahtarla yeni bir denetim dönemi açar.
- `tests/test_audit_rotation.py`: 4/4 yeni test başarılı.
- Önceki 107 testlik tam paket bu turda **tamamlanamadı** (çalıştırma süresi sınırı); 111/111 iddiası yoktur.
- Atomiklik sınırı: eski döneme devir kaydı yazıldıktan sonra yeni dönemin oluşturulması başarısız olursa operatör uzlaştırması gerekir.
- Anahtarların güvenli yedeği, donanım kesintisi testi ve tam model benchmark doğrulaması tamamlanmadı.
- GitHub deposu değiştirilmedi.
