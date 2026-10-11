# Aşama 1 V5 — Bağımsız yerel doğrulama (2026-10-11)

## Uygulanan değişiklik
- `AccessGrantManager` için isteğe bağlı `audit_anchor_path` eklendi.
- Her kalıcı denetim eklemesinden sonra HMAC imzalı `[son sıra, son MAC]` kontrol noktası atomik dosya değiştirme, `fsync` ve 0600 dosya izniyle yazılıyor.
- Başlangıçta SQLite zinciri ve kontrol noktası eşleşmesi kontrol ediliyor. Zincirin sonundan kayıt silme, kontrol noktasını değiştirme ve kontrol noktasının eksik olması reddediliyor.
- SQLite kayıtlarının olduğu konumdan ayrı, erişimi kısıtlanmış bir dizin gereklidir. Tek bir saldırganın anahtarı ve tüm dosyaları kontrol etmesi durumunda bu yöntem mutlak güvenlik sağlamaz.

## Çalıştırılan testler
- `python -m pytest -q --disable-warnings`: **90 passed in 30.20s**.
- Yeni üç saldırı testi: son kayıt silme, kontrol noktası değiştirme, kontrol noktası silme.

## Henüz tamamlanmayan doğrulamalar
- Kalıcı imzalama anahtarının OS secret manager/HSM ile yönetimi, rotasyon ve felaket kurtarma.
- SQLite işlemi ile harici kontrol noktası güncellemesinin güç kesintisine karşı atomik koordinasyonu (çökme halinde fail-closed, fakat otomatik kurtarma yok).
- Tam model kalite benchmark'ı ve bağımsız bilimsel başarım doğrulaması.
- Python aynı süreç içindeki kötü niyetli kodlara karşı donanımsal izolasyon yok.

**Durum: Aşama 1'e koşulsuz %100 onay verilmedi.**
