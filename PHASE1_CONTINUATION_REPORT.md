# Aşama 1 — bağımsız devam çalışması (2026-10-11)

## Bu turda yapılanlar
- `BlockNetwork.process_query` beklenmeyen hata üretirse sorgu kimliği ve hata türünü içeren `FAILED` kaydını HMAC ile imzalayıp mevcut SQLite denetim zincirine ekler. Asıl hata yeniden yükseltilir.
- SQLite kalıcı denetimi için açıkça verilen sabit `secret_key` zorunlu; yeniden başlatmada rastgele anahtarla eski kayıtların doğrulanamaması engellenir.
- Hata kaydı ve anahtar yapılandırması için iki yeni regresyon testi eklendi.

## Test
- Bütün otomatik testler: 87 geçti.
- Yeni negatif testler: hata enjeksiyonu sonrası imzalı kalıcı FAILED kaydı ve sabit anahtar zorunluluğu.

## Hâlâ garanti edilmeyenler
- SQLite/HMAC zinciri harici sabitlenmiş son-kayıt özeti (external anchor) olmadan sondan kayıt silmeyi tespit edemez. Harici güven sınırı olmadan %100 bütünlük iddiası doğru olmaz.
- Anahtarın OS secret manager/HSM ile saklanması, rotasyonu ve yedek kurtarma tamamlanmadı.
- Kod içi Python nesneleri ve süreç yönetimi kötü niyetli aynı süreç koduna karşı izolasyon sağlamaz.
- Tam model kalite benchmark'ı yeniden çalıştırılmadı. Bu testler model kalitesini doğrulamaz.
