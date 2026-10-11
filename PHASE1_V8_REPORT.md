# Aşama 1 V8 — Kontrollü güç kesintisi kurtarması

- `AccessGrantManager.recover_interrupted_anchor(anchor_path)` eklendi.
- Kurtarma **yalnızca açık operatör çağrısıyla** ve **DB-only doğrulama oturumunda** yapılır.
- HMAC denetim zinciri tamamen doğrulanır, eski kontrol noktası imzası doğrulanır, kontrol noktası zincir ucundan **tam bir kayıt** geride olmalıdır ve önceki MAC ile eşleşmelidir.
- Başarılı durumda kontrol noktası atomik dosya değiştirme ve fsync ile yeniden yazılır; zincir yeniden doğrulanır.
- Sahte kontrol noktası, birden fazla kayıt farkı ve değiştirilmiş zincir reddedilir.
- Test: `python -m pytest -q --disable-warnings` → **99 passed in 29.07s**.

## Güvenlik sınırları
- Bu bir *yazılım simülasyonu* ve dar kapsamlı kurtarma yöntemidir; fiziksel elektrik kesintisi, disk flush sıralaması, WAL bozulması, key rotation, işletim sistemi anahtar kasası veya tam model benchmark doğrulanmamıştır.
- Kurtarma işlemi öncesinde olay kaydı, bağımsız yedek ve operatör doğrulaması gerekir. İmzalama anahtarı saldırganın elindeyse HMAC güvenlik sağlamaz.
- GitHub'a değişiklik yapılmadı.
