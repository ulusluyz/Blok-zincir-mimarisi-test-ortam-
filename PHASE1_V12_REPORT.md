# Aşama 1 V12 — Bağımsız yerel geliştirme

- Kaynak: V11 yerel sürümü.
- Yeni `resume_interrupted_rotation`: Eski dönemdeki son HMAC imzalı devir kaydını doğrular; yeni dönem dosyaları henüz yoksa aynı devri ikinci kez yazmadan yeni dönemi başlatır.
- Reddedilen durumlar: eksik/uyuşmayan devir kaydı, farklı hedef, aynı anahtar, mevcut/kısmi hedef dosyaları, doğrulanamayan eski zincir.
- İki yeni regresyon testi eklendi.
- İlgili güvenlik testleri: 29/29 PASS (1.90 sn).
- V11 tam test paketi daha önce süre sınırına takıldı; V12 tam paketinin tamamı henüz doğrulanmadı.
- Gerçek güç kesintisi ve disk arızası donanım testleri yapılmadı.
- GitHub deposuna yazılmadı.
