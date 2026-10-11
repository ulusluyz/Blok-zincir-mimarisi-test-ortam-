# Yerel bağımsız düzeltme ve test raporu

Bu paket kullanıcının yüklediği ZIP kopyasından oluşturuldu; GitHub deposu değiştirilmedi.

## Uygulanan düzeltmeler

1. `BlockNetwork.process_query()` tüm sorgu işlemi boyunca `threading.RLock` ile korunur. Bu güvenli ama **seri** yürütme sağlar; paralel sorgu hızlandırması sağlamaz.
2. Sorgu denetimi, A+ `access_logs` üzerindeki sorgu-başlangıç ofsetlerinden türetilir; kümülatif sayaçlar sorgu metriği olarak kullanılmaz.
3. Denetleyici `selected`, `authorized`, `read` ve gerçek `access_events` kayıtlarını karşılaştırır; eksik/tutarsız kanıtta `NOT_EVALUATED`, kanıtlanmış yetkisiz okumada `FAIL` üretir.
4. Benchmark'ta boş olumsuzluk ve çok-adımlı çıkarım kümeleri artık `%100 PASS` oluşturmaz; ilgili kriterler `NOT_EVALUATED` olur. OOD ve A+ audit değerlendirme durumları da kriterlere taşınır.
5. Mevcut denetim testine gerçek erişim olayları eklendi; dört bağımsız regresyon testi eklendi.

## Testler

`PYTHONPATH=. pytest -q tests`: **76 passed** (18.79s).

Ek stres testi: 8 iş parçacığı x 40 sorgu = 320 sorgu; sorgu ID ve A+ erişim denetimi doğrulandı.

## Sınırlar

- `RLock` eşzamanlı sorguları sıraya alır; tam paralel mimari değildir.
- HMAC tokenların yaşam döngüsü hâlâ daha güçlü merkezi kayıt, son kullanma zamanı ve atomik tek kullanımlık tüketim ile geliştirilebilir.
- `access_logs` Python içindeki değiştirilebilir listelerdir; kötü niyetli kod tarafından değiştirilmesine karşı bağımsız/kurcalamaya dayanıklı kanıt deposu değildir.
- Tam model benchmark'ı bu yerel düzeltme turunda yeniden çalıştırılmadı. Eski `VERIFIED_RESULTS.json` ve diğer raporlar güncel sonuç olarak sunulmamalıdır.
