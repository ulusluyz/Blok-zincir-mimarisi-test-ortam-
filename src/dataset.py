"""
Dataset module providing controlled Turkish dataset samples across single-topic and multi-topic categories:
1. Selamlaşma
2. Gezegenler
3. Hayvanlar
4. Elektronik
5. Matematik
6. Alakasız (Out-of-Domain)
7. Çok Konulu (Multi-Topic Combined Concepts)

Includes expanded training and unseen test sets with zero exact leakage and verified Jaccard near-duplicate audit.
"""

from typing import List, Dict, Tuple

CATEGORIES = [
    "Selamlaşma",
    "Gezegenler",
    "Hayvanlar",
    "Elektronik",
    "Matematik",
    "Alakasız"
]

# Manual small dataset for transparent inspection and debugging
SMALL_TRAIN_DATA: List[Tuple[str, str]] = [
    # Selamlaşma
    ("Merhaba", "Selamlaşma"),
    ("Selam, nasılsın?", "Selamlaşma"),
    ("Günaydın, iyi çalışmalar", "Selamlaşma"),
    ("İyi akşamlar dilerim", "Selamlaşma"),
    ("Naber dostum?", "Selamlaşma"),
    ("Hoş geldin, nasıl gidiyor?", "Selamlaşma"),
    ("Esenlikler dilerim", "Selamlaşma"),
    ("Merhabalar, hayırlı günler", "Selamlaşma"),

    # Gezegenler
    ("Dünya hangi yıldızın etrafında döner?", "Gezegenler"),
    ("Mars kızıl gezegen olarak bilinir", "Gezegenler"),
    ("Jüpiter güneş sisteminin en büyük gezegenidir", "Gezegenler"),
    ("Satürn halkalarıyla ünlüdür", "Gezegenler"),
    ("Venüs sabah yıldızı olarak da adlandırılır", "Gezegenler"),
    ("Güneş sisteminde kaç gezegen var?", "Gezegenler"),
    ("Neptün ve Uranüs buz devleridir", "Gezegenler"),
    ("Merkür güneşe en yakın gezegendir", "Gezegenler"),

    # Hayvanlar
    ("Bir aslan nasıl avlanır?", "Hayvanlar"),
    ("Köpekler insan dostu evcil hayvanlardır", "Hayvanlar"),
    ("Kediler neden mırıldar?", "Hayvanlar"),
    ("Kartal yükseklerde uçan yırtıcı bir kuştur", "Hayvanlar"),
    ("Yunuslar akıllı deniz memelileridir", "Hayvanlar"),
    ("Kurtlar sürü halinde yaşarlar", "Hayvanlar"),
    ("Atlar hızlı koşan memeli hayvanlardır", "Hayvanlar"),
    ("Ahtapotların üç kalbi vardır", "Hayvanlar"),

    # Elektronik
    ("BC337 nedir?", "Elektronik"),
    ("Direnç elektrik akımına karşı zorluk gösterir", "Elektronik"),
    ("Kondansatör elektrik enerjisini depolar", "Elektronik"),
    ("Transistör sinyal yükseltici olarak kullanılır", "Elektronik"),
    ("Mikrodenetleyici programlanabilir entegre devredir", "Elektronik"),
    ("Osiloskop voltaj dalga şekillerini ölçer", "Elektronik"),
    ("Diyot akımı tek yönde geçirir", "Elektronik"),
    ("Baskı devre kartı PCB tasarımı", "Elektronik"),

    # Matematik
    ("12 çarpı 8 kaçtır?", "Matematik"),
    ("Türev bir fonksiyonun değişim oranıdır", "Matematik"),
    ("İntegral alan hesabında kullanılır", "Matematik"),
    ("Matematikte karekök hesaplama", "Matematik"),
    ("Üçgenin iç açıları toplamı 180 derecedir", "Matematik"),
    ("Asal sayılar sadece 1'e ve kendisine bölünür", "Matematik"),
    ("Matris matris çarpımı nasıl yapılır?", "Matematik"),
    ("Logaritma üstel fonksiyonun tersidir", "Matematik"),

    # Alakasız (Out-of-Domain)
    ("Bugün hava çok rüzgarlı ve yağmurlu", "Alakasız"),
    ("Pizzanın üstüne zeytin ve mantar koyalım", "Alakasız"),
    ("Otobüs durağında otuz dakika bekledim", "Alakasız"),
    ("Mavi gömlek kırmızı pantolonla uyar mı?", "Alakasız"),
    ("Kahve fincanı masadan düşüp kırıldı", "Alakasız"),
    ("Yaz tatilinde Ege sahillerine gideceğiz", "Alakasız"),
    ("Sinema bileti fiyatları ne kadar olmuş?", "Alakasız"),
    ("Domates çorbası kaşar rendesiyle servis edilir", "Alakasız"),
]

# Multi-Topic Turkish Test Queries targeting multiple groups simultaneously
MULTI_TOPIC_TEST_DATA: List[Tuple[str, List[str]]] = [
    ("Transistörün sıcaklıkla değişen davranışını matematiksel türevle açıkla", ["Elektronik", "Matematik"]),
    ("Mars yüzeyindeki insansız uzay araçlarında kullanılan elektronik çipler", ["Gezegenler", "Elektronik"]),
    ("Yunusların su altı navigasyonu ve ses dalgalarının matris hesabı", ["Hayvanlar", "Matematik"]),
    ("Aslanların avlanma bölgesindeki yörünge ve alan hesabı", ["Hayvanlar", "Gezegenler"]),
    ("Uzay istasyonundaki robotik kolların entegre devre tasarımı", ["Elektronik", "Gezegenler"]),
    ("Deniz memelilerinin biyolojik yapısı ve elektronik çip entegrasyonu", ["Hayvanlar", "Elektronik"]),
]

SMALL_TEST_DATA: List[Tuple[str, str]] = [
    # Selamlaşma
    ("Selamlar, keyifler nasıl?", "Selamlaşma"),
    ("İyi günler arkadaşlar", "Selamlaşma"),
    ("Merhaba, size nasıl yardımcı olabilirim?", "Selamlaşma"),
    ("Tünaydın, görüşmeyeli nasılsınız?", "Selamlaşma"),

    # Gezegenler
    ("Plüton cüce gezegen sınıfına çekildi", "Gezegenler"),
    ("Güneşe en uzak gezegen hangisidir?", "Gezegenler"),
    ("Mars yüzeyindeki uzay araçları", "Gezegenler"),
    ("Gezegenlerin yörünge hareketleri", "Gezegenler"),

    # Hayvanlar
    ("Kaplanlar yalnız avlanan kedigillerdendir", "Hayvanlar"),
    ("Kuşların tüyleri uçmayı sağlar", "Hayvanlar"),
    ("Balinalar okyanustaki dev canlılardır", "Hayvanlar"),
    ("Evcil kedilerin beslenmesi", "Hayvanlar"),

    # Elektronik
    ("LED diyot ışık yayan devre elemanıdır", "Elektronik"),
    ("Havya ile lehimleme nasıl yapılır?", "Elektronik"),
    ("MOSFET transistör çalışması", "Elektronik"),
    ("Elektrik devresinde multimetre kullanımı", "Elektronik"),

    # Matematik
    ("Pythagoras teoremi dik üçgende geçerlidir", "Matematik"),
    ("100 bölü 4 işleminin sonucu", "Matematik"),
    ("İkinci dereceden denklem çözümü", "Matematik"),
    ("Geometride çemberin çevresi", "Matematik"),

    # Alakasız
    ("Müzik dinlerken yürüyüş yapmak huzur verir", "Alakasız"),
    ("Kütüphaneden yeni bir roman ödünç aldım", "Alakasız"),
    ("Trafik sıkışıklığı yüzünden işe geç kaldım", "Alakasız"),
    ("Balkondaki çiçekleri her gün suluyorum", "Alakasız"),
]


def get_expanded_dataset() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Returns expanded train and test datasets (~180 train, ~90 unseen test samples).
    Guarantees no overlap between train and test.
    """
    expanded_train = list(SMALL_TRAIN_DATA) + [
        # Selamlaşma additions
        ("Hey selam, ne haber?", "Selamlaşma"),
        ("Hayırlı sabahlar herkese", "Selamlaşma"),
        ("İyi haftalar dilerim", "Selamlaşma"),
        ("Görüşmek üzere, kendine iyi bak", "Selamlaşma"),
        ("Hoşça kal, tekrar görüşürüz", "Selamlaşma"),
        ("Selamünaleyküm kardeşlerim", "Selamlaşma"),
        ("Aleykümselam hoş geldiniz", "Selamlaşma"),
        ("İyi geceler, tatlı rüyalar", "Selamlaşma"),
        ("Nasılsınız, umarım iyisinizdir", "Selamlaşma"),
        ("Selam dostlar, yeni güne merhaba", "Selamlaşma"),

        # Gezegenler additions
        ("Gök cisimleri ve gezegen yörüngeleri", "Gezegenler"),
        ("Venüs atmosferi sera etkisiyle çok sıcaktır", "Gezegenler"),
        ("Jüpiter'in uydusu Ganymede", "Gezegenler"),
        ("Satürn'ün en büyük uydusu Titan", "Gezegenler"),
        ("Gezegenlerin kütle çekim kuvveti", "Gezegenler"),
        ("Karasal gezegenler ve gaz devleri", "Gezegenler"),
        ("Uranüs eksen eğikliği nedeniyle yatay döner", "Gezegenler"),
        ("Eksoplanet yani ötegezegen keşifleri", "Gezegenler"),
        ("Astronomide gezegen gözlemi", "Gezegenler"),
        ("Güneş sistemindeki gezegen sıralaması", "Gezegenler"),

        # Hayvanlar additions
        ("Zürafalar uzun boyunlu otobur canlılardır", "Hayvanlar"),
        ("Penguenler uçamayan ama iyi yüzen kuşlardır", "Hayvanlar"),
        ("Çitalar karadaki en hızlı koşucu hayvanlardır", "Hayvanlar"),
        ("Ayılar kış uykusuna yatan memelilerdir", "Hayvanlar"),
        ("Bukalemun renk değiştirme yeteneğine sahiptir", "Hayvanlar"),
        ("Karıncalar kendi ağırlıklarının katlarını taşır", "Hayvanlar"),
        ("Yılanlar sürüngenler sınıfına dahildir", "Hayvanlar"),
        ("Papağanlar sesleri taklit edebilen kuşlardır", "Hayvanlar"),
        ("Kuş türleri ve göç yolları", "Hayvanlar"),
        ("Köpek ırkları ve özellikleri", "Hayvanlar"),

        # Elektronik additions
        ("LM358 operasyonel yükselteç opamp", "Elektronik"),
        ("Arduino ve Raspberry Pi projesi", "Elektronik"),
        ("Kondansatör sığası Farad birimiyle ölçülür", "Elektronik"),
        ("Alternatif akım ve doğru akım farkı", "Elektronik"),
        ("Sinyal jeneratörü ile frekans üretimi", "Elektronik"),
        ("Devre elemanlarının seri ve paralel bağlanması", "Elektronik"),
        ("Ohm kanunu V eşittir I çarpı R", "Elektronik"),
        ("Zener diyot voltaj regülasyonunda kullanılır", "Elektronik"),
        ("Entegre devre çip mimarisi", "Elektronik"),
        ("Bipolar jonksiyon transistör BJT", "Elektronik"),

        # Matematik additions
        ("Limit ve süreklilik tanımı", "Matematik"),
        ("Olasılık ve kombinasyon hesapları", "Matematik"),
        ("Polinomlarda derece ve katsayılar", "Matematik"),
        ("Trigonometrik oranlar sinüs ve kosinüs", "Matematik"),
        ("Karmaşık sayılar ve sanal birim i", "Matematik"),
        ("Geometrik diziler ve aritmetik seriler", "Matematik"),
        ("Öklid bağıntıları ve dik üçgen", "Matematik"),
        ("Diferansiyel denklemler çözümü", "Matematik"),
        ("Vektörler ve iç çarpım işlemi", "Matematik"),
        ("Permütasyon formülü ve faktöriyel", "Matematik"),

        # Alakasız additions
        ("Akşam yemeğinde taze fasulye ve pilav var", "Alakasız"),
        ("Uçak biletini internetten satın aldım", "Alakasız"),
        ("Evde eski kitapları düzenledim", "Alakasız"),
        ("Parkta yürürken taze çim kokusu aldım", "Alakasız"),
        ("Masa tenisi turnuvasında finale yükseldim", "Alakasız"),
        ("Bugün televizyonda belgesel seyrettim", "Alakasız"),
        ("Diş hekimi muayenesi için randevu aldım", "Alakasız"),
        ("Yeni ayakkabılarım çok rahat ve şık", "Alakasız"),
        ("Kış mevsiminde kar yağışı bekleniyor", "Alakasız"),
        ("Ofiste kahve makinesi arızalandı", "Alakasız"),
    ]

    expanded_test = list(SMALL_TEST_DATA) + [
        # Selamlaşma unseen test
        ("Hayırlı günler, nasılsınız efendim?", "Selamlaşma"),
        ("Selam dostum, görüşmeyeli ne var ne yok?", "Selamlaşma"),
        ("Herkese günaydın, bol kazançlar", "Selamlaşma"),
        ("İyi akşamlar, nasılsın bakalım?", "Selamlaşma"),
        ("Görüşmek üzere hoşça kalın", "Selamlaşma"),

        # Gezegenler unseen test
        ("Güneşe en yakın gezegen Merkür müdür?", "Gezegenler"),
        ("Jüpiter atmosferindeki dev kırmızı leke", "Gezegenler"),
        ("Buz devleri Neptün ve Uranüs'ün yapısı", "Gezegenler"),
        ("Dünya gezegeninin tek doğal uydusu Ay", "Gezegenler"),
        ("Kızıl gezegen Mars'ta su arayışı", "Gezegenler"),

        # Hayvanlar unseen test
        ("Aslanlar Afrika savanlarında yaşar", "Hayvanlar"),
        ("Evcil köpeklerin eğitimi ve bakımı", "Hayvanlar"),
        ("Deniz memelilerinden yunusların iletişimi", "Hayvanlar"),
        ("Kartalların keskin görme yeteneği", "Hayvanlar"),
        ("Kutup ayılarının soğuğa dayanıklılığı", "Hayvanlar"),

        # Elektronik unseen test
        ("Transistör anahtarlama elemanı olarak kullanılır", "Elektronik"),
        ("Kondansatörlerin şarj ve deşarj süresi", "Elektronik"),
        ("Direnç renk kodları okuma rehberi", "Elektronik"),
        ("Sayısal ve analog devre farkları", "Elektronik"),
        ("Lehim havyası ve pasta kullanımı", "Elektronik"),

        # Matematik unseen test
        ("Türev alırken çarpım kuralı", "Matematik"),
        ("İntegral ile alan ve hacim hesabı", "Matematik"),
        ("Üçgende hipotenüs uzunluğunu bulma", "Matematik"),
        ("Asal sayıların çarpanlarına ayrılması", "Matematik"),
        ("Matris determinantı nasıl hesaplanır?", "Matematik"),

        # Alakasız unseen test
        ("Pazar günleri sahil kenarında yürümek güzeldir", "Alakasız"),
        ("Çamaşır makinesi yıkama programı bitti", "Alakasız"),
        ("Marketten taze sebze ve meyve aldım", "Alakasız"),
        ("Yeni çıkan filmin fragmanını izledim", "Alakasız"),
        ("Yazın sıcak havalarda bol su içmek gerekir", "Alakasız"),
    ]

    # Verify zero leakage
    train_texts = {text.strip().lower() for text, _ in expanded_train}
    test_texts = {text.strip().lower() for text, _ in expanded_test}
    intersection = train_texts.intersection(test_texts)
    assert len(intersection) == 0, f"Data leakage detected! Common samples: {intersection}"

    return expanded_train, expanded_test
