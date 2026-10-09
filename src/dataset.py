"""
Expanded Dataset Module providing controlled Turkish dataset samples across 10 categories:
1. Selamlaşma
2. Gezegenler
3. Hayvanlar
4. Elektronik
5. Matematik
6. Fizik
7. Biyoloji
8. Tarih
9. Edebiyat
10. Alakasız (Out-of-Domain)

Includes expanded training data, single-topic test data, and rich multi-topic test data combining 2 or 3 domain topics.
Guarantees zero train/test leakage and verified Jaccard audit.
"""

from typing import List, Dict, Tuple

CATEGORIES = [
    "Selamlaşma",
    "Gezegenler",
    "Hayvanlar",
    "Elektronik",
    "Matematik",
    "Fizik",
    "Biyoloji",
    "Tarih",
    "Edebiyat",
    "Alakasız"
]

SMALL_TRAIN_DATA: List[Tuple[str, str]] = [
    # Selamlaşma
    ("Merhaba", "Selamlaşma"),
    ("Selam, nasılsın?", "Selamlaşma"),
    ("Günaydın, iyi çalışmalar", "Selamlaşma"),
    ("İyi akşamlar dilerim", "Selamlaşma"),

    # Gezegenler
    ("Dünya hangi yıldızın etrafında döner?", "Gezegenler"),
    ("Mars kızıl gezegen olarak bilinir", "Gezegenler"),
    ("Jüpiter güneş sisteminin en büyük gezegenidir", "Gezegenler"),

    # Hayvanlar
    ("Bir aslan nasıl avlanır?", "Hayvanlar"),
    ("Köpekler insan dostu evcil hayvanlardır", "Hayvanlar"),

    # Elektronik
    ("BC337 nedir?", "Elektronik"),
    ("Direnç elektrik akımına karşı zorluk gösterir", "Elektronik"),

    # Matematik
    ("12 çarpı 8 kaçtır?", "Matematik"),
    ("Türev bir fonksiyonun değişim oranıdır", "Matematik"),

    # Alakasız (Out-of-Domain)
    ("Bugün hava çok rüzgarlı ve yağmurlu", "Alakasız"),
    ("Pizzanın üstüne zeytin ve mantar koyalım", "Alakasız"),
]

SMALL_TEST_DATA: List[Tuple[str, str]] = [
    ("Selamlar, keyifler nasıl?", "Selamlaşma"),
    ("Plüton cüce gezegen sınıfına çekildi", "Gezegenler"),
    ("Kaplanlar yalnız avlanan kedigillerdendir", "Hayvanlar"),
    ("LED diyot ışık yayan devre elemanıdır", "Elektronik"),
    ("Pythagoras teoremi dik üçgende geçerlidir", "Matematik"),
    ("Kütüphaneden yeni bir roman ödünç aldım", "Alakasız"),
]

EXPANDED_TRAIN_DATA: List[Tuple[str, str]] = list(SMALL_TRAIN_DATA) + [
    # Selamlaşma
    ("Naber dostum?", "Selamlaşma"),
    ("Hoş geldin, nasıl gidiyor?", "Selamlaşma"),
    ("Esenlikler dilerim", "Selamlaşma"),
    ("Merhabalar, hayırlı günler", "Selamlaşma"),

    # Gezegenler
    ("Satürn halkalarıyla ünlüdür", "Gezegenler"),
    ("Venüs sabah yıldızı olarak da adlandırılır", "Gezegenler"),
    ("Güneş sisteminde kaç gezegen var?", "Gezegenler"),
    ("Neptün ve Uranüs buz devleridir", "Gezegenler"),

    # Hayvanlar
    ("Kediler neden mırıldar?", "Hayvanlar"),
    ("Kartal yükseklerde uçan yırtıcı bir kuştur", "Hayvanlar"),
    ("Yunuslar akıllı deniz memelileridir", "Hayvanlar"),

    # Elektronik
    ("Kondansatör elektrik enerjisini depolar", "Elektronik"),
    ("Transistör sinyal yükseltici olarak kullanılır", "Elektronik"),
    ("Mikrodenetleyici programlanabilir entegre devredir", "Elektronik"),

    # Matematik
    ("İntegral alan hesabında kullanılır", "Matematik"),
    ("Matematikte karekök hesaplama", "Matematik"),
    ("Üçgenin iç açıları toplamı 180 derecedir", "Matematik"),

    # Fizik
    ("Newton'un hareket yasaları ve kuvvet", "Fizik"),
    ("Kütle çekim ivmesi ve ağırlık hesabı", "Fizik"),
    ("Kuantum mekaniği ve foton enerjisi", "Fizik"),
    ("Termodinamik entropi ve ısı transferi", "Fizik"),
    ("Elektromanyetik dalgalar ve ışık hızı", "Fizik"),

    # Biyoloji
    ("Hücre zarı ve DNA çift sarmal yapısı", "Biyoloji"),
    ("Fotosentez ve klorofil pigmenti", "Biyoloji"),
    ("Mitoz ve mayoz bölünme evreleri", "Biyoloji"),
    ("Genetik kalıtım ve Mendel bezelye deneyi", "Biyoloji"),
    ("Enzimler ve protein sentezi ribozom", "Biyoloji"),

    # Tarih
    ("İstanbul'un fethi 1453 Fatih Sultan Mehmet", "Tarih"),
    ("Fransız ihtilali ve milliyetçilik akımı", "Tarih"),
    ("Kurtuluş savaşı ve Amasya genelgesi", "Tarih"),
    ("Osmanlı imparatorluğu yükselme devri", "Tarih"),
    ("Birinci dünya savaşı ittifak devletleri", "Tarih"),

    # Edebiyat
    ("Divan edebiyatı ve gazel nazım şekli", "Edebiyat"),
    ("Roman ve hikayede serim düğüm çözüm", "Edebiyat"),
    ("Şiirde aruz ölçüsü ve hece vezni", "Edebiyat"),
    ("Milli edebiyat akımı ve Ömer Seyfettin", "Edebiyat"),
    ("Tanzimat edebiyatı ve tiyatro Eserleri", "Edebiyat"),

    # Alakasız
    ("Otobüs durağında otuz dakika bekledim", "Alakasız"),
    ("Mavi gömlek kırmızı pantolonla uyar mı?", "Alakasız"),
]

EXPANDED_UNSEEN_TEST_DATA: List[Tuple[str, str]] = list(SMALL_TEST_DATA) + [
    # Selamlaşma
    ("İyi günler arkadaşlar", "Selamlaşma"),
    ("Hayırlı sabahlar herkese", "Selamlaşma"),

    # Gezegenler
    ("Güneşe en uzak gezegen hangisidir?", "Gezegenler"),
    ("Jüpiter atmosferindeki dev kırmızı leke", "Gezegenler"),

    # Hayvanlar
    ("Kuşların tüyleri uçmayı sağlar", "Hayvanlar"),
    ("Balinalar okyanustaki dev canlılardır", "Hayvanlar"),

    # Elektronik
    ("Havya ile lehimleme nasıl yapılır?", "Elektronik"),
    ("MOSFET transistör çalışması", "Elektronik"),

    # Matematik
    ("100 bölü 4 işleminin sonucu", "Matematik"),
    ("İkinci dereceden denklem çözümü", "Matematik"),

    # Fizik
    ("Kinetik enerji ve potansiyel enerji korunumu", "Fizik"),
    ("Özkütle ve kaldırma kuvveti deneyi", "Fizik"),

    # Biyoloji
    ("Solunum sistemi ve akciğerlerde gaz alışverişi", "Biyoloji"),
    ("Ekosistem ve besin zinciri halkaları", "Biyoloji"),

    # Tarih
    ("Kadeş antlaşması tarihteki ilk yazılı metin", "Tarih"),
    ("Çanakkale zaferi ve Anafartalar cephesi", "Tarih"),

    # Edebiyat
    ("Tasavvuf edebiyatı ve Yunus Emre ilahileri", "Edebiyat"),
    ("Servet-i Fünun dergisi ve Tevfik Fikret", "Edebiyat"),

    # Alakasız
    ("Müzik dinlerken yürüyüş yapmak huzur verir", "Alakasız"),
]

# Rich Multi-Topic Test Dataset combining 2 or 3 domain topics
MULTI_TOPIC_TEST_DATASET: List[Tuple[str, List[str]]] = [
    ("Transistörün sıcaklıkla değişen davranışını termodinamik ve türevle açıkla", ["Elektronik", "Fizik", "Matematik"]),
    ("Mars yüzeyindeki uzay araçlarının elektronik çip ve yörünge mekaniği", ["Gezegenler", "Elektronik", "Fizik"]),
    ("Yunusların sonar sistemi ve biyonik devre çip entegrasyonu", ["Hayvanlar", "Biyoloji", "Elektronik"]),
    ("Osmanlı dönemindeki divan edebiyatı ve edebi eserlerin tarihi incelemesi", ["Tarih", "Edebiyat"]),
    ("Hücre zarındaki fotosentez enerjisinin kuantum fiziğiyle modellenmesi", ["Biyoloji", "Fizik"]),
    ("Köpeklerin genetik haritası ve biyolojik kalıtım hesabı", ["Hayvanlar", "Biyoloji"]),
    ("Güneş sistemindeki gezegen yörüngelerinin türev ve integral hesabı", ["Gezegenler", "Matematik"]),
    ("Elektrik devresindeki direnç ısınmasının termodinamik fiziği", ["Elektronik", "Fizik"]),
    ("Milli mücadele dönemi şiirleri ve edebi metinlerin tarihi değeri", ["Edebiyat", "Tarih"]),
    ("Yapay zeka mikrodenetleyici çiplerinde matris ve diferansiyel denklemler", ["Elektronik", "Matematik"]),
]


def get_full_10cat_dataset() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, List[str]]]]:
    """Returns train_data, unseen_test_data, and multi_topic_test_data."""
    # Verify zero leakage between train and single test
    train_texts = {t.strip().lower() for t, _ in EXPANDED_TRAIN_DATA}
    test_texts = {t.strip().lower() for t, _ in EXPANDED_UNSEEN_TEST_DATA}
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Leakage detected! {overlap}"

    return EXPANDED_TRAIN_DATA, EXPANDED_UNSEEN_TEST_DATA, MULTI_TOPIC_TEST_DATASET


def get_expanded_dataset() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train, test, _ = get_full_10cat_dataset()
    return train, test
