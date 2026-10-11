"""
Expanded Dataset Module with Train / Validation / Test Splits
and Explicit Acceptable Block ID Ground-Truth Annotations.
"""

from typing import List, Dict, Tuple, Any, Set

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
    ("Merhaba", "Selamlaşma"),
    ("Selam, nasılsın?", "Selamlaşma"),
    ("Dünya hangi yıldızın etrafında döner?", "Gezegenler"),
    ("Mars kızıl gezegen olarak bilinir", "Gezegenler"),
    ("Bir aslan nasıl avlanır?", "Hayvanlar"),
    ("BC337 nedir?", "Elektronik"),
    ("Direnç elektrik akımına karşı zorluk gösterir", "Elektronik"),
    ("12 çarpı 8 kaçtır?", "Matematik"),
    ("Türev bir fonksiyonun değişim oranıdır", "Matematik"),
    ("Bugün hava çok rüzgarlı ve yağmurlu", "Alakasız"),
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
    ("Naber dostum?", "Selamlaşma"),
    ("Hoş geldin, nasıl gidiyor?", "Selamlaşma"),
    ("Esenlikler dilerim", "Selamlaşma"),
    ("Merhabalar, hayırlı günler", "Selamlaşma"),
    ("İyi akşamlar dilerim", "Selamlaşma"),

    ("Jüpiter güneş sisteminin en büyük gezegenidir", "Gezegenler"),
    ("Satürn halkalarıyla ünlüdür", "Gezegenler"),
    ("Venüs sabah yıldızı olarak da adlandırılır", "Gezegenler"),
    ("Güneş sisteminde kaç gezegen var?", "Gezegenler"),
    ("Neptün ve Uranüs buz devleridir", "Gezegenler"),

    ("Köpekler insan dostu evcil hayvanlardır", "Hayvanlar"),
    ("Kediler neden mırıldar?", "Hayvanlar"),
    ("Kartal yükseklerde uçan yırtıcı bir kuştur", "Hayvanlar"),
    ("Yunuslar akıllı deniz memelileridir", "Hayvanlar"),
    ("Kurtlar sürü halinde yaşarlar", "Hayvanlar"),

    ("Kondansatör elektrik enerjisini depolar", "Elektronik"),
    ("Transistör sinyal yükseltici olarak kullanılır", "Elektronik"),
    ("Mikrodenetleyici programlanabilir entegre devredir", "Elektronik"),
    ("Osiloskop voltaj dalga şekillerini ölçer", "Elektronik"),
    ("Diyot akımı tek yönde geçirir", "Elektronik"),

    ("İntegral alan hesabında kullanılır", "Matematik"),
    ("Matematikte karekök hesaplama", "Matematik"),
    ("Üçgenin iç açıları toplamı 180 derecedir", "Matematik"),
    ("Asal sayılar sadece 1'e ve kendisine bölünür", "Matematik"),
    ("Logaritma üstel fonksiyonun tersidir", "Matematik"),

    ("Newton'un hareket yasaları ve kuvvet", "Fizik"),
    ("Kütle çekim ivmesi ve ağırlık hesabı", "Fizik"),
    ("Kuantum mekaniği ve foton enerjisi", "Fizik"),
    ("Termodinamik entropi ve ısı transferi", "Fizik"),
    ("Elektromanyetik dalgalar ve ışık hızı", "Fizik"),

    ("Hücre zarı ve DNA çift sarmal yapısı", "Biyoloji"),
    ("Fotosentez ve klorofil pigmenti", "Biyoloji"),
    ("Mitoz ve mayoz bölünme evreleri", "Biyoloji"),
    ("Genetik kalıtım ve Mendel bezelye deneyi", "Biyoloji"),
    ("Enzimler ve protein sentezi ribozom", "Biyoloji"),

    ("İstanbul'un fethi 1453 Fatih Sultan Mehmet", "Tarih"),
    ("Fransız ihtilali ve milliyetçilik akımı", "Tarih"),
    ("Kurtuluş savaşı ve Amasya genelgesi", "Tarih"),
    ("Osmanlı imparatorluğu yükselme devri", "Tarih"),
    ("Birinci dünya savaşı ittifak devletleri", "Tarih"),

    ("Divan edebiyatı ve gazel nazım şekli", "Edebiyat"),
    ("Roman ve hikayede serim düğüm çözüm", "Edebiyat"),
    ("Şiirde aruz ölçüsü ve hece vezni", "Edebiyat"),
    ("Milli edebiyat akımı ve Ömer Seyfettin", "Edebiyat"),
    ("Tanzimat edebiyatı ve tiyatro Eserleri", "Edebiyat"),

    ("Pizzanın üstüne zeytin ve mantar koyalım", "Alakasız"),
    ("Otobüs durağında otuz dakika bekledim", "Alakasız"),
    ("Mavi gömlek kırmızı pantolonla uyar mı?", "Alakasız"),
    ("Kahve fincanı masadan düşüp kırıldı", "Alakasız"),
    ("Yaz tatilinde Ege sahillerine gideceğiz", "Alakasız"),
]

# Validation Set (For Hyperparameter tuning & Phase 3 Rollback checks)
VAL_DATASET: List[Dict[str, Any]] = [
    {"text": "Tünaydın nasılsınız efendim?", "category": "Selamlaşma", "acceptable_block_ids": ["kb_Selamlaşma_0", "kb_Selamlaşma_1"]},
    {"text": "Güneşe en yakın gezegen Merkür müdür?", "category": "Gezegenler", "acceptable_block_ids": ["kb_Gezegenler_0", "kb_Gezegenler_1"]},
    {"text": "Evcil köpeklerin beslenmesi ve eğitimi", "category": "Hayvanlar", "acceptable_block_ids": ["kb_Hayvanlar_0", "kb_Hayvanlar_1"]},
    {"text": "Sayısal ve analog devre kartı farkları", "category": "Elektronik", "acceptable_block_ids": ["kb_Elektronik_0", "kb_Elektronik_1"]},
    {"text": "Matris determinantı ve matris çarpımı", "category": "Matematik", "acceptable_block_ids": ["kb_Matematik_0", "kb_Matematik_1"]},
    {"text": "Kinetik enerji ve potansiyel enerji denklem hesabı", "category": "Fizik", "acceptable_block_ids": ["kb_Fizik_0", "kb_Fizik_1"]},
    {"text": "Solunum ve dolaşım sisteminde oksijen taşınması", "category": "Biyoloji", "acceptable_block_ids": ["kb_Biyoloji_0", "kb_Biyoloji_1"]},
    {"text": "Çanakkale cephesi ve Anafartalar zaferi", "category": "Tarih", "acceptable_block_ids": ["kb_Tarih_0", "kb_Tarih_1"]},
    {"text": "Servet-i Fünun dergisi ve Tevfik Fikret şiirleri", "category": "Edebiyat", "acceptable_block_ids": ["kb_Edebiyat_0", "kb_Edebiyat_1"]},
    {"text": "Balkondaki saksı çiçeklerini düzenledim", "category": "Alakasız", "acceptable_block_ids": ["kb_Alakasız_0", "kb_Alakasız_1"]},
]

# Unseen Test Dataset with Block ID Ground Truth Annotations
UNSEEN_TEST_SINGLE_DATASET: List[Dict[str, Any]] = [
    {"text": "Hayırlı günler, nasılsınız efendim?", "category": "Selamlaşma", "acceptable_block_ids": ["kb_Selamlaşma_0", "kb_Selamlaşma_1", "kb_Selamlaşma_2"]},
    {"text": "Kızıl gezegen Mars yüzeyinde uzay keşfi", "category": "Gezegenler", "acceptable_block_ids": ["kb_Gezegenler_0", "kb_Gezegenler_1", "kb_Gezegenler_2"]},
    {"text": "Afrika savanlarında yaşayan vahşi kedigiller", "category": "Hayvanlar", "acceptable_block_ids": ["kb_Hayvanlar_0", "kb_Hayvanlar_1", "kb_Hayvanlar_2"]},
    {"text": "Lehim havyası ve pasta ile devre kartı tamiri", "category": "Elektronik", "acceptable_block_ids": ["kb_Elektronik_0", "kb_Elektronik_1", "kb_Elektronik_2"]},
    {"text": "İkinci dereceden denklem köklerini bulma", "category": "Matematik", "acceptable_block_ids": ["kb_Matematik_0", "kb_Matematik_1", "kb_Matematik_2"]},
    {"text": "Özkütle ve suyun kaldırma kuvveti deneyi", "category": "Fizik", "acceptable_block_ids": ["kb_Fizik_0", "kb_Fizik_1", "kb_Fizik_2"]},
    {"text": "Ekosistemdeki besin zinciri ve klorofil", "category": "Biyoloji", "acceptable_block_ids": ["kb_Biyoloji_0", "kb_Biyoloji_1", "kb_Biyoloji_2"]},
    {"text": "Kadeş antlaşması tarihteki ilk yazılı barış metni", "category": "Tarih", "acceptable_block_ids": ["kb_Tarih_0", "kb_Tarih_1", "kb_Tarih_2"]},
    {"text": "Tasavvuf edebiyatında Yunus Emre ilahileri", "category": "Edebiyat", "acceptable_block_ids": ["kb_Edebiyat_0", "kb_Edebiyat_1", "kb_Edebiyat_2"]},
    {"text": "Pazar günleri sahil kenarında yürüyüş yapmak", "category": "Alakasız", "acceptable_block_ids": ["kb_Alakasız_0", "kb_Alakasız_1", "kb_Alakasız_2"]},
]

# Rich Multi-Topic Test Dataset with Group & Block ID Ground Truth Annotations
UNSEEN_TEST_MULTI_DATASET: List[Dict[str, Any]] = [
    {
        "text": "Transistörün sıcaklıkla değişen davranışını termodinamik ve türevle açıkla",
        "expected_categories": ["Elektronik", "Fizik", "Matematik"],
        "acceptable_block_ids": ["kb_Elektronik_0", "kb_Fizik_0", "kb_Matematik_0"]
    },
    {
        "text": "Mars yüzeyindeki uzay araçlarının elektronik çip ve yörünge mekaniği",
        "expected_categories": ["Gezegenler", "Elektronik", "Fizik"],
        "acceptable_block_ids": ["kb_Gezegenler_0", "kb_Elektronik_0", "kb_Fizik_0"]
    },
    {
        "text": "Yunusların sonar sistemi ve biyonik devre çip entegrasyonu",
        "expected_categories": ["Hayvanlar", "Biyoloji", "Elektronik"],
        "acceptable_block_ids": ["kb_Hayvanlar_0", "kb_Biyoloji_0", "kb_Elektronik_0"]
    },
    {
        "text": "Osmanlı dönemindeki divan edebiyatı ve edebi eserlerin tarihi incelemesi",
        "expected_categories": ["Tarih", "Edebiyat"],
        "acceptable_block_ids": ["kb_Tarih_0", "kb_Edebiyat_0"]
    },
    {
        "text": "Hücre zarındaki fotosentez enerjisinin kuantum fiziğiyle modellenmesi",
        "expected_categories": ["Biyoloji", "Fizik"],
        "acceptable_block_ids": ["kb_Biyoloji_0", "kb_Fizik_0"]
    },
    {
        "text": "Köpeklerin genetik haritası ve biyolojik kalıtım hesabı",
        "expected_categories": ["Hayvanlar", "Biyoloji"],
        "acceptable_block_ids": ["kb_Hayvanlar_0", "kb_Biyoloji_0"]
    },
    {
        "text": "Güneş sistemindeki gezegen yörüngelerinin türev ve integral hesabı",
        "expected_categories": ["Gezegenler", "Matematik"],
        "acceptable_block_ids": ["kb_Gezegenler_0", "kb_Matematik_0"]
    },
    {
        "text": "Elektrik devresindeki direnç ısınmasının termodinamik fiziği",
        "expected_categories": ["Elektronik", "Fizik"],
        "acceptable_block_ids": ["kb_Elektronik_0", "kb_Fizik_0"]
    },
    {
        "text": "Milli mücadele dönemi şiirleri ve edebi metinlerin tarihi değeri",
        "expected_categories": ["Edebiyat", "Tarih"],
        "acceptable_block_ids": ["kb_Edebiyat_0", "kb_Tarih_0"]
    },
    {
        "text": "Yapay zeka mikrodenetleyici çiplerinde matris ve diferansiyel denklemler",
        "expected_categories": ["Elektronik", "Matematik"],
        "acceptable_block_ids": ["kb_Elektronik_0", "kb_Matematik_0"]
    },
]


def get_full_split_datasets():
    """
    Returns (train_data, val_data, test_single_data, test_multi_data).
    Verifies normalized exact leakage across all four partitions.
    """
    from src.split_audit import assert_no_split_leakage
    assert_no_split_leakage(EXPANDED_TRAIN_DATA, VAL_DATASET, UNSEEN_TEST_SINGLE_DATASET, UNSEEN_TEST_MULTI_DATASET)
    return EXPANDED_TRAIN_DATA, VAL_DATASET, UNSEEN_TEST_SINGLE_DATASET, UNSEEN_TEST_MULTI_DATASET


def get_full_10cat_dataset():
    train, val, single_test, multi_test = get_full_split_datasets()
    single_tuples = [(item["text"], item["category"]) for item in single_test]
    multi_tuples = [(item["text"], item["expected_categories"]) for item in multi_test]
    return train, single_tuples, multi_tuples


def get_expanded_dataset() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train, val, single_test, multi_test = get_full_split_datasets()
    single_tuples = [(item["text"], item["category"]) for item in single_test]
    return train, single_tuples
