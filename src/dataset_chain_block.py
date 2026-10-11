"""
Expanded Multi-Domain Benchmark Suite for Non-Transformer Chain-Block Architecture.

Generates structured test suites including:
1. Multi-domain Turkish/English queries
2. Negated and imperative command pairs
3. Multi-hop relational deduction queries (2 to 5 hops)
4. Out-of-domain / unknown queries (triggering BELİRSİZ_VEYA_BİLİNMİYOR)
5. Overlapping / ambiguous queries (triggering ÇELİŞKİLİ_KANIT)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ChainBlockTestCase:
    id: str
    query: str
    expected_category: str
    expected_block_id: Optional[str]
    is_negated: bool
    expected_decision: str  # "ACCEPT", "BELİRSİZ_VEYA_BİLİNMİYOR", "ÇELİŞKİLİ_KANIT"
    relation_chain: Optional[List[str]] = None  # e.g., ["Aslan", "Memeli", "Hayvan", "Canlı"]


def get_chain_block_benchmark_dataset() -> List[ChainBlockTestCase]:
    cases = []

    # 1. Standard In-Domain Categories (ACCEPT)
    in_domain_samples = [
        ("CB_001", "Merhaba, iyi günler nasılsınız?", "Selamlaşma", "BLOCK_SELAM_01", False, "ACCEPT"),
        ("CB_002", "Dünya hangi yıldızın etrafında döner?", "Gezegenler", "BLOCK_GEZEGEN_01", False, "ACCEPT"),
        ("CB_003", "Aslanlar savanna bölgesinde nasıl avlanır?", "Hayvanlar", "BLOCK_HAYVAN_01", False, "ACCEPT"),
        ("CB_004", "BC337 transistörünün bacak yapısı nedir?", "Elektronik", "BLOCK_ELEKTRONIK_01", False, "ACCEPT"),
        ("CB_005", "12 ile 8 in çarpımı kaçtır?", "Matematik", "BLOCK_MATEMATIK_01", False, "ACCEPT"),
        ("CB_006", "Kütleçekim ivmesi nedir?", "Fizik", "BLOCK_FIZIK_01", False, "ACCEPT"),
        ("CB_007", "Hücre organelleri ve mitokondri görevi", "Biyoloji", "BLOCK_BIYOLOJI_01", False, "ACCEPT"),
        ("CB_008", "İstanbul kaç yılında fethedildi?", "Tarih", "BLOCK_TARIH_01", False, "ACCEPT"),
        ("CB_009", "Araba sevdası romanı kimin eseridir?", "Edebiyat", "BLOCK_EDEBIYAT_01", False, "ACCEPT"),
        # English / Multilingual
        ("CB_010", "Hello, how are you doing today?", "Selamlaşma", "BLOCK_SELAM_01", False, "ACCEPT"),
        ("CB_011", "How many planets orbit around the Sun?", "Gezegenler", "BLOCK_GEZEGEN_01", False, "ACCEPT"),
        ("CB_012", "What is the square root of 144?", "Matematik", "BLOCK_MATEMATIK_01", False, "ACCEPT"),
    ]

    for cid, query, cat, bid, neg, dec in in_domain_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_block_id=bid,
            is_negated=neg, expected_decision=dec
        ))

    # 2. Negated Commands (PROHIBITION / DISAMBIGUATION)
    negated_samples = [
        ("CB_NEG_001", "Buraya gel", "Selamlaşma", "BLOCK_KOMUT_GEL", False, "ACCEPT"),
        ("CB_NEG_002", "Sakın buraya gelme", "Selamlaşma", "BLOCK_KOMUT_GELME", True, "ACCEPT"),
        ("CB_NEG_003", "5 ile 5 i çarp", "Matematik", "BLOCK_MAT_CARP", False, "ACCEPT"),
        ("CB_NEG_004", "5 ile 5 i çarpma", "Matematik", "BLOCK_MAT_CARPMA", True, "ACCEPT"),
        ("CB_NEG_005", "Işığı kapat", "Elektronik", "BLOCK_ELEK_KAPAT", False, "ACCEPT"),
        ("CB_NEG_006", "Işığı kapatma", "Elektronik", "BLOCK_ELEK_KAPATMA", True, "ACCEPT"),
    ]

    for cid, query, cat, bid, neg, dec in negated_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_block_id=bid,
            is_negated=neg, expected_decision=dec
        ))

    # 3. Out-Of-Domain / Unknown (BELİRSİZ_VEYA_BİLİNMİYOR)
    unknown_samples = [
        ("CB_UNK_001", "qwertyuiop asdfghjkl zxcvbnm 123890", "Alakasız", None, False, "BELİRSİZ_VEYA_BİLİNMİYOR"),
        ("CB_UNK_002", "Zırvalamak ve kuantum spatulası ile dondurma satmak", "Alakasız", None, False, "BELİRSİZ_VEYA_BİLİNMİYOR"),
        ("CB_UNK_003", "Xylophone submarine quantum banana pancake", "Alakasız", None, False, "BELİRSİZ_VEYA_BİLİNMİYOR"),
    ]

    for cid, query, cat, bid, neg, dec in unknown_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_block_id=bid,
            is_negated=neg, expected_decision=dec
        ))

    # 4. Ambiguous / High Overlap (ÇELİŞKİLİ_KANIT)
    ambiguous_samples = [
        ("CB_AMB_001", "Matematiksel fizikte ivme ve 12 çarpı 8 hesabının biyolojik hücreye etkisi", "Karışık", None, False, "ÇELİŞKİLİ_KANIT"),
        ("CB_AMB_002", "Elektronik devrede gezegenlerin orbit frekansı tarihi", "Karışık", None, False, "ÇELİŞKİLİ_KANIT"),
    ]

    for cid, query, cat, bid, neg, dec in ambiguous_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_block_id=bid,
            is_negated=neg, expected_decision=dec
        ))

    # 5. Multi-Hop Deduction Cases (2 to 5 hops)
    multihop_samples = [
        ("CB_HOP_2", "Aslan cankurtaran mıdır?", "Hayvanlar", "BLOCK_CANLI", False, "ACCEPT", ["Aslan", "Memeli", "Hayvan"]),
        ("CB_HOP_3", "Aslan hücresel yapıda mıdır?", "Hayvanlar", "BLOCK_HAYVAN", False, "ACCEPT", ["Aslan", "Memeli", "Hayvan", "Canlı"]),
        ("CB_HOP_4", "Güneş enerjisi fotosentez ile biyolojik kütleye dönüşür mü?", "Biyoloji", "BLOCK_KUTLE", False, "ACCEPT", ["Güneş", "Işık", "Fotosentez", "Biyokütle", "Kütle"]),
        ("CB_HOP_5", "Atom çekirdeğinden galaksi kümesine kütleçekim zinciri", "Fizik", "BLOCK_GALAKSI", False, "ACCEPT", ["Quark", "Proton", "Atom", "Yıldız", "Galaksi", "Evren"]),
    ]

    for cid, query, cat, bid, neg, dec, chain in multihop_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_block_id=bid,
            is_negated=neg, expected_decision=dec, relation_chain=chain
        ))

    return cases
