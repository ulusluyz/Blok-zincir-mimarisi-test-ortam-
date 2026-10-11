"""
Expanded Multi-Domain Benchmark Suite for Non-Transformer Chain-Block Architecture.

Generates rigorous, leakage-free benchmark test suites including:
1. Single-topic in-domain queries
2. Multi-topic queries (for Precision@k, Recall@k, mAP@k)
3. Unseen negated & imperative commands
4. True Out-Of-Distribution (OOD) unknown queries (expecting BELİRSİZ_VEYA_BİLİNMİYOR)
5. Overlapping / ambiguous queries (expecting ÇELİŞKİLİ_KANIT)
6. Multi-hop relational deduction queries (2 to 5 hops)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class ChainBlockTestCase:
    id: str
    query: str
    expected_category: Optional[str]               # Single target category or None if OOD
    expected_categories: Optional[List[str]]      # Multi-target categories if multi-topic
    expected_block_id: Optional[str]
    is_negated: bool
    is_ood: bool                                  # True Out-Of-Distribution query flag
    expected_decision: str                        # "ACCEPT", "BELİRSİZ_VEYA_BİLİNMİYOR", "ÇELİŞKİLİ_KANIT"
    relation_chain: Optional[List[str]] = None     # Symbolic ground truth (for evaluation only, NOT passed to inference)


def get_chain_block_benchmark_dataset() -> List[ChainBlockTestCase]:
    cases = []

    # 1. Standard Single-Topic In-Domain Queries (ACCEPT)
    in_domain_samples = [
        ("CB_001", "Hayırlı günler, nasılsınız efendim?", "Selamlaşma", ["Selamlaşma"], "BLOCK_SELAM_01", False, False, "ACCEPT"),
        ("CB_002", "Kızıl gezegen Mars yüzeyinde uzay keşfi", "Gezegenler", ["Gezegenler"], "BLOCK_GEZEGEN_01", False, False, "ACCEPT"),
        ("CB_003", "Afrika savanlarında yaşayan vahşi kedigiller", "Hayvanlar", ["Hayvanlar"], "BLOCK_HAYVAN_01", False, False, "ACCEPT"),
        ("CB_004", "Lehim havyası ve pasta ile devre kartı tamiri", "Elektronik", ["Elektronik"], "BLOCK_ELEKTRONIK_01", False, False, "ACCEPT"),
        ("CB_005", "İkinci dereceden denklem köklerini bulma", "Matematik", ["Matematik"], "BLOCK_MATEMATIK_01", False, False, "ACCEPT"),
        ("CB_006", "Özkütle ve suyun kaldırma kuvveti deneyi", "Fizik", ["Fizik"], "BLOCK_FIZIK_01", False, False, "ACCEPT"),
        ("CB_007", "Ekosistemdeki besin zinciri ve klorofil", "Biyoloji", ["Biyoloji"], "BLOCK_BIYOLOJI_01", False, False, "ACCEPT"),
        ("CB_008", "Kadeş antlaşması tarihteki ilk yazılı barış metni", "Tarih", ["Tarih"], "BLOCK_TARIH_01", False, False, "ACCEPT"),
        ("CB_009", "Tasavvuf edebiyatında Yunus Emre ilahileri", "Edebiyat", ["Edebiyat"], "BLOCK_EDEBIYAT_01", False, False, "ACCEPT"),
        ("CB_010", "Pazar günleri sahil kenarında yürüyüş yapmak", "Alakasız", ["Alakasız"], "BLOCK_ALAKASIZ_01", False, False, "ACCEPT"),
    ]

    for cid, query, cat, cats, bid, neg, ood, dec in in_domain_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_categories=cats,
            expected_block_id=bid, is_negated=neg, is_ood=ood, expected_decision=dec
        ))

    # 2. Multi-Topic Queries (Precision@k, Recall@k, mAP@k)
    multi_topic_samples = [
        ("CB_MUL_001", "Transistörün sıcaklıkla değişen davranışını termodinamik ve türevle açıkla", None, ["Elektronik", "Fizik", "Matematik"], "BLOCK_ELEKTRONIK_01", False, False, "ACCEPT"),
        ("CB_MUL_002", "Mars yüzeyindeki uzay araçlarının elektronik çip ve yörünge mekaniği", None, ["Gezegenler", "Elektronik", "Fizik"], "BLOCK_GEZEGEN_01", False, False, "ACCEPT"),
        ("CB_MUL_003", "Yunusların sonar sistemi ve biyonik devre çip entegrasyonu", None, ["Hayvanlar", "Biyoloji", "Elektronik"], "BLOCK_HAYVAN_01", False, False, "ACCEPT"),
        ("CB_MUL_004", "Osmanlı dönemindeki divan edebiyatı ve edebi eserlerin tarihi incelemesi", None, ["Tarih", "Edebiyat"], "BLOCK_TARIH_01", False, False, "ACCEPT"),
    ]

    for cid, query, cat, cats, bid, neg, ood, dec in multi_topic_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_categories=cats,
            expected_block_id=bid, is_negated=neg, is_ood=ood, expected_decision=dec
        ))

    # 3. Unseen Negated Commands (Using UNSEEN verbs, subjects, objects)
    unseen_negated_samples = [
        ("CB_NEG_001", "Pencereyi hemen aç", "Selamlaşma", ["Selamlaşma"], "BLOCK_KOMUT_AC", False, False, "ACCEPT"),
        ("CB_NEG_002", "Sakın o pencereyi açma", "Selamlaşma", ["Selamlaşma"], "BLOCK_KOMUT_ACMA", True, False, "ACCEPT"),
        ("CB_NEG_003", "Ağaçtaki elmaları topla", "Matematik", ["Matematik"], "BLOCK_KOMUT_TOPLA", False, False, "ACCEPT"),
        ("CB_NEG_004", "Ağaçtaki elmaları toplama", "Matematik", ["Matematik"], "BLOCK_KOMUT_TOPLAMA", True, False, "ACCEPT"),
        ("CB_NEG_005", "Radyonun sesini yükselt", "Elektronik", ["Elektronik"], "BLOCK_KOMUT_YUKSELT", False, False, "ACCEPT"),
        ("CB_NEG_006", "Radyonun sesini yükseltme", "Elektronik", ["Elektronik"], "BLOCK_KOMUT_YUKSELTME", True, False, "ACCEPT"),
    ]

    for cid, query, cat, cats, bid, neg, ood, dec in unseen_negated_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_categories=cats,
            expected_block_id=bid, is_negated=neg, is_ood=ood, expected_decision=dec
        ))

    # 4. True Out-Of-Distribution (OOD) Queries (expecting BELİRSİZ_VEYA_BİLİNMİYOR)
    ood_samples = [
        ("CB_OOD_001", "qwertyuiop asdfghjkl zxcvbnm 123890", None, None, None, False, True, "BELİRSİZ_VEYA_BİLİNMİYOR"),
        ("CB_OOD_002", "Zırvalamak ve kuantum spatulası ile dondurma satmak", None, None, None, False, True, "BELİRSİZ_VEYA_BİLİNMİYOR"),
        ("CB_OOD_003", "Xylophone submarine quantum banana pancake", None, None, None, False, True, "BELİRSİZ_VEYA_BİLİNMİYOR"),
    ]

    for cid, query, cat, cats, bid, neg, ood, dec in ood_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_categories=cats,
            expected_block_id=bid, is_negated=neg, is_ood=ood, expected_decision=dec
        ))

    # 5. Overlapping / Ambiguous Queries (expecting ÇELİŞKİLİ_KANIT)
    ambiguous_samples = [
        ("CB_AMB_001", "Matematiksel fizikte ivme ve 12 çarpı 8 hesabının biyolojik hücreye etkisi", None, ["Fizik", "Matematik", "Biyoloji"], None, False, False, "ÇELİŞKİLİ_KANIT"),
        ("CB_AMB_002", "Elektronik devrede gezegenlerin orbit frekansı tarihi", None, ["Elektronik", "Gezegenler", "Tarih"], None, False, False, "ÇELİŞKİLİ_KANIT"),
    ]

    for cid, query, cat, cats, bid, neg, ood, dec in ambiguous_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_categories=cats,
            expected_block_id=bid, is_negated=neg, is_ood=ood, expected_decision=dec
        ))

    # 6. Multi-Hop Transitive Deduction Queries
    multihop_samples = [
        ("CB_HOP_2", "Aslan canlı mıdır?", "Hayvanlar", ["Hayvanlar"], "BLOCK_CANLI", False, False, "ACCEPT", ["Aslan", "Memeli", "Hayvan"]),
        ("CB_HOP_3", "Aslan hücresel organizma mıdır?", "Hayvanlar", ["Hayvanlar"], "BLOCK_ORGANIZMA", False, False, "ACCEPT", ["Aslan", "Memeli", "Hayvan", "Canlı"]),
        ("CB_HOP_4", "Güneş enerjisi kütleye dönüşür mü?", "Biyoloji", ["Biyoloji"], "BLOCK_KUTLE", False, False, "ACCEPT", ["Güneş", "Işık", "Fotosentez", "Biyokütle", "Kütle"]),
        ("CB_HOP_5", "Quark galaksi kümesi oluşturur mu?", "Fizik", ["Fizik"], "BLOCK_GALAKSI", False, False, "ACCEPT", ["Quark", "Proton", "Atom", "Yıldız", "Galaksi", "Evren"]),
    ]

    for cid, query, cat, cats, bid, neg, ood, dec, chain in multihop_samples:
        cases.append(ChainBlockTestCase(
            id=cid, query=query, expected_category=cat, expected_categories=cats,
            expected_block_id=bid, is_negated=neg, is_ood=ood, expected_decision=dec, relation_chain=chain
        ))

    return cases
