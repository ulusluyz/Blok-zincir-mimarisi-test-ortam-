from src.bitwise import BinaryVector
from src.blocks import AddressBlock, KnowledgeBlock
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork


def test_probe_scores_are_reused_and_counted_once():
    q = BinaryVector(7, bit_length=64)
    groups = []
    for i in range(3):
        g = GroupRepresentative(str(i), str(i), q)
        for j in range(2):
            g.add_member_a_block(KnowledgeBlock(f'a{i}_{j}', str(i), q, AddressBlock(f'ap{i}_{j}', payload={})))
        groups.append(g)
    net = HierarchicalBlockNetwork(groups)
    net.register_flat_blocks([b for g in groups for b in g.member_a_blocks])
    r = net.process_query_hierarchical_2level(q, group_routing_strategy='evidence_probe', probe_group_count=3, max_active_groups=2)
    assert r['probe_a_evaluations'] == 6
    assert r['total_a_score_operations'] == 6
    assert r['total_a_evaluated'] == 4


def test_shared_a_block_scored_once_across_probe_groups():
    q = BinaryVector(7, bit_length=64)
    shared = KnowledgeBlock('shared', 'shared', q, AddressBlock('ap_shared', payload={}))
    groups = []
    for i in range(2):
        g = GroupRepresentative(str(i), str(i), q)
        g.add_member_a_block(shared)
        groups.append(g)
    net = HierarchicalBlockNetwork(groups)
    net.register_flat_blocks([shared])
    r = net.process_query_hierarchical_2level(q, group_routing_strategy='evidence_probe', probe_group_count=2, max_active_groups=2)
    assert r['probe_a_evaluations'] == 1
    assert r['total_a_score_operations'] == 1
