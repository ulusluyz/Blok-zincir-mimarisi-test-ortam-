import pytest
from src.bitwise import BinaryVector
from src.blocks import AddressBlock, KnowledgeBlock
from src.hierarchical_blocks import GroupRepresentative, HierarchicalBlockNetwork


def test_probe_recovers_group_with_misleading_g_and_preserves_a_plus_isolation():
    q = BinaryVector(0b11110000, bit_length=64)
    wrong = GroupRepresentative('wrong', 'wrong', q)
    correct = GroupRepresentative('correct', 'correct', BinaryVector(0b00001111, bit_length=64))
    wrong_address = AddressBlock('wrong_ap', payload={'secret': 'wrong'})
    correct_address = AddressBlock('correct_ap', payload={'secret': 'correct'})
    wrong.add_member_a_block(KnowledgeBlock('wrong_a', 'wrong', BinaryVector(0b00001111, bit_length=64), wrong_address))
    correct.add_member_a_block(KnowledgeBlock('correct_a', 'correct', q, correct_address))
    net = HierarchicalBlockNetwork([wrong, correct])
    net.register_flat_blocks(wrong.member_a_blocks + correct.member_a_blocks)
    baseline = net.process_query_hierarchical_2level(q, group_routing_strategy='top_k', group_top_k=1)
    assert baseline['active_groups'][0][1] == 'wrong'
    result = net.process_query_hierarchical_2level(q, group_routing_strategy='evidence_probe', probe_group_count=2, probe_weight=0.8, max_active_groups=1)
    assert result['active_groups'][0][1] == 'correct'
    assert correct_address.read_count == 1
    assert wrong_address.read_count == 1  # Only from baseline; probing must not read A+.
    assert result['total_a_plus_read'] == 1


@pytest.mark.parametrize('kwargs', [{'probe_group_count': 0}, {'probe_weight': -0.1}, {'probe_weight': 1.1}, {'max_active_groups': 0}])
def test_probe_rejects_invalid_parameters(kwargs):
    net = HierarchicalBlockNetwork()
    with pytest.raises(ValueError):
        net.process_query_hierarchical_2level(BinaryVector(0, bit_length=64), group_routing_strategy='evidence_probe', **kwargs)
