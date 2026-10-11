import pytest
from audit_g_uncertainty_v28 import analyze


def sample():
    return [dict(query=q,seed=s,method=m,hits=h,expected=2,joint=int(h==2))
            for q in range(3) for s in (1,2) for m,h in (('first',1),('max_member',2 if q==0 else 1))]


def test_query_not_seed_independent():
    r=analyze(sample(),draws=100,seed=4)
    assert r['independent_query_count']==3
    assert r['paired_seed_query_count']==6
    assert r['query_wins']==1 and r['query_ties']==2


def test_reproducible_bootstrap():
    assert analyze(sample(),draws=100,seed=4)==analyze(sample(),draws=100,seed=4)


def test_unpaired_rejected():
    with pytest.raises(ValueError,match='unpaired'):
        analyze(sample()[:-1],draws=20)


def test_duplicate_rejected():
    rows=sample()
    with pytest.raises(ValueError,match='duplicate'):
        analyze(rows+[rows[0]],draws=20)
