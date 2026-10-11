"""Query-cluster paired bootstrap: encoder seeds are repeated measurements, not new queries."""
import json
import random
from collections import defaultdict
from pathlib import Path


def analyze(records, *, baseline='first', candidate='max_member', draws=10000, seed=2801):
    by_query = defaultdict(dict)
    for row in records:
        key = (row['query'], row['seed'])
        method = row['method']
        if method in (baseline, candidate):
            if method in by_query[key]:
                raise ValueError('duplicate seed/query/method')
            by_query[key][method] = row
    if not by_query or any(set(v) != {baseline, candidate} for v in by_query.values()):
        raise ValueError('unpaired or empty experiment')
    clusters = defaultdict(list)
    for (query, _seed), pair in by_query.items():
        b, c = pair[baseline], pair[candidate]
        if b['expected'] != c['expected'] or b['expected'] <= 0:
            raise ValueError('invalid target counts')
        clusters[query].append((c['hits']-b['hits'], b['expected'], c['joint']-b['joint']))
    keys = sorted(clusters)
    def score(sample):
        values = [v for q in sample for v in clusters[q]]
        return (sum(x[0] for x in values)/sum(x[1] for x in values), sum(x[2] for x in values)/len(values))
    rng = random.Random(seed)
    bootstrap = [score([keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(draws)]
    def interval(index):
        values = sorted(row[index] for row in bootstrap)
        return [round(values[int(.025*(draws-1))], 6),round(values[int(.975*(draws-1))], 6)]
    wins = sum(sum(x[0] for x in clusters[q]) > 0 for q in keys)
    losses = sum(sum(x[0] for x in clusters[q]) < 0 for q in keys)
    return {'independent_query_count':len(keys),'paired_seed_query_count':len(by_query),
            'delta_recall':round(score(keys)[0],6),'delta_joint':round(score(keys)[1],6),
            'query_cluster_bootstrap_95ci_recall':interval(0),
            'query_cluster_bootstrap_95ci_joint':interval(1),
            'query_wins':wins,'query_losses':losses,'query_ties':len(keys)-wins-losses,
            'bootstrap_draws':draws,'bootstrap_seed':seed,
            'caveat':'Exploratory 10-query sample; seeds do not increase independent query count. Candidate selected after observing this test set: confidence intervals are descriptive, not confirmatory.'}


if __name__ == '__main__':
    data=json.loads(Path('PHASE1_V27_G_PROTOTYPES.json').read_text())
    output={m:analyze(data['records'],candidate=m) for m in ('medoid','majority','max_member')}
    Path('PHASE1_V28_G_UNCERTAINTY.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))
