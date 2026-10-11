"""Reproducible held-out multi-topic comparison; no tuning on test labels."""
import json, statistics, time
from pathlib import Path
from src.dataset import CATEGORIES, get_full_split_datasets
from src.features import FeatureExtractor
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock
from src.learnable_g import StaticGRepresentative
from src.hierarchical_blocks import HierarchicalBlockNetwork

train, val, single, multi = get_full_split_datasets()
fe = FeatureExtractor(max_features=256, sublinear_tf=True).fit([t for t,_ in train])
enc = TrainableBinaryEncoder(input_dim=256, bit_length=128, num_categories=len(CATEGORIES), seed=42)
enc.fit([fe.transform_dense(t) for t,_ in train], [CATEGORIES.index(c) for _,c in train], epochs=15, lr=0.05, contrastive_weight=0.6)
groups=[]
for cat in CATEGORIES:
    samples=[t for t,c in train if c==cat][:3]
    if not samples: continue
    g=StaticGRepresentative('g_'+cat,cat,enc.encode(fe.transform_dense(samples[0])))
    for i,t in enumerate(samples):
        b=KnowledgeBlock('kb_'+cat+'_'+str(i),cat,enc.encode(fe.transform_dense(t)),AddressBlock('ap_'+cat+'_'+str(i),payload={'category':cat}))
        g.add_member_a_block(b)
    groups.append(g)
net=HierarchicalBlockNetwork(groups)
net.register_flat_blocks([b for g in groups for b in g.member_a_blocks])
queries=[(enc.encode(fe.transform_dense(item['text'])),set(item['expected_categories'])) for item in multi]
settings=[('dynamic_multi_label',{}),('probe_3',{'group_routing_strategy':'evidence_probe','probe_group_count':3,'probe_weight':0.8}),('probe_5',{'group_routing_strategy':'evidence_probe','probe_group_count':5,'probe_weight':0.8}),('probe_10',{'group_routing_strategy':'evidence_probe','probe_group_count':10,'probe_weight':0.8})]
results=[]
for name,kw in settings:
    found=expected=joint=selected=0; elapsed=[]; a_evaluated=[]; probe_ops=[]; selected_groups=[]
    for Q,targets in queries:
        for rep in range(3):
            args={'group_routing_strategy':'dynamic_multi_label','max_active_groups':3,'block_top_k':1}
            args.update(kw)
            t=time.perf_counter(); r=net.process_query_hierarchical_2level(Q,**args); elapsed.append((time.perf_counter()-t)*1000)
            if rep: continue
            chosen={x[1] for x in r['active_groups']}; hit=len(chosen&targets)
            found+=hit;expected+=len(targets);joint+=int(targets<=chosen);selected+=len(chosen)
            a_evaluated.append(r['total_a_evaluated']);probe_ops.append(r['probe_a_evaluations']);selected_groups.append(sorted(chosen))
    results.append({'strategy':name,'multi_topic_group_recall':round(found/expected,4),'joint_selection_recall':round(joint/len(queries),4),'median_latency_ms':round(statistics.median(elapsed),4),'mean_a_evaluated':round(statistics.mean(a_evaluated),2),'mean_probe_a_evaluations':round(statistics.mean(probe_ops),2),'mean_total_a_score_operations':round(statistics.mean([a+b for a,b in zip(a_evaluated,probe_ops)]),2),'mean_groups_selected':round(selected/len(queries),2),'found':found,'expected':expected,'queries':len(queries),'selected_groups':selected_groups})
report={'dataset':'src.dataset.get_full_split_datasets UNSEEN_TEST_MULTI_DATASET','encoder':'128-bit seed=42 epochs=15; static G first train sample; 3 A blocks/category','warning':'Experimental static-G configuration, not a reproduction of the reported 51.7% trained-G result. No statistical significance implied. Timing includes full routing and A+ read.','results':results}
Path('PHASE1_V18_PROBE_COMPARISON.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for row in results: print({k:v for k,v in row.items() if k!='selected_groups'})
