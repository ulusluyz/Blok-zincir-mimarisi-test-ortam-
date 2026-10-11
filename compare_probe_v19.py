"""Repeated-seed, paired evaluation. Seeds change encoder initialization, not test labels."""
import json, statistics, time
from pathlib import Path
from src.dataset import CATEGORIES, get_full_split_datasets
from src.features import FeatureExtractor
from src.encoder import TrainableBinaryEncoder
from src.blocks import AddressBlock, KnowledgeBlock
from src.learnable_g import StaticGRepresentative
from src.hierarchical_blocks import HierarchicalBlockNetwork

train, val, single, multi = get_full_split_datasets()
fe=FeatureExtractor(max_features=256,sublinear_tf=True).fit([t for t,_ in train])
settings={'baseline':{},'fixed_top3':{'group_routing_strategy':'top_k','group_top_k':3},'probe_3':{'group_routing_strategy':'evidence_probe','probe_group_count':3,'probe_weight':0.8},'probe_5':{'group_routing_strategy':'evidence_probe','probe_group_count':5,'probe_weight':0.8},'probe_10':{'group_routing_strategy':'evidence_probe','probe_group_count':10,'probe_weight':0.8}}
records=[]
for seed in (11,23,42,67,101):
    enc=TrainableBinaryEncoder(input_dim=256,bit_length=128,num_categories=len(CATEGORIES),seed=seed)
    enc.fit([fe.transform_dense(t) for t,_ in train],[CATEGORIES.index(c) for _,c in train],epochs=15,lr=0.05,contrastive_weight=0.6)
    groups=[]
    for cat in CATEGORIES:
        samples=[t for t,c in train if c==cat][:3]
        if not samples: continue
        g=StaticGRepresentative('g_'+cat,cat,enc.encode(fe.transform_dense(samples[0])))
        for i,t in enumerate(samples):
            g.add_member_a_block(KnowledgeBlock('kb_'+cat+'_'+str(i),cat,enc.encode(fe.transform_dense(t)),AddressBlock('ap_'+cat+'_'+str(i),payload={'category':cat})))
        groups.append(g)
    net=HierarchicalBlockNetwork(groups)
    net.register_flat_blocks([b for g in groups for b in g.member_a_blocks])
    for qi,item in enumerate(multi):
        q=enc.encode(fe.transform_dense(item['text'])); target=set(item['expected_categories'])
        for name,kw in settings.items():
            args={'group_routing_strategy':'dynamic_multi_label','max_active_groups':3,'block_top_k':1};args.update(kw)
            t=time.perf_counter(); r=net.process_query_hierarchical_2level(q,**args); ms=(time.perf_counter()-t)*1000
            chosen={x[1] for x in r['active_groups']}
            records.append({'seed':seed,'query':qi,'method':name,'hits':len(chosen&target),'expected':len(target),'joint':int(target<=chosen),'operations':r['total_a_score_operations'],'latency_ms':ms,'groups':len(chosen)})
summary={}
for name in settings:
    rows=[r for r in records if r['method']==name]
    per_seed=[]
    for seed in (11,23,42,67,101):
        s=[r for r in rows if r['seed']==seed]
        per_seed.append(round(sum(r['hits'] for r in s)/sum(r['expected'] for r in s),4))
    summary[name]={'recall':round(sum(r['hits'] for r in rows)/sum(r['expected'] for r in rows),4),'per_seed_recall':per_seed,'joint_rate':round(statistics.mean(r['joint'] for r in rows),4),'mean_operations':round(statistics.mean(r['operations'] for r in rows),2),'median_latency_ms':round(statistics.median(r['latency_ms'] for r in rows),4),'mean_selected_groups':round(statistics.mean(r['groups'] for r in rows),2)}
report={'description':'Five random encoder initializations, 10 fixed held-out multi-topic queries each; paired comparisons; static G; no hyperparameter tuning on test data. Not 50 independent queries.', 'seeds':[11,23,42,67,101],'queries_per_seed':len(multi),'summary':summary,'records':records}
Path('PHASE1_V19_REPEATED_SEED_COMPARISON.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
