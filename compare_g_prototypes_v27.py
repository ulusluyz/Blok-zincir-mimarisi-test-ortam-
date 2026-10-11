"""Training-only G prototype alternatives, evaluated on untouched multi-topic queries."""
import json, statistics
from pathlib import Path
from src.dataset import CATEGORIES,get_full_split_datasets
from src.features import FeatureExtractor
from src.encoder import TrainableBinaryEncoder
from src.bitwise import similarity_xor_popcount, BinaryVector

train,val,single,multi=get_full_split_datasets()
fe=FeatureExtractor(max_features=256,sublinear_tf=True).fit([t for t,_ in train])
X=[fe.transform_dense(t) for t,_ in train]
records=[]
for seed in (11,23,42,67,101):
    enc=TrainableBinaryEncoder(input_dim=256,bit_length=128,num_categories=len(CATEGORIES),seed=seed)
    enc.fit(X,[CATEGORIES.index(c) for _,c in train],epochs=15,lr=0.05,contrastive_weight=0.6)
    enc_train=[enc.encode(x) for x in X]
    prototypes={}
    for cat in CATEGORIES:
        samples=[bits for bits,(_,c) in zip(enc_train,train) if c==cat]
        first=samples[0]
        medoid=max(enumerate(samples),key=lambda kv:(sum(similarity_xor_popcount(kv[1],other) for other in samples),-kv[0]))[1]
        # majority prototype constructed solely from training examples
        majority=[sum(bits.to_bool_list()[j] for bits in samples)*2>=len(samples) for j in range(128)]
        centroid=BinaryVector.from_bool_list(majority)
        prototypes[cat]={'first':first,'medoid':medoid,'majority':centroid,'max_member':samples}
    for qi,item in enumerate(multi):
        q=enc.encode(fe.transform_dense(item['text']))
        target=set(item['expected_categories'])
        for method in ('first','medoid','majority','max_member'):
            scores=[]
            for cat in CATEGORIES:
                p=prototypes[cat][method]
                score=max(similarity_xor_popcount(q,x) for x in p) if isinstance(p,list) else similarity_xor_popcount(q,p)
                scores.append((cat,score))
            ranked=sorted(scores,key=lambda pair:(-pair[1],CATEGORIES.index(pair[0])))
            chosen={cat for cat,_ in ranked[:3]}
            records.append({'seed':seed,'query':qi,'method':method,'hits':len(chosen&target),'expected':len(target),'joint':int(target<=chosen),'g_similarity_operations':sum(len(prototypes[cat]['max_member']) for cat in CATEGORIES) if method=='max_member' else len(CATEGORIES)})
    print('seed',seed,flush=True)
summary={}
for method in ('first','medoid','majority','max_member'):
    rows=[r for r in records if r['method']==method]
    summary[method]={'recall':round(sum(r['hits'] for r in rows)/sum(r['expected'] for r in rows),4),'joint_rate':round(statistics.mean(r['joint'] for r in rows),4),'mean_g_similarity_operations':statistics.mean(r['g_similarity_operations'] for r in rows)}
report={'notes':'Training-only prototypes. 5 encoder initializations x 10 fixed held-out queries, not 50 independent queries. Top-3 groups; no A evidence operations. No hyperparameter tuning.', 'summary':summary,'records':records}
Path('PHASE1_V27_G_PROTOTYPES.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(summary,indent=2))
