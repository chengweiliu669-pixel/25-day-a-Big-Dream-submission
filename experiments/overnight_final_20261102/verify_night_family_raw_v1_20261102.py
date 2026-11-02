"""Rebuild causal raw inputs for every fixed candidate; never select on February."""
import copy,json,sys,time,hashlib,os
from pathlib import Path
import numpy as np
import lightgbm as lgb
import train_ongoing_horizon as t
B=t.BASE;F=B/'outputs/overnight_horizon_family_v3_20261102';O=B/'outputs/night_family_raw_v1_20261102';START=time.time()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)
def save(n,x):t.write(O/n,x)
def main():
    O.mkdir(exist_ok=False);protocol=read(F/'protocol.json');selection=read(F/'January_selection.json');expected=read(F/'February_results.json');candidates=protocol['candidates'];frozen=read(F/'frozen_before_February.json')
    for p,h in frozen.items():assert sha(B/p)==h,p
    save('protocol.json',dict(all_candidates=[c['id'] for c in candidates],recommended_model=protocol['recommended_model_by_October_only'],raw_populations=['new_phase_population','old410'],no_model_or_threshold_selection=True,default_unchanged=True,Kaggle_upload=False))
    # Load every new model before installing the unchanged parent archive guard.
    runtimes={}
    for c in candidates:
        root=B/c['source'];cfg=c['config'];seeds=read(root/'protocol.json')['seeds'];models={}
        for panel in t.PANELS:
            models[panel]=[(lgb.Booster(model_file=str(root/'base_models'/f'final_{seed}'/f'{panel}.txt')),[lgb.Booster(model_file=str(root/f"final_{cfg['capacity']}_{seed}"/f'{panel}-h{h}.txt')) for h in range(5,31,5)]) for seed in seeds]
        runtimes[c['id']]=models
    task=B/'work/repo/experiments/task2';sys.path.insert(0,str(task/'best_overall_082639_jaccard_20261031/code'))
    from hybrid_ongoing import Hybrid,features,spatial_features,odds_mix
    shared=Hybrid(task/'best_overall_082639_jaccard_20261031');sys.path.insert(0,str(task/'r127_reproduction_20261027'));from repro_common import setup
    parent,sample,_,audit=setup(B/'work/r127/ConditionalPosition-R127',B/'work/local_transfer/data/kaggle_public',O/'raw-cache',True,extra_allowed=(B/'outputs',task));shared.attach_parent(parent)
    import ramp_features as rf
    def construct(s):
        panel=s['panel'];basis=shared.onset.bases[panel]['position'];x,n,_=features(s,rf.ramp_history(s),rf.topology(panel)['counts'],basis,shared.context[panel]);extra,en=spatial_features(s,basis);assert n==shared.names['base'] and en==shared.names['extra'];return np.column_stack([x,extra]).astype('float32'),np.asarray(parent.probability(s).T,dtype='float32')
    results={};checks=[]
    for pop in ['new_phase_population','old410']:
        cached,meta=t.speed.query(pop);raw={k:np.empty_like(cached[k]) for k in ['X','parent','y','m']};labels=np.empty(len(cached['X']),dtype='U16');entries=read(task/'r127_reproduction_20261027/indices'/f'february-{pop}.json')['samples'];lookup={(e['panel'],t.old.prior.old.normtime(e['origin'])):e for e in entries if e['condition']==1 and e['has_future_queue']};seen=set()
        for i,w in enumerate(meta['windows']):
            e=lookup[(w['panel'],t.old.prior.old.normtime(w['origin']))];s=sample(e);x,p=construct(s);a,b=w['start'],w['end'];assert len(x)==b-a;raw['X'][a:b]=x;raw['parent'][a:b]=p;raw['y'][a:b]=np.asarray(s['y']).T;raw['m'][a:b]=np.asarray(s['m']).T;labels[a:b]=w['panel']
            if w['panel'] not in seen:
                other=copy.deepcopy(s);other['y']=1-s['y'];other['m']=~s['m'];other['speed_y']=np.full_like(s['speed_y'],999);other['has_future_queue']=not s['has_future_queue'];xx,pp=construct(other);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(p,pp);seen.add(w['panel']);checks.append(dict(population=pop,panel=w['panel'],future_label_mutation_no_effect=True))
            if (i+1)%32==0:log(stage='RAW',population=pop,windows=i+1)
        for k in raw:np.testing.assert_array_equal(raw[k],cached[k],err_msg=pop+'/'+k)
        log(stage='CACHE_PARITY',population=pop,windows=len(meta['windows']),rows=len(labels))
        rows={};r127=t.old.score(raw['parent'],raw,meta)
        for c in candidates:
            cfg=c['config'];acc=np.empty_like(raw['parent'])
            for panel in np.unique(labels):
                ids=np.flatnonzero(labels==panel);zsum=np.zeros((len(ids),6));members=runtimes[c['id']][panel]
                for base,heads in members:
                    for h,head in enumerate(heads):
                        xx=np.column_stack([raw['X'][ids],np.full(len(ids),(h+1)*5)]);z=base.predict(xx,raw_score=True,num_threads=4)+cfg['alpha']*head.predict(xx,raw_score=True,num_threads=4);zsum[:,h]+=np.exp(-np.logaddexp(0.,-z))/len(members)
                acc[ids]=zsum
            p=odds_mix(raw['parent'],acc);cachedp=np.load(B/c['source']/f"{pop}_{cfg['name']}_probability.npy");error=float(np.max(np.abs(p-cachedp)));assert error<=1e-7,(c['id'],error)
            threshold=np.asarray(selection[c['id']]['profile']['thresholds']);score=t.old.score(p,raw,meta,threshold);gold=expected[pop]['candidates'][c['id']];assert abs(score['onset_IoU']-gold['score']['onset_IoU'])<1e-12;assert score['cell_counts']==gold['score']['cell_counts']
            rows[c['id']]=dict(max_probability_error=error,score=score,overall_IoU=(score['onset_IoU']+gold['fixed_onset_IoU'])/2,IoU_vs_R127=t.old.prior.REF.bootstrap(score,r127),IoU_vs_retained_smooth=gold['IoU_vs_retained_smooth']);log(stage='VERIFIED',population=pop,id=c['id'],overall=rows[c['id']]['overall_IoU'])
        results[pop]=dict(R127_ongoing=r127,candidates=rows,raw_windows=len(meta['windows']));save('results.json',results)
    for p,h in frozen.items():assert sha(B/p)==h,p
    t.check();save('verification.json',dict(passed=True,all_candidates_recomputed=True,raw_X_parent_y_m_exact_cache_parity=True,all_February_IoU_and_FP_FN_reproduced=True,causality_checks=checks,query_data_rebuilt_and_hash_verified=len(audit['generated_day_keys']),onset_hashes_unchanged=True,no_default_updated=True,no_model_selection_on_February=True,official_S_queue=None,Kaggle_upload=False));log(stage='COMPLETE')
if __name__=='__main__':main()
