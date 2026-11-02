"""Save candidate, independently reconstruct masked inputs, evaluate full55 days.
Never tunes parameters or thresholds and never submits to Kaggle.
"""
import os,sys,json,hashlib,shutil,time,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
B=Path(__file__).resolve().parents[1];M=B/'work/repo/experiments/task1/best_road_combination_20261026';SOURCE=B/'outputs/state_quality_spatial_v3_20261102';OUT=B/'outputs/state_quality_spatial_v3_full55_20261102';BUNDLE=B/'outputs/retained_state_quality_spatial_v3_20261102';DATA=B/'work/local_transfer/data/kaggle_public';OUT.mkdir(exist_ok=True);BUNDLE.mkdir(exist_ok=True);START=time.time()
sys.path.insert(0,str(M));import predict_selected as parent;import feature_runtime as base
import state_quality_spatial_v3_runtime as runtime
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)
def aggregate(rows,model):
    roads=[]
    for panel in sorted({r['panel'] for r in rows}):
        rr=[r for r in rows if r['panel']==panel];regs=[]
        for regime in ['R1','R2','R3']:
            selected=[r for r in rr if r['regime']==regime];n=sum(r['rows'] for r in selected)
            if n:
                sv=np.sqrt(sum(r[model]['speed_sse'] for r in selected)/n);sq=np.sqrt(sum(r[model]['flow_sse'] for r in selected)/n);regs.append(dict(regime=regime,rows=n,speed_RMSE=float(sv),flow_per_lane_RMSE=float(sq),S_state=float(.54*max(0,1-sv/25)+.46*max(0,1-sq/600))))
        pairs=sum(r[model]['pairs'] for r in rr);roads.append(dict(panel=panel,S_state=float(np.mean([r['S_state'] for r in regs])),dN_MAE=sum(r[model]['dN_abs_error_sum'] for r in rr)/pairs,rows=sum(r['rows'] for r in rr),regimes=regs))
    return dict(S_state=float(np.mean([r['S_state'] for r in roads])),dN_MAE=float(np.mean([r['dN_MAE'] for r in roads])),rows=sum(r['rows'] for r in roads),roads=roads)
def errors(e,a,physical,truth):
    ii,jj=a['ii'],a['jj'];tv=e.grid(truth,'speed_kmh');tq=e.grid(truth,'flow_vph')/e.lanes[:,None];y=np.column_stack([tv[ii,jj],tq[ii,jj]]);assert np.isfinite(y).all();ev=physical[:,0]-y[:,0];eq=physical[:,1]-y[:,1]
    v=e.grid(a['obs'],'speed_kmh');q=e.grid(a['obs'],'flow_vph')/e.lanes[:,None];v[ii,jj]=physical[:,0];q[ii,jj]=physical[:,1];length=(e.static.length_km.to_numpy()*e.lanes)[:,None];N=q/np.maximum(v,1)*length;TN=tq/np.maximum(tv,1)*length;valid=np.isfinite(tv)&np.isfinite(tq)&(e.grid(truth,'is_score_eligible')>.5);pair=valid[:,:-1]&valid[:,1:]&a['usable'][:,:-1]&a['usable'][:,1:]&(a['mask'][:,:-1]|a['mask'][:,1:]);d=(np.diff(N,axis=1)-np.diff(TN,axis=1))[pair];return dict(speed_sse=float(np.square(ev).sum()),flow_sse=float(np.square(eq).sum()),pairs=int(pair.sum()),dN_abs_error_sum=float(np.abs(d).sum()))
def main():
    assert read(SOURCE/'verification.json')['passed'];selection=read(SOURCE/'selection.json');name=selection['selected'];assert name!='parent'
    for p,h in read(SOURCE/'frozen_before_January_February.json').items():assert sha(Path(p))==h,p
    checkpoint=SOURCE/'models'/(name+'.pt')
    for file in [checkpoint,SOURCE/'protocol.json',SOURCE/'selection.json',SOURCE/'decision_before_training.json',SOURCE/'January_results.json',SOURCE/'February_results.json',SOURCE/'verification.json',B/'outputs/state_quality_spatial_v3_runtime.py',B/'outputs/train_state_quality_spatial_v3_20261102.py']:
        dst=BUNDLE/file.name
        if dst.exists():assert sha(dst)==sha(file)
        else:shutil.copy2(file,dst)
    save(BUNDLE/'parent_release.json',dict(url='https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/download/best-road-combination-20261026/TrafficFlowBench-Best-20261026.zip',sha256='efbc9223b2d45c131840e6c64f460e90b92a6a70700e0bc6a36d5480e0f2a847',state_models_unchanged=True,Task2_unchanged=True,official_total=None,full55_validation_pending=True))
    save(BUNDLE/'manifest.json',{p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file() and p.name!='manifest.json'})
    model=runtime.load_model(BUNDLE/checkpoint.name);routes=parent.load_routes();plan=read(M/'reproduction/vendor/jayden2/reference-split.json');days=plan['holdout_dates'];assert len(days)==55;rows=[];parity=[];recomputed=[];known_checks=[]
    for panel,route in routes.items():
        e=base.Experiment(SimpleNamespace(release_root=DATA,panel=panel,output=OUT/'metadata'/panel,seed=20261102));h=np.load(route['paths'].get('history',route.get('history')));e.history=h['mean'] if isinstance(h,np.lib.npyio.NpzFile) else h;truth_files=e.truth_files.copy();e.truth_files={};pieces=[]
        for chunk in pd.read_csv(DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str,chunksize=250000):pieces.append(chunk[chunk.timestamp.str[:10].isin(days)])
        keys=pd.concat(pieces,ignore_index=True);daily={d:k.reset_index(drop=True) for d,k in keys.groupby(keys.timestamp.str[:10])};parent.keys_for=lambda data,road,split,date:daily[date].copy()
        for date in days:
            cache=B/'outputs/state_density_pilot_20261102/parent_predictions'/panel/(panel+'_'+date+'.parquet');local=OUT/'parent_predictions'/panel/(panel+'_'+date+'.parquet');record=OUT/'daily'/panel/(date+'.json')
            if record.exists():rows.append(read(record));continue
            original=pd.read_parquet
            def guarded(path,*a,**kw):
                if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Raw target read attempted during independent inference')
                return original(path,*a,**kw)
            pd.read_parquet=guarded
            try:
                if date==days[0] or (not cache.exists() and not local.exists()):
                    p=parent.predict(panel,date,route,DATA,local.parent,'cuda')
                    if cache.exists():
                        old=pd.read_parquet(cache);err=float(np.max(np.abs(p[['speed_kmh','flow_vph']].to_numpy()-old[['speed_kmh','flow_vph']].to_numpy())));assert err<1e-6;recomputed.append(dict(panel=panel,date=date,max_parent_error=err))
                else:p=pd.read_parquet(cache if cache.exists() else local)
                k=daily[date];assert p[base.KEYS].reset_index(drop=True).equals(k[base.KEYS].reset_index(drop=True));a=runtime.inputs(e,date,k,p);out=runtime.predict(model,a);ii,jj=a['ii'],a['jj'];pred=np.column_stack([out[:,0][ii,jj]*25,out[:,1][ii,jj]*600]);assert np.isfinite(pred).all() and (pred>=0).all()
                cached=SOURCE/'data'/panel/(date+'.npz')
                if not cached.exists():cached=B/'outputs/state_quality_spatial_overnight_20261102/data'/panel/(date+'.npz')
                if cached.exists():
                    with np.load(cached) as z:
                        np.testing.assert_array_equal(a['x'],z['x']);np.testing.assert_array_equal(a['p'],z['p']);np.testing.assert_array_equal(a['mask'],z['mask']);parity.append(dict(panel=panel,date=date,features_and_parent_and_mask_exact=True))
                known_checks.append(dict(panel=panel,date=date,known_cells_unchanged=True,unmasked_label_reads_blocked=True))
            finally:pd.read_parquet=original
            truth=pd.read_parquet(truth_files[date]);baseline=p[['speed_kmh','flow_vph']].to_numpy();baseline[:,1]/=e.lanes[ii];r=dict(panel=panel,date=date,regime=k.mask_regime.iloc[0],rows=len(k),baseline=errors(e,a,baseline,truth),candidate=errors(e,a,pred,truth));rows.append(r);save(record,r);log(stage='FULL55',panel=panel,date=date,rows=len(k))
        save(OUT/'running_metrics.json',dict(baseline=aggregate(rows,'baseline'),candidate=aggregate(rows,'candidate'),complete_roads=len({r['panel'] for r in rows}),expected_roads=10));gc.collect()
    assert len(rows)==550 and sum(r['rows'] for r in rows)==6636705;baseline=aggregate(rows,'baseline');candidate=aggregate(rows,'candidate');historical=.9428286319696028;assert abs(baseline['S_state']-historical)<1e-7,(baseline['S_state'],historical)
    rng=np.random.default_rng(20261102);deltas=[]
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True);rr=[r for d in draw for r in rows if r['date']==d];deltas.append(aggregate(rr,'candidate')['S_state']-aggregate(rr,'baseline')['S_state'])
    ci=np.quantile(deltas,[.025,.975]).tolist();result=dict(passed=True,rows=6636705,days=55,roads=10,baseline=baseline,candidate=candidate,delta_S_state=candidate['S_state']-baseline['S_state'],dN_relative_change=candidate['dN_MAE']/baseline['dN_MAE']-1,day_cluster_95CI=ci,resamples=2000,official_total=None,official_S_physics=None,caveat='Full55 previously used development holdout;proxy accumulation error against released observations,not official LWR. Same unchangedTask2;no Kaggle upload.',checkpoint_sha256=sha(checkpoint));save(OUT/'results.json',result)
    verification=dict(passed=True,independent_runtime=True,features_parent_mask_exact_cache_checks=parity,raw_parent_recomputations=recomputed,guarded_inference_checks=len(known_checks),all_known_cells_unchanged=True,no_label_arguments_to_runtime=True,no_Kaggle_upload=True)
    save(OUT/'verification.json',verification)
    for file in [OUT/'results.json',OUT/'verification.json']:shutil.copy2(file,BUNDLE/('full55_'+file.name))
    for p,h in read(SOURCE/'frozen_before_January_February.json').items():assert sha(Path(p))==h
    save(BUNDLE/'manifest.json',{p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file() and p.name!='manifest.json'});log(stage='COMPLETE',delta=result['delta_S_state'],CI=ci,candidate=candidate['S_state'])
if __name__=='__main__':main()
