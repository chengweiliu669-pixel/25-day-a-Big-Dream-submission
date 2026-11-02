"""Chronological density-head pilot; frozen state models; February report-only.

No claimed official Task3 score: transition density errors use released noisy
state targets, not the organizer's withheld boundary flux.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,json,hashlib,time,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
B=Path(__file__).resolve().parents[1];MODEL=B/'work/repo/experiments/task1/best_road_combination_20261026';DATA=B/'work/local_transfer/data/kaggle_public';OUT=B/'outputs/state_joint_pilot_20261102';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(MODEL))
import predict_selected as parent
import feature_runtime as base
torch.set_num_threads(4)
PANELS=['D7_I10_E','D12_I405_S'];HEADS=['joint'];ALPHAS=[0.,.05,.1,.25,.5];START=time.time()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)
def choose_dates(dates,per_month):
    out=[]
    for month in sorted({d[:7] for d in dates}):
        items=[d for d in dates if d.startswith(month)];out += [items[i] for i in np.unique(np.linspace(0,len(items)-1,min(per_month,len(items))).round().astype(int))]
    return sorted(out)
def feature(e,date,keys):
    # All numeric input construction is guarded against unmasked state reads.
    original=pd.read_parquet
    def guarded(p,*a,**kw):
        if '/mainline_states/' in str(p).replace('\\','/'):raise RuntimeError('Unmasked target read during feature construction')
        return original(p,*a,**kw)
    pd.read_parquet=guarded
    try:values,bases,_,obs=e.feature_grid(date)
    finally:pd.read_parquet=original
    ts=pd.to_datetime(keys.timestamp,utc=True);ii=keys.station_id.map(e.si).to_numpy(int);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
    assert not np.isfinite(e.grid(obs,'speed_kmh')[ii,jj]).any() and not np.isfinite(e.grid(obs,'flow_vph')[ii,jj]).any()
    names=list(values);xx=np.column_stack([v[ii,jj] for v in values.values()]);extra=[];en=[]
    for name in names:
        if name.startswith('speed_kmh_') and 'flow_vph_'+name[len('speed_kmh_'):] in values:
            suffix=name[len('speed_kmh_'):]
            if suffix.endswith('_count') or 'gap_' in suffix or 'fallback' in suffix:continue
            v=values[name][ii,jj];q=values['flow_vph_'+suffix][ii,jj];extra.append(q/np.maximum(v,1.));en.append('density_'+suffix)
    x=np.column_stack([xx,*extra]).astype('float32');assert not np.isinf(x).any();initial=bases[1][ii,jj]/np.maximum(bases[0][ii,jj],1.)
    return x,initial,ii,jj,obs,names+en
def score(pred,truth,regimes):
    scores=[]
    for regime in ['R1','R2','R3']:
        take=regimes==regime
        if not take.any():continue
        rmse=np.sqrt(np.mean((pred[take]-truth[take])**2,axis=0));scores.append(dict(regime=regime,n=int(take.sum()),speed_RMSE=float(rmse[0]),flow_per_lane_RMSE=float(rmse[1]),S_state=float(.54*max(0,1-rmse[0]/25)+.46*max(0,1-rmse[1]/600))))
    return dict(S_state=float(np.mean([r['S_state'] for r in scores])),regimes=scores)
def transition(e,pred,truth,ii,jj,obs):
    # Insert target estimates into the observed frame. Never alter observations.
    length=e.static.length_km.to_numpy();v=e.grid(obs,'speed_kmh');q=e.grid(obs,'flow_vph');eligible=e.grid(obs,'is_score_eligible')>.5
    v[ii,jj]=pred[:,0];q[ii,jj]=pred[:,1]*e.lanes[ii];k=q/np.maximum(v,1);N=k*length[:,None]
    tv=e.grid(truth,'speed_kmh');tq=e.grid(truth,'flow_vph');TN=tq/np.maximum(tv,1)*length[:,None];target=np.zeros_like(v,dtype=bool);target[ii,jj]=True
    valid=eligible[:,:-1]&eligible[:,1:]&(target[:,:-1]|target[:,1:])&np.isfinite(N[:,:-1])&np.isfinite(N[:,1:])&np.isfinite(TN[:,:-1])&np.isfinite(TN[:,1:])
    error=(np.diff(N,axis=1)-np.diff(TN,axis=1))[valid]
    return dict(pairs=len(error),absolute_error_sum=float(np.abs(error).sum()),squared_error_sum=float(np.square(error).sum()))
def main():
    route=parent.load_routes();plan=read(MODEL/'reproduction/vendor/jayden2/reference-split.json');fit_dates=choose_dates([d for d in plan['training_dates'] if d<='2030-11-30'],6);dec=choose_dates([d for d in plan['internal_validation_dates'] if d<'2031-01-01'],6);jan=choose_dates([d for d in plan['holdout_dates'] if d.startswith('2031-01')],6);feb=choose_dates([d for d in plan['holdout_dates'] if d.startswith('2031-02')],6)
    protocol=dict(panels=PANELS,fit_dates=fit_dates,fit_rows_per_day=2500,heads=HEADS,alpha_grid=ALPHAS,December_model_selection_dates=dec,January_confirmation_dates=jan,February_report_only_dates=feb,selection='December S_state >=parent+.0001 and dN_MAE<=parent*1.001 and each road state>=parent-.001;alpha0 alwaysavailable. Maximize macro S_state,tie lowerdN_MAE. January confirms only;February report only.',target='Two L2 residual regressors for speed and perlane flow relative to masked interpolation,using51history features plus density-ratio context. Same500tree capacity. Blend both state channels with frozen parent using one shared alpha. Do not reconstruct speed by dividing predicted density.',truth='Released noisy high-quality states only, not hidden state or official Task3 boundary flows',inference='masked observations only; known state read is blocked while building inputs',Task2='All onset/ongoing models and thresholds unchanged',caveat='Pilot2roads sampled days, not full55day score, not official total; state models trained throughDec13,development already used historically')
    save(OUT/'protocol.json',protocol);frozen={str(MODEL/'selected-road-models.json'):sha(MODEL/'selected-road-models.json')}
    for panel in PANELS:
        e=base.Experiment(SimpleNamespace(release_root=DATA,panel=panel,output=OUT/'metadata'/panel,seed=20261102));r=route[panel];h=np.load(r['paths'].get('history',r.get('history')));e.history=h['mean'] if isinstance(h,np.lib.npyio.NpzFile) else h
        for p in r['paths'].values():frozen[p]=sha(Path(p))
        if r.get('history'):frozen[r['history']]=sha(Path(r['history']))
        keys=pd.read_csv(DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str);daily={d:g.reset_index(drop=True) for d,g in keys.groupby(keys.timestamp.str[:10])};rng=np.random.default_rng(20261102);xs=[];ys=[];weights=[]
        for date in fit_dates:
            k=daily[date];k=k.iloc[np.sort(rng.choice(len(k),min(len(k),2500),replace=False))].reset_index(drop=True);x,initial,ii,jj,_,names=feature(e,date,k);truth=pd.read_parquet(e.truth_files[date]);v=e.grid(truth,'speed_kmh')[ii,jj];q=e.grid(truth,'flow_vph')[ii,jj]/e.lanes[ii];y=np.column_stack([v-x[:,names.index('speed_kmh_initial')],q-x[:,names.index('flow_vph_initial')]]);assert np.isfinite(y).all();xs.append(x);ys.append(y);weights.append(np.full(len(y),1/len(y)));log(stage='DENSITY_FEATURES',panel=panel,date=date,rows=len(y))
        X=np.concatenate(xs);Y=np.concatenate(ys);W=np.concatenate(weights);W/=W.mean();dest=OUT/'models'/panel;dest.mkdir(parents=True,exist_ok=True);save(dest/'feature_names.json',names)
        for component in range(2):
            model=lgb.train(dict(objective='regression',learning_rate=.03,num_leaves=31,min_data_in_leaf=100,lambda_l2=50,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,seed=20261102,deterministic=True,force_col_wise=True,verbosity=-1,num_threads=8),lgb.Dataset(X,label=Y[:,component],weight=W),num_boost_round=500);model.save_model(str(dest/f'joint_{component}.txt'));log(stage='TRAINED',panel=panel,component=component,rows=len(Y),features=X.shape[1])
        del X,Y,xs,ys;gc.collect()
        # Both heads and alpha candidates are run only on December here.
        evaluate_panel(e,r,panel,daily,dec,'December',HEADS,ALPHAS)
    decres=aggregate('December',HEADS,ALPHAS);baseline=decres['joint_a0'];eligible=[(k,v) for k,v in decres.items() if k.endswith('_a0') or (v['S_state']>=baseline['S_state']+.0001 and v['dN_MAE']<=baseline['dN_MAE']*1.001 and all(a['S_state']>=b['S_state']-.001 for a,b in zip(v['roads'],baseline['roads'])))];name,r=min(eligible,key=lambda item:(-item[1]['S_state'],item[1]['dN_MAE']));head,alpha=name.split('_a');alpha=float(alpha);save(OUT/'selection.json',dict(name=name,head=head,alpha=alpha,December=r,baseline=baseline,eligible=[k for k,v in eligible],all=decres));log(stage='DECEMBER_SELECTED',name=name,**r)
    files=[Path(__file__),OUT/'protocol.json',OUT/'selection.json']+list((OUT/'models').rglob('*'));files=[p for p in files if p.is_file()];frozen.update({str(p):sha(p) for p in files});save(OUT/'frozen_before_January_February.json',frozen)
    for role,dates in [('January',jan),('February',feb)]:
        for panel in PANELS:
            e=base.Experiment(SimpleNamespace(release_root=DATA,panel=panel,output=OUT/'metadata'/panel,seed=20261102));r=route[panel];h=np.load(r['paths'].get('history',r.get('history')));e.history=h['mean'] if isinstance(h,np.lib.npyio.NpzFile) else h;keys=pd.read_csv(DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str);daily={d:g.reset_index(drop=True) for d,g in keys.groupby(keys.timestamp.str[:10])};evaluate_panel(e,r,panel,daily,dates,role,[head],sorted(set([0.,alpha])))
        res=aggregate(role,[head],sorted(set([0.,alpha])));save(OUT/f'{role}_results.json',res);log(stage=role.upper(),results=res)
    for p,h in frozen.items():assert sha(Path(p))==h,p
    save(OUT/'verification.json',dict(passed=True,all_parent_hashes_unchanged=True,model_and_selection_frozen_before_January_February=True,official_total=None,official_S_physics=None));log(stage='COMPLETE')

def evaluate_panel(e,r,panel,daily,dates,role,heads,alphas):
    rows=[];models={h:[lgb.Booster(model_file=str(OUT/'models'/panel/f'{h}_{c}.txt')) for c in range(2)] for h in heads};parent.keys_for=lambda data,road,split,date:daily[date].copy()
    for date in dates:
        k=daily[date];x,initial,ii,jj,obs,names=feature(e,date,k);cache=B/'outputs/state_density_pilot_20261102'/'parent_predictions'/panel/(panel+'_'+date+'.parquet')
        if cache.exists():pred=pd.read_parquet(cache)
        else:
            original=pd.read_parquet
            def guarded(path,*a,**kw):
                if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Parent inference tried to read truth')
                return original(path,*a,**kw)
            pd.read_parquet=guarded
            try:pred=parent.predict(panel,date,r,DATA,cache.parent,'cuda')
            finally:pd.read_parquet=original
        assert pred[base.KEYS].reset_index(drop=True).equals(k[base.KEYS].reset_index(drop=True));p=pred[['speed_kmh','flow_vph']].to_numpy();p[:,1]/=e.lanes[ii];truth=pd.read_parquet(e.truth_files[date]);y=np.column_stack([e.grid(truth,'speed_kmh')[ii,jj],e.grid(truth,'flow_vph')[ii,jj]/e.lanes[ii]]);kp=p[:,1]/np.maximum(p[:,0],1);free=e.static.free_speed_kmh.to_numpy()[ii]
        for head,model in models.items():
            candidate=np.maximum(np.column_stack([x[:,names.index('speed_kmh_initial')]+model[0].predict(x,num_threads=8),x[:,names.index('flow_vph_initial')]+model[1].predict(x,num_threads=8)]),0)
            for alpha in alphas:
                changed=p.copy()
                if alpha:changed=(1-alpha)*p+alpha*candidate
                item=dict(date=date,panel=panel,head=head,alpha=alpha,n=len(p),regime=k.mask_regime.iloc[0],speed_sse=float(np.square(changed[:,0]-y[:,0]).sum()),flow_sse=float(np.square(changed[:,1]-y[:,1]).sum()),transition=transition(e,changed,truth,ii,jj,obs));rows.append(item)
        log(stage=role,panel=panel,date=date,rows=len(p))
    save(OUT/f'{role}_{panel}.json',rows)
def aggregate(role,heads,alphas):
    rows=sum([read(OUT/f'{role}_{p}.json') for p in PANELS],[]);out={}
    for head in heads:
        for alpha in alphas:
            selected=[r for r in rows if r['head']==head and r['alpha']==alpha];road=[]
            for panel in PANELS:
                regs=[]
                for regime in ['R1','R2','R3']:
                    rr=[r for r in selected if r['panel']==panel and r['regime']==regime];n=sum(r['n'] for r in rr)
                    if n:regs.append(.54*max(0,1-np.sqrt(sum(r['speed_sse'] for r in rr)/n)/25)+.46*max(0,1-np.sqrt(sum(r['flow_sse'] for r in rr)/n)/600))
                rr=[r for r in selected if r['panel']==panel];pairs=sum(r['transition']['pairs'] for r in rr);road.append(dict(panel=panel,S_state=float(np.mean(regs)),dN_MAE=sum(r['transition']['absolute_error_sum'] for r in rr)/max(pairs,1),transition_pairs=pairs))
            out[f'{head}_a{alpha:g}']=dict(S_state=float(np.mean([r['S_state'] for r in road])),dN_MAE=float(np.mean([r['dN_MAE'] for r in road])),roads=road)
    return out
if __name__=='__main__':main()
