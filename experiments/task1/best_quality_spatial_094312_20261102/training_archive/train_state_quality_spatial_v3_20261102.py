"""Masked-state temporal corrector with supervised accumulation-increment loss.

All features use released masked state or frozen model outputs. Official LWR
boundary flux is unavailable, so this is NOT an official physics-score trainer.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,time,json,hashlib,gc,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
B=Path(__file__).resolve().parents[1];M=B/'work/repo/experiments/task1/best_road_combination_20261026';DATA=B/'work/local_transfer/data/kaggle_public';OUT=B/'outputs/state_quality_spatial_v3_20261102';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(M));import predict_selected as parent;import feature_runtime as base
from train_state_density_pilot_20261102 import choose_dates
torch.set_num_threads(2);PANELS=list(json.loads((M/'selected-road-models.json').read_text(encoding='utf8'))['routes']);LAMBDAS=[0.];SEEDS=[20261104,20261105];START=time.time()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(**kw):print(json.dumps(dict(seconds=round(time.time()-START,1),**kw)),flush=True)
class Corrector(nn.Module):
    def __init__(self,features):
        super().__init__();self.input=nn.Conv1d(features,64,1);self.blocks=nn.ModuleList([nn.Conv1d(64,64,3,padding=d,dilation=d) for d in [1,2,4,8]]);self.output=nn.Conv1d(64,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        h=F.silu(self.input(x))
        for block in self.blocks:h=h+.2*F.silu(block(h))
        return .2*torch.tanh(self.output(h))
class Ensemble(nn.Module):
    def __init__(self,members):super().__init__();self.members=nn.ModuleList(members)
    def forward(self,x):return torch.stack([m(x) for m in self.members]).mean(dim=0)
def grids(e,date,keys,r,role):
    file=OUT/'data'/e.args.panel/(date+'.npz')
    prior=B/'outputs/state_quality_spatial_overnight_20261102/data'/e.args.panel/(date+'.npz')
    if not file.exists() and prior.exists():
        # Immutable read-only cache: identical masked inputs, parent and features.
        # The caller requests only fit/December dates until selection freezes.
        with np.load(prior) as z:cached={k:z[k] for k in z.files}
        assert str(cached['panel'])==e.args.panel and str(cached['date'])==date and cached['x'].shape[1]==50
        return cached
    if file.exists():
        try:
            with np.load(file) as z:return {k:z[k] for k in z.files}
        except (EOFError,OSError,ValueError,zipfile.BadZipFile):
            log(stage='REBUILD_INTERRUPTED_CACHE',file=str(file))
    obs=pd.read_parquet(e.masked_files[date]);ts=pd.to_datetime(keys.timestamp,utc=True);ii=keys.station_id.map(e.si).to_numpy(int);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
    original=pd.read_parquet
    def guarded(path,*a,**kw):
        if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Unmasked labels accessed during temporal input construction')
        return original(path,*a,**kw)
    pd.read_parquet=guarded
    try:
        values,initial,_,_=e.feature_grid(date);cache=B/'outputs/state_density_pilot_20261102/parent_predictions'/e.args.panel/(e.args.panel+'_'+date+'.parquet')
        if not cache.exists():
            parent.keys_for=lambda data,road,split,which:keys.copy();pred=parent.predict(e.args.panel,date,r,DATA,cache.parent,'cuda')
        else:pred=pd.read_parquet(cache)
    finally:pd.read_parquet=original
    assert pred[base.KEYS].reset_index(drop=True).equals(keys[base.KEYS].reset_index(drop=True))
    v=e.grid(obs,'speed_kmh');q=e.grid(obs,'flow_vph')/e.lanes[:,None];observed=np.isfinite(v)&np.isfinite(q);mask=np.zeros_like(v,dtype=bool);mask[ii,jj]=True;assert not observed[ii,jj].any()
    v[ii,jj]=pred.speed_kmh;q[ii,jj]=pred.flow_vph/e.lanes[ii];usable=np.isfinite(v)&np.isfinite(q);v=np.where(np.isfinite(v),v,initial[0]);q=np.where(np.isfinite(q),q,initial[1]);p=np.stack([v/25,q/600],axis=1).astype('float32');n=e.n
    features=[v/100,q/600,(q/np.maximum(v,1))/30,observed.astype(float),mask.astype(float),initial[0]/100,initial[1]/600,values['speed_kmh_prev_gap_min']/60,values['speed_kmh_next_gap_min']/60,np.broadcast_to(e.static.free_speed_kmh.to_numpy()[:,None]/100,v.shape),np.broadcast_to(e.static.length_km.to_numpy()[:,None],v.shape),np.broadcast_to(e.lanes[:,None]/5,v.shape),values['slot_sin'],values['slot_cos']]
    extra=[('pct_observed',100.),('is_imputed',1.),('weekday',6.),('speed_kmh_up',100.),('speed_kmh_down',100.),('flow_vph_up',600.),('flow_vph_down',600.),('speed_kmh_up_count',4.),('speed_kmh_down_count',4.),('relative_speed_initial',1.),('capacity_utilization_initial',1.)];assert all(k in values for k,scale in extra);features += [values[k]/scale for k,scale in extra];x=np.stack(features,axis=1);x=np.concatenate([np.nan_to_num(x,nan=0,posinf=0,neginf=0).clip(-20,20),np.isfinite(x)],axis=1).astype('float32');assert np.isfinite(x).all()
    # Read targets only after the entire input tensor and frozen prediction exist.
    truth=pd.read_parquet(e.truth_files[date]);tv=e.grid(truth,'speed_kmh');tq=e.grid(truth,'flow_vph')/e.lanes[:,None];valid=np.isfinite(tv)&np.isfinite(tq)&(e.grid(truth,'is_score_eligible')>.5);y=np.nan_to_num(np.stack([tv/25,tq/600],axis=1),nan=0).astype('float32');pair=valid[:,:-1]&valid[:,1:]&usable[:,:-1]&usable[:,1:]&(mask[:,:-1]|mask[:,1:]);length=(e.static.length_km.to_numpy()*e.lanes).astype('float32');regime=str(keys.mask_regime.iloc[0]);result=dict(x=x,p=p,y=y,mask=mask&valid,pair=pair,length=length,regime=np.array(regime),panel=np.array(e.args.panel),date=np.array(date));file.parent.mkdir(parents=True,exist_ok=True);tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,file);return result
def accumulate(model,arrays):
    totals={}
    for a in arrays:
        with torch.no_grad():
            x=torch.from_numpy(a['x']).cuda();p=torch.from_numpy(a['p']).cuda();mask=torch.from_numpy(a['mask']).cuda();delta=torch.zeros_like(p) if model is None else model(x);pred=p+delta*mask[:,None];pred=torch.stack([pred[:,0].clamp(min=0.),pred[:,1].clamp(min=0)],dim=1);v=pred[:,0]*25;q=pred[:,1]*600;N=(q/v.clamp(min=1))*torch.from_numpy(a['length']).cuda()[:,None];ty=torch.from_numpy(a['y']).cuda();TN=ty[:,1]*600/(ty[:,0]*25).clamp(min=1)*torch.from_numpy(a['length']).cuda()[:,None];diff=torch.diff(N,dim=1)-torch.diff(TN,dim=1);pair=torch.from_numpy(a['pair']).cuda();err=(pred-ty)*torch.tensor([25.,600.],device='cuda')[None,:,None];sse=(err.square()*mask[:,None]).sum(dim=(0,2)).cpu().numpy();n=int(mask.sum());ae=float(diff[pair].abs().sum());npair=int(pair.sum())
        key=(str(a['panel']),str(a['regime']));z=totals.setdefault(key,dict(n=0,speed_sse=0.,flow_sse=0.,pair=0,ae=0.));z['n']+=n;z['speed_sse']+=float(sse[0]);z['flow_sse']+=float(sse[1]);z['pair']+=npair;z['ae']+=ae
    roads=[]
    for panel in PANELS:
        rows=[z for (p,r),z in totals.items() if p==panel];ss=[.54*max(0,1-np.sqrt(z['speed_sse']/z['n'])/25)+.46*max(0,1-np.sqrt(z['flow_sse']/z['n'])/600) for z in rows];roads.append(dict(panel=panel,S_state=float(np.mean(ss)),dN_MAE=sum(z['ae'] for z in rows)/sum(z['pair'] for z in rows),target_rows=sum(z['n'] for z in rows)))
    return dict(S_state=float(np.mean([r['S_state'] for r in roads])),dN_MAE=float(np.mean([r['dN_MAE'] for r in roads])),roads=roads)
def main():
    torch.manual_seed(20261102);torch.cuda.manual_seed_all(20261102);routes=parent.load_routes();plan=read(M/'reproduction/vendor/jayden2/reference-split.json');dates=dict(fit=choose_dates([d for d in plan['training_dates'] if d<='2030-11-30'],6),December=choose_dates([d for d in plan['internal_validation_dates'] if d<'2031-01-01'],6),January=choose_dates([d for d in plan['holdout_dates'] if d.startswith('2031-01')],6),February=choose_dates([d for d in plan['holdout_dates'] if d.startswith('2031-02')],6));protocol=dict(decision_basis='December-v2-only:unpenalized control gains S_state .00011985 and reduces dN_MAE .82percent;physics penalties regress. V3 increases June-Nov fit coverage4to6days/month,width32to64,24epochs,lowerlr,strongerweightdecay,two new fixedseeds. January/February are not used tochoose changes.',panels=PANELS,dates=dates,lambdas=LAMBDAS,epochs=24,batch_station_sequences=64,seeds=SEEDS,loss='0.54speedMSE/25²+0.46flowperlaneMSE/600² +lambda*(dN_pred-dN_observed)MSE/25; only valid target-adjacent transitions. Known states hard-preserved.',architecture='All10roads shared quality-aware actual-neighbor temporal residual CNN,64channels,dilations1,2,4,8;two fixed-seed ensemble. Adds released QC,imputation,weekday,observed real upstream/downstream states andcounts;no adjacency invented. Sampling equalizes road/regime. Zero-init bounded5kmh/120vph perlane. Task1 offline observed temporalcontext allowed;Task2unchanged.',selection='Decemberonly,eachroadS_state>=parent-.001;accept either stategain>=.0002 anddN_MAE<=parent*.999,orstate>=parent-.0003 anddN_MAE<=parent*.98. MaximizeS_state amongeligible,tie lowerdN_MAE. Otherwise parent. January confirmation andFebruaryreport no tuning;no officialphysics claim.',official_S_physics=None,caveat='Observed truth surrogate,not hidden boundaryflux. All10roads butsampleddates,notfull55orofficialtotal. Two fixed seeds,modelselectionbeforeJanuary/February. Task2completelyunchanged.')
    save(OUT/'protocol.json',protocol);data={role:[] for role in dates};frozen={str(M/'selected-road-models.json'):sha(M/'selected-road-models.json')}
    # Do not build January/February features or inspect labels until selection freezes.
    def prepare(role):
        for panel in PANELS:
            e=base.Experiment(SimpleNamespace(release_root=DATA,panel=panel,output=OUT/'metadata'/panel,seed=20261102));r=routes[panel];h=np.load(r['paths'].get('history',r.get('history')));e.history=h['mean'] if isinstance(h,np.lib.npyio.NpzFile) else h;keys=pd.read_csv(DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str);daily={d:k.reset_index(drop=True) for d,k in keys.groupby(keys.timestamp.str[:10])}
            for p in r['paths'].values():frozen[p]=sha(Path(p))
            for date in dates[role]:data[role].append(grids(e,date,daily[date],r,role));log(stage='DATA',role=role,panel=panel,date=date)
    prepare('fit');prepare('December');fit=data['fit'];X=np.concatenate([a['x'] for a in fit]);P=np.concatenate([a['p'] for a in fit]);Y=np.concatenate([a['y'] for a in fit]);Mask=np.concatenate([a['mask'] for a in fit]);Pair=np.concatenate([a['pair'] for a in fit]);Length=np.concatenate([a['length'] for a in fit]);ds=torch.utils.data.TensorDataset(*[torch.from_numpy(z) for z in [X,P,Y,Mask,Pair,Length]]);baseline=accumulate(None,data['December']);reports={};models={}
    sequence_weight=[]
    counts={}
    for a in fit:
        key=(str(a['panel']),str(a['regime']));counts[key]=counts.get(key,0)+1
    for a in fit:
        key=(str(a['panel']),str(a['regime']));sequence_weight.extend([1/(len(a['x'])*counts[key])]*len(a['x']))
    sequence_weight=torch.tensor(sequence_weight,dtype=torch.double)
    for lam in LAMBDAS:
        members=[]
        for seed in SEEDS:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Corrector(X.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=3e-3);generator=torch.Generator().manual_seed(seed);sampler=torch.utils.data.WeightedRandomSampler(sequence_weight,len(ds),replacement=True,generator=generator);loader=torch.utils.data.DataLoader(ds,batch_size=64,sampler=sampler,num_workers=0)
            for epoch in range(24):
                model.train();total=0.;steps=0
                for batch in loader:
                    x,p,y,mask,pair,length=[v.cuda() for v in batch];pred=p+model(x)*mask[:,None];pred=torch.stack([pred[:,0].clamp(min=0.),pred[:,1].clamp(min=0)],dim=1);err=(pred-y).square();state=((.54*err[:,0]+.46*err[:,1])*mask).sum()/mask.sum().clamp(min=1);N=pred[:,1]*600/(pred[:,0]*25).clamp(min=1)*length[:,None];TN=y[:,1]*600/(y[:,0]*25).clamp(min=1)*length[:,None];d=torch.diff(N,dim=1)-torch.diff(TN,dim=1);physics=(d.square()*pair).sum()/pair.sum().clamp(min=1)/25;loss=state+lam*physics;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach());steps+=1
                log(stage='TRAIN',lambda_value=lam,seed=seed,epoch=epoch+1,loss=total/steps)
            model.eval();members.append(model)
        model=Ensemble(members).cuda().eval();result=accumulate(model,data['December']);name=f'lambda{lam:g}';reports[name]=result;models[name]=model;path=OUT/'models'/(name+'.pt');path.parent.mkdir(exist_ok=True);torch.save(dict(state_dict=model.state_dict(),features=X.shape[1],lambda_value=lam,seeds=SEEDS,architecture='quality_spatial_ensemble_v2'),path);log(stage='DECEMBER',name=name,**result)
    eligible=[(k,v) for k,v in reports.items() if all(a['S_state']>=b['S_state']-.001 for a,b in zip(v['roads'],baseline['roads'])) and ((v['S_state']>=baseline['S_state']+.0002 and v['dN_MAE']<=baseline['dN_MAE']*.999) or (v['S_state']>=baseline['S_state']-.0003 and v['dN_MAE']<=baseline['dN_MAE']*.98))];name=max(eligible,key=lambda kv:(kv[1]['S_state'],-kv[1]['dN_MAE']))[0] if eligible else 'parent';save(OUT/'selection.json',dict(selected=name,baseline=baseline,candidates=reports,eligible=[k for k,v in eligible]));log(stage='SELECTED',selected=name)
    files=[Path(__file__),OUT/'protocol.json',OUT/'selection.json']+list((OUT/'models').glob('*.pt'));frozen.update({str(p):sha(p) for p in files});save(OUT/'frozen_before_January_February.json',frozen);selected=models.get(name)
    for role in ['January','February']:
        prepare(role);res=dict(baseline=accumulate(None,data[role]),selected=accumulate(selected,data[role]),selected_name=name);save(OUT/(role+'_results.json'),res);log(stage=role.upper(),**res)
    for p,h in frozen.items():assert sha(Path(p))==h
    save(OUT/'verification.json',dict(passed=True,all_parent_hashes_unchanged=True,selection_frozen_before_January_February=True,official_total=None,official_S_physics=None));log(stage='COMPLETE')
if __name__=='__main__':main()
