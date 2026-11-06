"""Retained conservation and boundary stages from the February retrain."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import stock_math as p

NORM_EPS = 1e-9


def prep(day):
    x,b,y,m=(day[k] for k in ('x','b','y','mask'))
    obs=x[...,4].astype(bool)
    t=np.arange(288)[:,None]
    left=np.maximum.accumulate(np.where(obs,t,-1),axis=0)
    right=np.minimum.accumulate(np.where(obs,t,288)[::-1],axis=0)[::-1]
    dl,dr=t-left,right-t
    a,c=left>=0,right<288
    length=right-left-1
    gap=np.take(['1','2-3','4-6','7-12','13+'],np.digitize(length,[2,4,7,13]))
    border=np.take(['both day edges','day start','day end','interior'],a.astype(int)*2+c)
    anchor=np.take(['none','one','two'],a.astype(int)+c)
    keys=dict(regime=np.full(m.shape,'R'+str(day['regime']+1)),gap=gap,anchor=anchor,border=border)
    day['keys']=keys
    day['dist']=(np.where(a,dl,np.nan),np.where(c,dr,np.nan))
    interp=(x[...,8:10]*dr[...,None]+x[...,10:12]*dl[...,None])/(dl+dr+obs)[...,None]
    interp=np.where((a&c)[...,None],interp,np.where(a[...,None],x[...,8:10],x[...,10:12]))
    temporal=np.where((a|c)[...,None],(interp-b)/np.sqrt(np.maximum(1,np.minimum(np.where(a,dl,288),np.where(c,dr,288))))[...,None],0)
    count=x[...,16:18].sum(-1)
    neighbor=(x[...,12:14]*x[...,16,None]+x[...,14:16]*x[...,17,None])/np.maximum(.25,count)[...,None]
    spatial=np.where((count>0)[...,None],neighbor-b,0)
    day['features']=np.stack([temporal*(anchor==kind)[...,None] for kind in ('one','two')]+[spatial],axis=-1)
    err=(b-y)*[25,600]
    points=pd.DataFrame({k:v[m] for k,v in keys.items()})
    points=points.assign(n=1, speed_sse=err[...,0][m]**2,flow_sse=err[...,1][m]**2,
                         left=day['dist'][0][m],right=day['dist'][1][m],neighbor=(count[m]>0).astype(int))
    for i,name in enumerate(('one','two','neighbor')):
        f=day['features'][m,:,i]*[25,600]
        for j,k in enumerate(('speed','flow')):
            points[name+'_'+k+'_dot']=-err[m,j]*f[:,j]
            points[name+'_'+k+'_norm']=f[:,j]**2
    pair=day['pairs']
    first=m[:-1]
    edges=pd.DataFrame({k:np.where(first,v[:-1],v[1:])[pair] for k,v in keys.items()})
    edge_err=(np.diff(p.stock(b,day['physical']),axis=0)-day['delta'])[pair]
    edges=edges.assign(edge=np.where((first&m[1:])[pair],'target-target','target-observed'),n=1,absolute=abs(edge_err),sse=edge_err**2)
    return points,edges


def features(d,month,panel):
    date=d['date']
    day=month[month.timestamp.dt.strftime('%Y-%m-%d').eq(date)]
    ts=day.timestamp
    t=(ts.dt.hour*12+ts.dt.minute//5).to_numpy()
    n=day.station_id.map(panel.exp.si).to_numpy(int)
    m=d['mask']
    lookup=np.zeros((*m.shape,15),dtype='float32')
    own=day.flow_vph.to_numpy()
    up=day.up_proxy_vph.to_numpy()
    down=day.down_proxy_vph.to_numpy()
    ramp_on=day.on_ramp_estimated_vph.to_numpy()
    ramp_off=day.off_ramp_estimated_vph.to_numpy()
    stock_net=day.required_mainline_net_vph.to_numpy()
    assumed=day.inferred_out_vph.to_numpy()
    inferred_in=day.inferred_in_vph.to_numpy()
    reliable_up=(day.up_observed_count==day.up_linked_count)&day.up_linked_count.gt(0)
    reliable_down=(day.down_observed_count==day.down_linked_count)&day.down_linked_count.gt(0)
    ramp_good=day.ramp_all_observed
    columns=[(up-own)/1200,(down-own)/1200, (ramp_on-ramp_off)/600, stock_net/1200,
             (assumed-own)/1200,(inferred_in-own)/1200, np.isfinite(up),np.isfinite(down),
             np.isfinite(assumed),np.isfinite(inferred_in),reliable_up,reliable_down,ramp_good,
             day.up_declared_count-day.up_linked_count,day.down_declared_count-day.down_linked_count]
    lookup[t,n]=np.nan_to_num(np.stack(columns,axis=-1),nan=0).clip(-4,4)
    # Training targets use only published task labels, never inferred flows.
    baseline=np.concatenate((np.ones((*m.shape,1)),d['x'][...,:3],d['x'][...,6:8],
                             d['x'][...,16:21],d['features'].reshape(*m.shape,6)),axis=-1)
    for k,design in [('control',baseline),('derived',np.concatenate((baseline,lookup),axis=-1))]:
        d[k]=(design*m[...,None]).astype('float32')
    context=lookup.copy()
    context[...,4:6]=0
    context[...,8:10]=0
    d['context_only']=(np.concatenate((baseline,context),axis=-1)*m[...,None]).astype('float32')
    gated=lookup.copy()
    reliable=lookup[...,12]>0
    up_good=lookup[...,10]>0
    down_good=lookup[...,11]>0
    gated[...,0]*=up_good
    gated[...,1]*=down_good
    gated[...,2:4]*=reliable[...,None]
    gated[...,4]*=reliable&up_good
    gated[...,5]*=reliable&down_good
    d['quality_gated']=(np.concatenate((baseline,gated),axis=-1)*m[...,None]).astype('float32')


def predict(d,arm,coef,scale):
    b=d['b']
    v=b[...,0].clip(.04)
    ratio=b[...,1]/v
    correction=scale*(d[arm]@coef)
    updated=np.maximum(0,ratio+correction)
    pred=b.copy()
    m=d['mask']
    pred[...,1][m]=(updated*v)[m]
    return pred


def measure(d,pred):
    dp=np.diff(p.stock(pred,d['physical']),axis=0)
    truth=d['delta']
    m=d['mode']
    t=m&d['pairs']
    signed=dp[t]-truth[t]
    gap=abs(dp[t])-abs(truth[t])
    return dict(regime=d['regime'],n=int(m.sum()),error=float(abs(dp[m]-truth[m]).sum()),
                rhs=float(abs(truth[m]).sum()),target_edges=int(t.sum()),
                reverse=int(((dp[t]*truth[t]<0)&(abs(dp[t])>1e-9)&(abs(truth[t])>1e-9)).sum()),
                smaller=int((gap<-1e-9).sum()),larger=int((gap>1e-9).sum()),
                signed_bias=float(signed.sum()))


def summary(rows):
    by=[]
    for k in sorted({r['regime'] for r in rows}):
        group=[r for r in rows if r['regime']==k]
        agg={key:sum(r[key] for r in group) for key in ('n','error','rhs','target_edges','reverse','smaller','larger','signed_bias')}
        agg.update(regime=f'R{k+1}',normalized_error=agg['error']/max(NORM_EPS,agg['rhs']),
                   mae_vph=12*agg['error']/agg['n'])
        by.append(agg)
    return dict(normalized_error_macro=float(np.mean([r['normalized_error'] for r in by])),
                mae_vph_macro=float(np.mean([r['mae_vph'] for r in by])),
                missing_regimes=[f'R{k+1}' for k in range(3) if f'R{k+1}' not in {r['regime'] for r in by}],
                by_regime=by)


def old_predict(d,lock,coef):
    return predict(d,lock['arm'],coef,lock['scale'])


def attach_features(d,baseline):
    d['prior']=baseline
    m=d['mask']
    shape=m.shape
    t=np.arange(288)[:,None]
    d['block']=np.broadcast_to(t//36,shape)
    v=baseline[...,0]
    q=baseline[...,1]
    n=p.stock(baseline,d['physical'])
    prev=np.zeros_like(n)
    nxt=np.zeros_like(n)
    diff=np.diff(n,axis=0)*d['valid']
    prev[1:]=diff
    nxt[:-1]=diff
    z=np.concatenate((d['quality_gated'],np.stack((prev/8,nxt/8,(nxt-prev)/8),axis=-1)),axis=-1)
    d['z']=np.clip(z,-4,4)
    d['local']=np.stack((np.ones(shape),v/4,q,n/12,prev/8,nxt/8),axis=-1).clip(-4,4)
    d['residual']=d['delta']-np.diff(n,axis=0)
    d['eligible']=d['mode']&d['pairs']


def design(d,hot):
    active=hot[d['block'],np.arange(hot.shape[1])]&d['mask']
    return d['z']*active[...,None],d['local']*active[...,None]


def model_predict(d,hot,shared,local,scale):
    b=d['prior']
    x,z=design(d,hot)
    correction=x@shared
    if local is not None:
        correction+=np.einsum('tnf,nf->tn',z,local)
    v=b[...,0].clip(.04)
    pred=b.copy()
    touched=(correction!=0)&d['mask']
    pred[...,1][touched]=(np.maximum(0,b[...,1]/v+scale*correction)*v)[touched]
    return pred


def baseline_stage(ds,lock,coef):
    for d in ds:
        attach_features(d,old_predict(d,lock,coef))


def cascade_frozen(prior,hot):
    return (np.asarray(prior['coef']),np.asarray(hot['hot'],dtype=bool),
            np.asarray(hot['shared']),np.asarray(hot['local']) if hot['local'] is not None else None,hot['scale'])


def boundary_stage(ds,prior,coef):
    a,h,b,c,scale=coef
    baseline_stage(ds,prior,a)
    for d in ds:
        d['reference']=model_predict(d,h,b,c,scale)
        m=d['mask']
        obs=d['x'][...,4].astype(bool)
        eligible=d['mode']&d['pairs']
        entry=eligible&obs[:-1]&m[1:]
        exit_=eligible&m[:-1]&obs[1:]
        interior=eligible&m[:-1]&m[1:]
        d['scopes']=dict(entry=entry,exit=exit_,interior=interior)
        stock=p.stock(d['reference'],d['physical'])
        dp=np.diff(stock,axis=0)
        d['boundary_residual']=d['delta']-dp
        shape=m.shape
        z0=np.concatenate((d['quality_gated'],(stock/12)[...,None]),axis=-1)
        signals=[]
        for k,edge in enumerate((entry,exit_)):
            t,n=np.where(edge)
            target_t=t+1 if k==0 else t
            # Difference follows the same time direction for both edge types.
            z=np.zeros((*shape,z0.shape[-1]+3),dtype='float32')
            z[target_t,n,:-3]=z0[target_t,n]
            z[target_t,n,-3]=dp[t,n]/8
            z[target_t,n,-2]=stock[t,n]/12 if k==0 else stock[t+1,n]/12
            z[target_t,n,-1]=1
            signals.append(z.clip(-4,4))
        d['boundary_design']=signals


def boundary_design(d,arm):
    x,y=d['boundary_design']
    if arm=='entry':
        return x
    if arm=='exit':
        return y
    if arm=='shared':
        return x+y
    return np.concatenate((x,y),axis=-1)


def boundary_predict(d,arm,coef,scale):
    b=d['reference']
    dz=scale*(boundary_design(d,arm)@coef)
    changed=dz!=0
    v=b[...,0].clip(.04)
    out=b.copy()
    out[...,1][changed]=(np.maximum(0,b[...,1]/v+dz)*v)[changed]
    return out


def design_rmse(d,h):
    b=d['reference']
    n=p.stock(b,d['physical'])
    shape=n.shape
    tick=np.broadcast_to(np.arange(shape[0])[:,None],shape)
    nodes=np.broadcast_to(np.arange(shape[1]),shape)
    exact,near,count=h
    fixed=np.stack((n,b[...,0]),axis=-1)
    exact=np.where(np.isfinite(exact),exact,np.where(np.isfinite(near),near,fixed))
    near=np.where(np.isfinite(near),near,exact)
    arr=[d['quality_gated'],d['x'][...,5:16],(n/12)[...,None],b]
    for k in (-2,-1,1,2):
        shifted=np.roll(n,k,axis=0)
        known=np.roll(d['x'][...,5]>0,k,axis=0)
        if k>0:
            known[:k]=False
        else:
            known[k:]=False
        arr.extend([np.where(known,(shifted-n)/12,0)[...,None],known[...,None]])
    arr.append(np.stack((tick/288,nodes/shape[1],d['physical'],(exact[...,0]-n)/12,
                         (near[...,0]-n)/12,exact[...,1]-b[...,0],near[...,1]-b[...,0],
                         count/10,d['x'][...,4]),axis=-1))
    x=np.concatenate(arr,axis=-1).astype('float32')
    candidate=np.where(count>=2,exact[...,0]-n,near[...,0]-n)
    return x,candidate


def allocate(d,adjust,scale,alpha):
    b=d['reference']
    stock=p.stock(b,d['physical'])
    proposed=(stock+scale*adjust).clip(0)
    m=d['mask']&(abs(proposed-stock)>1e-10)
    old_v=b[...,0].clip(.04)
    ratio=proposed/np.maximum(1e-9,24*d['physical'])
    ratio0=b[...,1]/old_v
    mult=np.divide(ratio,ratio0,out=np.ones_like(ratio),where=ratio0>1e-9)
    v=(old_v/np.maximum(.1,mult)**alpha).clip(.04,6.)
    q=ratio*v
    return np.where(m[...,None],np.stack((v,q),axis=-1),b)


def adjacency(panel):
    result=[]
    for key in ('up','down'):
        a=np.zeros((panel.n,panel.n),dtype='float32')
        for i,ids in enumerate(panel.exp.neighbors[key]):
            if len(ids):
                a[ids,i]=1
        result.append(a)
    return result


def shifted(x,known,lag):
    if lag==0:
        return x,known
    q=np.roll(x,lag,axis=0)
    ok=np.roll(known,lag,axis=0)
    if lag>0:
        ok[:lag]=False
    else:
        ok[lag:]=False
    return q,ok


def tail_design(d,h,graph):
    x,_=design_rmse(d,h)
    ref=d['reference']
    n=p.stock(ref,d['physical'])
    ratio=ref[...,1]/ref[...,0].clip(.04)
    known=d['x'][...,5]>0
    obs=d['x'][...,4]>0
    added=[]
    for adj in graph:
        for mask in (known,obs):
            count=mask.astype('float32')@adj
            mean=np.divide((ratio*mask)@adj,count,out=np.zeros_like(ratio),where=count>0)
            for lag in (-1,0,1):
                a,b=shifted(mean,count>0,lag)
                added.extend([np.where(b,(a-ratio)/2,0),b])
    before,seen0=shifted(n,known,1)
    after,seen1=shifted(n,known,-1)
    back=np.where(seen0,before-n,0)
    ahead=np.where(seen1,after-n,0)
    added.extend((back/12,ahead/12,abs(back-ahead)/12,np.minimum(abs(back),abs(ahead))/12))
    phase=np.stack((back,ahead),axis=-1)
    return np.concatenate((x,np.stack(added,axis=-1).astype('float32')),axis=-1),phase


def signals(d,h,graph,fit):
    tree,risk,phase=fit
    x,pv=tail_design(d,h,graph)
    m=d['mask']
    scores=risk.predict(x[m],num_threads=2)
    z=np.zeros(m.shape)
    z[m]=tree.predict(x[m],num_threads=2)
    return scores,dict(tree=z,phase=pv@phase)


def masks_by_budget(d,risk):
    index=np.flatnonzero(d['mask'])
    order=index[np.argsort(risk,kind='stable')[::-1]]
    masks={}
    for f in (.03,.10,.30):
        mask=np.zeros(d['mask'].size,dtype=bool)
        mask[order[:max(1,int(np.ceil(f*len(order))))]]=True
        masks[f]=mask.reshape(d['mask'].shape)
    return masks


def scored(d,pred,params):
    row=measure(d,pred)
    e=(np.diff(p.stock(pred,d['physical']),axis=0)-d['delta'])[d['mode']]
    v,q=pred[...,0]*25,pred[...,1]*600
    target=d['mask']
    err=(pred-d['y'])[target]
    row.update(sse=float(e@e),sv=float(err[:,0]@err[:,0]),sq=float(err[:,1]@err[:,1]),
               points=int(target.sum()),empty=int((q[target]*np.broadcast_to(params[-1],target.shape)[target]<50).sum()))
    vf,cap,kcrit,kjam,lanes=params
    known=target|d['x'][...,4].astype(bool)
    k=q/v.clip(1)
    qfd=np.where(k<=kcrit/lanes,vf*k,(cap/np.maximum(kjam-kcrit,1e-9))*np.maximum(kjam/lanes-k,0))
    row['fd_error']=float(abs(q-qfd)[known].sum())
    row['fd_den']=float(abs(q[known]).sum())
    return row


def scored_summary(rows):
    state=summary(rows)
    rmse=[]; points=[]; fd=[]
    for r in sorted({d['regime'] for d in rows}):
        group=[x for x in rows if x['regime']==r]
        t={k:sum(x[k] for x in group) for k in ('sse','n','sv','sq','points','fd_error','fd_den','empty')}
        rmse.append(float(np.sqrt(t['sse']/t['n'])))
        points.append(.54*max(0,1-np.sqrt(t['sv']/t['points']))+.46*max(0,1-np.sqrt(t['sq']/t['points'])))
        fd.append(0 if t['empty']/t['points']>.2 else max(0,1-t['fd_error']/max(1e-9,t['fd_den'])))
    return dict(state,rmse_macro=float(np.mean(rmse)),rmse_pooled=float(np.sqrt(sum(x['sse'] for x in rows)/sum(x['n'] for x in rows))),
                state_score=float(np.mean(points)),fd=float(np.mean(fd)))
