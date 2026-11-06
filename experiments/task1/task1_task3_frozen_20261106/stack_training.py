"""Retained supervised fitting functions."""
import numpy as np
import lightgbm as lgb

import stock_math as p
from stack import NORM_EPS, boundary_design, design, design_rmse, tail_design


def fit(ds,arm,penalty):
    x=np.concatenate([d['edge_data'][arm] for d in ds])
    y=np.concatenate([d['response'] for d in ds])
    weights=np.concatenate([np.full(len(d['response']),1/max(NORM_EPS,sum(x['truth_rhs'] for x in ds if x['regime']==d['regime'])))
                            for d in ds])
    weights/=weights.sum()
    c=np.zeros(x.shape[1])
    regularizer=penalty*np.diag(np.sum(x*x*weights[:,None],axis=0)+1e-8)
    for _ in range(8):
        # Smooth absolute residual, with a small fixed numerical transition.
        w=weights/np.maximum(.05,abs(x@c-y))
        c=np.linalg.solve((x.T*w)@x+regularizer,x.T@(w*y))
    return c


def hotspots(ds,fraction):
    n=ds[0]['mask'].shape[1]
    all_n=np.zeros((8,n))
    losses=np.zeros((8,n))
    dates=np.zeros((8,n))
    for d in ds:
        t,k=np.where(d['eligible'])
        key=(t//36)*n+k
        all_n+=np.bincount(key,minlength=8*n).reshape(8,n)
        dates+=np.bincount(key,minlength=8*n).reshape(8,n)>=8
        losses+=np.bincount(key,weights=abs(d['residual'][t,k]),minlength=8*n).reshape(8,n)
    ok=np.flatnonzero((all_n>=60)&(dates>=3))
    keep=int(np.ceil(fraction*len(ok)))
    chosen=ok[np.argsort(losses.ravel()[ok],kind='stable')[::-1][:keep]]
    result=np.zeros(8*n,dtype=bool)
    result[chosen]=True
    return result.reshape(8,n)


def robust(x,y,weights,penalty):
    if not len(x):
        return np.zeros(x.shape[-1])
    weights=weights/max(1e-9,weights.sum())
    c=np.zeros(x.shape[-1])
    shrink=penalty*np.diag(np.sum(x*x*weights[:,None],axis=0)+1e-8)
    for _ in range(6):
        w=weights/np.maximum(.08,abs(x@c-y))
        c=np.linalg.solve((x.T*w)@x+shrink,x.T@(w*y))
    return c


def fit_hot(ds,hot):
    k=[]; xx=[]; zz=[]; yy=[]; ww=[]
    rhs={r:sum(d['truth_rhs'] for d in ds if d['regime']==r) for r in {d['regime'] for d in ds}}
    for d in ds:
        x,z=design(d,hot)
        physical=(24*d['physical'])[...,None]
        gx=np.diff(x*physical,axis=0)
        gz=np.diff(z*physical,axis=0)
        mask=d['eligible']&((gx!=0).any(-1)|(gz!=0).any(-1))
        xx.append(gx[mask])
        zz.append(gz[mask])
        yy.append(d['residual'][mask])
        k.append(np.where(mask)[1])
        ww.append(np.full(mask.sum(),1/max(1e-9,rhs[d['regime']])))
    x,z,y,node,weight=map(np.concatenate,(xx,zz,yy,k,ww))
    shared=robust(x,y,weight,.35)
    delta=y-x@shared
    local=np.zeros((hot.shape[1],z.shape[1]))
    for n in np.unique(node):
        use=node==n
        if use.sum()>=120:
            local[n]=robust(z[use],delta[use],weight[use],1.)
    return shared,local


def boundary_fit(ds,arm):
    x=[]; y=[]; w=[]
    totals={r:sum(d['truth_rhs'] for d in ds if d['regime']==r) for r in {d['regime'] for d in ds}}
    for d in ds:
        z=boundary_design(d,arm)
        diff=np.diff(z*(24*d['physical'])[...,None],axis=0)
        use=(d['mode']&d['pairs'])&(diff!=0).any(-1)
        x.append(diff[use])
        y.append(d['boundary_residual'][use])
        w.append(np.full(use.sum(),1/max(1e-9,totals[d['regime']])))
    return robust(np.concatenate(x),np.concatenate(y),np.concatenate(w),.2)


def historical(ds):
    total=np.zeros((*ds[0]['mask'].shape,2))
    count=np.zeros(ds[0]['mask'].shape)
    for d in ds:
        known=d['mask']|d['x'][...,4].astype(bool)
        values=np.stack((p.stock(d['y'],d['physical']),d['y'][...,0]),axis=-1)
        total+=values*known[...,None]
        count+=known
    def summed(a,r):
        zero=np.zeros((1,*a.shape[1:]))
        prefix=np.concatenate((zero,a.cumsum(axis=0)),axis=0)
        t=np.arange(288)
        return prefix[(t+r+1).clip(max=288)]-prefix[(t-r).clip(min=0)]
    near_n=summed(count,2)
    exact=np.divide(total,count[...,None],out=np.full_like(total,np.nan),where=count[...,None]>0)
    near=np.divide(summed(total,2),near_n[...,None],out=np.full_like(total,np.nan),where=near_n[...,None]>0)
    return exact,near,count


def pack_tree(ds):
    matrix=[]; left=[]; right=[]; target=[]; weight=[]
    offset=0
    support={}
    for d in ds:
        support[d['regime']]=support.get(d['regime'],0)+int(d['mode'].sum())
    for d in ds:
        h=historical([e for e in ds if e['date']!=d['date']])
        x,_=design_rmse(d,h)
        m=d['mask']
        indices=np.full(m.shape,-1,dtype=int)
        count=int(m.sum())
        indices[m]=offset+np.arange(count)
        e=d['mode']&d['pairs']
        matrix.append(x[m])
        left.append(indices[:-1][e])
        right.append(indices[1:][e])
        target.append((np.diff(p.stock(d['reference'],d['physical']),axis=0)-d['delta'])[e])
        weight.append(np.full(e.sum(),1/max(1,support[d['regime']])))
        offset+=count
    x=np.concatenate(matrix)
    a,b,y,w=map(np.concatenate,(left,right,target,weight))
    a=np.where(a<0,offset,a)
    b=np.where(b<0,offset,b)
    w*=len(w)/w.sum()
    degree=(np.bincount(a,weights=w,minlength=offset+1)+np.bincount(b,weights=w,minlength=offset+1))[:-1]
    hess=np.maximum(1e-6,degree*2)
    def objective(pred,data):
        padded=np.append(pred,0.)
        error=(padded[b]-padded[a]+y)*w
        g=(np.bincount(b,weights=error,minlength=offset+1)-np.bincount(a,weights=error,minlength=offset+1))[:-1]
        return g,hess
    booster=lgb.train(dict(objective=objective,verbosity=-1,learning_rate=.08,num_leaves=31,
                           min_data_in_leaf=45,lambda_l2=10,feature_fraction=.85,
                           num_threads=2,deterministic=True,force_col_wise=True,boost_from_average=False),
                       lgb.Dataset(x,label=np.zeros(len(x))),num_boost_round=65)
    return booster


def train_tail(ds,graph):
    x=[]; e0=[]; e1=[]; residual=[]; weights=[]; labels=[]; shifts=[]
    counts={k:sum(int(d['mode'].sum()) for d in ds if d['regime']==k) for k in {d['regime'] for d in ds}}
    thresholds={}
    for r in counts:
        e=np.concatenate([(np.diff(p.stock(d['reference'],d['physical']),axis=0)-d['delta'])[d['mode']]
                          for d in ds if d['regime']==r])
        thresholds[r]=np.quantile(abs(e),.99)
    offset=0
    for d in ds:
        use=d['mode']&d['pairs']
        m=d['mask']
        local=np.full(m.shape,-1,dtype=int)
        local[m]=offset+np.arange(m.sum())
        y=(np.diff(p.stock(d['reference'],d['physical']),axis=0)-d['delta'])[use]
        tail=abs(y)>=max(1e-6,thresholds[d['regime']])
        a=local[:-1][use]
        b=local[1:][use]
        y_cls=np.zeros(int(m.sum()),dtype=np.int8)
        for ids in (a,b):
            hit=ids[(ids>=0)&tail]-offset
            y_cls[hit]=1
        h=historical([e for e in ds if e['date']!=d['date']])
        mat,phase=tail_design(d,h,graph)
        x.append(mat[m])
        shifts.append(np.where(m[...,None],phase,0))
        labels.append(y_cls)
        e0.append(a)
        e1.append(b)
        residual.append(y)
        weights.append((1+3*tail)/max(1,counts[d['regime']]))
        offset+=int(m.sum())
    x,a,b,y,w,labels=map(np.concatenate,(x,e0,e1,residual,weights,labels))
    a=np.where(a>=0,a,offset)
    b=np.where(b>=0,b,offset)
    w*=len(w)/w.sum()
    deg=(np.bincount(a,weights=w,minlength=offset+1)+np.bincount(b,weights=w,minlength=offset+1))[:-1]
    h=np.maximum(1e-6,2*deg)
    def objective(pred,data):
        pad=np.append(pred,0.)
        z=(pad[b]-pad[a]+y)*w
        return (np.bincount(b,weights=z,minlength=offset+1)-np.bincount(a,weights=z,minlength=offset+1))[:-1],h
    common=dict(verbosity=-1,learning_rate=.06,num_leaves=23,min_data_in_leaf=45,lambda_l2=12,
                feature_fraction=.85,num_threads=2,deterministic=True,force_col_wise=True)
    model=lgb.train(dict(common,objective=objective,boost_from_average=False),
                    lgb.Dataset(x,label=np.zeros(len(x))),num_boost_round=55)
    risk=lgb.train(dict(common,objective='binary',scale_pos_weight=3),
                   lgb.Dataset(x,label=labels),num_boost_round=40)
    g=np.concatenate([np.diff(q,axis=0)[d['mode']&d['pairs']] for q,d in zip(shifts,ds)])
    # Learned mixture of shifting stock one tick earlier or later.
    phase=np.linalg.solve((g.T*w)@g+.5*np.diag(np.sum(g*g*w[:,None],axis=0)+1e-8),g.T@(-y*w))
    return model,risk,phase
