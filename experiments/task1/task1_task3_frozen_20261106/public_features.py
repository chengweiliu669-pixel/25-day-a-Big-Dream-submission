"""Build public masked-input context exactly as in the frozen stack."""
import numpy as np
import pandas as pd


def grid(f,ids,link):
    time=pd.to_datetime(f.timestamp,utc=True)
    assert (time.dt.minute%5==0).all()
    t=(time.dt.hour*12+time.dt.minute//5).to_numpy()
    n=f[link].map(ids)
    assert n.notna().all()
    n=n.to_numpy(dtype=int)
    assert len(np.unique(t*len(ids)+n))==len(n)
    return t,n


def interpolate(values):
    valid=np.isfinite(values)
    ticks=np.arange(len(values))[:,None]
    left=np.maximum.accumulate(np.where(valid,ticks,-1),axis=0)
    right=np.minimum.accumulate(np.where(valid,ticks,len(values))[::-1],axis=0)[::-1]
    columns=np.arange(values.shape[1])
    a=values[left.clip(0),columns]
    b=values[right.clip(max=len(values)-1),columns]
    weight=(ticks-left)/np.maximum(right-left,1)
    return np.where(left<0,b,np.where(right==len(values),a,a+(b-a)*weight))


class Corridor:
    def __init__(self,path,frozen):
        self.path=path
        self.frozen=frozen
        net=path/'network'
        links=pd.read_csv(net/'links.csv',dtype={'link_id':str}).set_index('link_id')
        topo=pd.read_csv(net/'lwr_mainline_topology.csv',dtype=str).fillna('').set_index('link_id')
        self.links=links.index.to_numpy()
        self.ids={k:i for i,k in enumerate(self.links)}
        self.length=links.length_km.to_numpy()
        self.lanes=links.lanes.to_numpy()
        self.free=links.free_speed_kmh.to_numpy()
        self.adj=[]
        self.expected=[]
        for key in ('incoming_link_ids','outgoing_link_ids'):
            neighbors=[list(filter(None,topo.loc[k,key].split(';'))) for k in self.links]
            adj=np.zeros((len(links),len(links)),dtype=np.float32)
            for i,group in enumerate(neighbors):
                for k in group:
                    if k in self.ids:
                        adj[self.ids[k],i]=1
            self.adj.append(adj)
            self.expected.append(np.asarray([len(g) for g in neighbors]))
        ramp=pd.read_csv(net/'ramp_attachment_map.csv',dtype=str)
        ramp=ramp[ramp.nearest_mainline_link_id.isin(self.ids)].copy()
        ramp['side']=ramp.ramp_type.str.upper().map({'OR':0,'ON':0,'FR':1,'OFF':1})
        assert ramp.side.notna().all()
        self.ramp_ids={k:i for i,k in enumerate(ramp.ramp_link_id)}
        self.ramp_adj=[np.zeros((len(ramp),len(links)),dtype=np.float32) for _ in range(2)]
        for i,row in enumerate(ramp.itertuples(index=False)):
            self.ramp_adj[row.side][i,self.ids[row.nearest_mainline_link_id]]=1

    def build(self,masked,ramps,date):
        f=pd.read_parquet(masked,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph',
                                         'pct_observed','is_score_eligible','mask_regime'])
        t,n=grid(f,self.ids,'link_id')
        shape=(288,len(self.ids))
        v=np.full(shape,np.nan,dtype=np.float32)
        q=v.copy()
        provenance=np.zeros(shape,dtype=np.uint8)
        good=f.is_score_eligible.astype(bool)&f.pct_observed.ge(75)&f[['speed_kmh','flow_vph']].notna().all(axis=1)
        v[t[good],n[good]]=f.speed_kmh[good]
        q[t[good],n[good]]=f.flow_vph[good]
        provenance[t[good],n[good]]=1
        # Fallback context uses public observations only. Temporal interpolation
        # uses this day's anchors and then a corridor-wide per-lane profile.
        vv,qq=interpolate(v),interpolate(q)
        for x,default in [(vv,self.free),(qq/self.lanes,0)]:
            count=np.isfinite(x).sum(axis=1)
            avg=np.divide(np.nansum(x,axis=1),count,out=np.full(288,np.nan),where=count>0)
            fill=interpolate(avg[:,None])[:,0]
            missing=~np.isfinite(x)
            x[missing]=np.broadcast_to(np.where(np.isfinite(fill),fill,np.nan)[:,None],x.shape)[missing]
            x[:]=np.where(np.isfinite(x),x,default)
            if x is not vv:
                qq=x*self.lanes
        provenance[(provenance==0)&np.isfinite(interpolate(v))]=3
        provenance[provenance==0]=4
        fp=self.frozen/f'{self.path.name}_{date}.parquet'
        if fp.is_file():
            p=pd.read_parquet(fp,columns=['timestamp','link_id','speed_kmh','flow_vph'])
            a,b=grid(p,self.ids,'link_id')
            assert (provenance[a,b]!=1).all()
            vv[a,b]=p.speed_kmh
            qq[a,b]=p.flow_vph
            provenance[a,b]=2
        assert np.isfinite(vv).all() and np.isfinite(qq).all()
        dens=qq/vv.clip(min=1)
        stock=dens*self.length
        change=np.vstack((np.diff(stock,axis=0),np.full((1,len(self.ids)),np.nan)))
        qs=[]; neighbors=[]
        for adj,expected in zip(self.adj,self.expected):
            count=adj.sum(0)
            qside=np.where(count>0,qq@adj,np.nan)
            observed_count=(provenance==1).astype('float32')@adj
            qs.append(qside)
            neighbors.extend([qside,np.broadcast_to(count,shape),np.broadcast_to(expected,shape),observed_count])

        ramp=np.full((288,len(self.ramp_ids)),np.nan,dtype=np.float32)
        if ramps.is_file():
            r=pd.read_parquet(ramps,columns=['timestamp','ramp_link_id','flow_vph','pct_observed','is_score_eligible'])
            r=r[r.ramp_link_id.isin(self.ramp_ids)]
            rt,rn=grid(r,self.ramp_ids,'ramp_link_id')
            good=r.is_score_eligible.astype(bool)&r.pct_observed.ge(75)&r.flow_vph.notna()
            ramp[rt[good],rn[good]]=r.flow_vph[good]
        ramp_est=interpolate(ramp)
        ramp_fields=[]; ramp_total=[]; ramp_complete=[]
        for adj in self.ramp_adj:
            expected=adj.sum(0)
            count=np.isfinite(ramp).astype('float32')@adj
            filled=np.isfinite(ramp_est).astype('float32')@adj
            measured_sum=np.nan_to_num(ramp)@adj
            estimated_sum=np.where(filled==expected,np.nan_to_num(ramp_est)@adj,np.nan)
            ramp_total.append(estimated_sum)
            ramp_complete.append(count==expected)
            ramp_fields.extend([np.where(count==expected,measured_sum,np.nan),estimated_sum,
                                np.broadcast_to(expected,shape),count])
        net_ramp=ramp_total[0]-ramp_total[1]
        # Net balance identifies a difference. A separate boundary is inferred
        # only when at least one side has a linked flow proxy.
        required_net=12*change-net_ramp
        incoming,outgoing=qs
        inferred_in=np.where(np.isnan(incoming),outgoing+required_net,np.nan)
        inferred_out=np.where(np.isnan(outgoing),incoming-required_net,np.nan)
        result=pd.DataFrame(dict(
            timestamp=pd.to_datetime(f.timestamp,utc=True),station_id=f.station_id,link_id=f.link_id,
            mask_regime=f.mask_regime,eligible=f.is_score_eligible.astype(bool),
            state_source=provenance[t,n],speed_kmh=vv[t,n],flow_vph=qq[t,n],
            flow_vph_lane=(qq/self.lanes)[t,n],density_vehpkm=dens[t,n],
            stock_veh=stock[t,n],delta_stock_5m=change[t,n],
            ramp_all_observed=(ramp_complete[0]&ramp_complete[1])[t,n],
            required_mainline_net_vph=required_net[t,n], inferred_in_vph=inferred_in[t,n],
            inferred_out_vph=inferred_out[t,n],
            topology_net_vph=(incoming-outgoing+net_ramp)[t,n],
            inlet_estimate_vph=np.where(np.isfinite(incoming),incoming,np.where(inferred_in>=0,inferred_in,np.nan))[t,n],
            outlet_estimate_vph=np.where(np.isfinite(outgoing),outgoing,np.where(inferred_out>=0,inferred_out,np.nan))[t,n]))
        for i,side in enumerate(('up','down')):
            for j,name in enumerate(('proxy_vph','linked_count','declared_count','observed_count')):
                result[f'{side}_{name}']=neighbors[4*i+j][t,n]
        for i,side in enumerate(('on','off')):
            for j,name in enumerate(('observed_vph','estimated_vph','attached_count','observed_count')):
                result[f'{side}_ramp_{name}']=ramp_fields[4*i+j][t,n]
        for key in result.select_dtypes(include='float64'):
            result[key]=result[key].astype('float32')
        return result
