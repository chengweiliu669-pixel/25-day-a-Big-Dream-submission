"""Independent inference-only runtime. No target/label arguments or truth reads."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class Corrector(nn.Module):
    def __init__(self,features,width):
        super().__init__();self.input=nn.Conv1d(features,width,1);self.blocks=nn.ModuleList([nn.Conv1d(width,width,3,padding=d,dilation=d) for d in [1,2,4,8]]);self.output=nn.Conv1d(width,2,1)
    def forward(self,x):
        h=F.silu(self.input(x))
        for block in self.blocks:h=h+.2*F.silu(block(h))
        return .2*torch.tanh(self.output(h))
class Ensemble(nn.Module):
    def __init__(self,features,width,members):super().__init__();self.members=nn.ModuleList([Corrector(features,width) for _ in range(members)])
    def forward(self,x):return torch.stack([member(x) for member in self.members]).mean(dim=0)
def load_model(checkpoint,device='cuda'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);state=ck['state_dict'];width=state['members.0.input.weight'].shape[0];features=state['members.0.input.weight'].shape[1];assert features==ck['features'];model=Ensemble(features,width,len(ck['seeds']));model.load_state_dict(state,strict=True);return model.to(device).eval()
def inputs(e,date,keys,pred):
    values,initial,_,obs=e.feature_grid(date);ts=__import__('pandas').to_datetime(keys.timestamp,utc=True);ii=keys.station_id.map(e.si).to_numpy(int);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
    assert np.array_equal(keys.link_id.to_numpy(),np.array(e.links)[ii]);v=e.grid(obs,'speed_kmh');q=e.grid(obs,'flow_vph')/e.lanes[:,None];observed=np.isfinite(v)&np.isfinite(q);mask=np.zeros_like(v,dtype=bool);mask[ii,jj]=True;assert not observed[ii,jj].any()
    v[ii,jj]=pred.speed_kmh;q[ii,jj]=pred.flow_vph/e.lanes[ii];usable=np.isfinite(v)&np.isfinite(q);v=np.where(np.isfinite(v),v,initial[0]);q=np.where(np.isfinite(q),q,initial[1]);p=np.stack([v/25,q/600],axis=1).astype('float32')
    fields=[v/100,q/600,(q/np.maximum(v,1))/30,observed.astype(float),mask.astype(float),initial[0]/100,initial[1]/600,values['speed_kmh_prev_gap_min']/60,values['speed_kmh_next_gap_min']/60,np.broadcast_to(e.static.free_speed_kmh.to_numpy()[:,None]/100,v.shape),np.broadcast_to(e.static.length_km.to_numpy()[:,None],v.shape),np.broadcast_to(e.lanes[:,None]/5,v.shape),values['slot_sin'],values['slot_cos']]
    extra=[('pct_observed',100.),('is_imputed',1.),('weekday',6.),('speed_kmh_up',100.),('speed_kmh_down',100.),('flow_vph_up',600.),('flow_vph_down',600.),('speed_kmh_up_count',4.),('speed_kmh_down_count',4.),('relative_speed_initial',1.),('capacity_utilization_initial',1.)];assert all(k in values for k,scale in extra);fields += [values[k]/scale for k,scale in extra];x=np.stack(fields,axis=1);x=np.concatenate([np.nan_to_num(x,nan=0,posinf=0,neginf=0).clip(-20,20),np.isfinite(x)],axis=1).astype('float32');assert np.isfinite(x).all()
    return dict(x=x,p=p,mask=mask,usable=usable,ii=ii,jj=jj,obs=obs)
@torch.inference_mode()
def predict(model,input_arrays,device='cuda'):
    x=torch.from_numpy(input_arrays['x']).to(device);p=torch.from_numpy(input_arrays['p']).to(device);mask=torch.from_numpy(input_arrays['mask']).to(device);delta=model(x);out=p+delta*mask[:,None];out=torch.stack([out[:,0].clamp(min=0),out[:,1].clamp(min=0)],dim=1);assert torch.equal(out.transpose(1,2)[~mask],p.transpose(1,2)[~mask]);return out.cpu().numpy()
