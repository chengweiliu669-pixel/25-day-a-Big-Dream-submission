"""Deterministic additional masking used by the retained training protocol."""
import hashlib
import numpy as np
import pandas as pd

SEED = 0


def selector(raw,panel,date,condition):
 valid=raw.is_score_eligible.astype(bool)&(raw.pct_observed>=75)&raw.speed_kmh.notna()&raw.flow_vph.notna()&raw.occupancy.notna();available=raw.loc[valid,['timestamp','station_id','link_id']].copy().astype(str).sort_values(['station_id','timestamp']).reset_index(drop=True);seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}/{date}/{condition}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
 if condition=='station15':
  stations=sorted(available.station_id.unique());chosen=rng.choice(stations,max(1,int(np.ceil(.15*len(stations)))),replace=False);selected=available[available.station_id.isin(chosen)]
 elif condition=='block30':
  pieces=[]
  for station,p in available.groupby('station_id',sort=True):
   ts=pd.to_datetime(p.timestamp,utc=True);ticks=(ts.dt.hour*12+ts.dt.minute//5).to_numpy();chosen=np.zeros(len(p),bool);budget=max(1,int(.30*len(p)))
   for start in rng.permutation(265):
    chosen|=(ticks>=start)&(ticks<start+24)
    if chosen.sum()>=budget:break
   pieces.append(p.loc[chosen])
  selected=pd.concat(pieces,ignore_index=True)
 else:raise ValueError(condition)
 return selected.sort_values(['timestamp','station_id','link_id']).reset_index(drop=True)
