"""Read the released masks, build the daily state grid, and attach labels for scoring."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from feature_runtime import Experiment
from network import network_parameters
from stock_math import neighbors, stock


KEYS = ["panel", "station_id", "link_id", "timestamp", "mask_regime"]
CHANNELS = ["speed_kmh", "flow_vph"]


class Panel:
    def __init__(self, data, road, split, dates, output):
        self.road = road
        self.root = data / "corridors" / road
        self.split = split
        self.exp = Experiment(SimpleNamespace(release_root=data, panel=road, output=output, seed=20261024))
        self.n = self.exp.n
        if split != "train":
            self.exp.masked_files = {
                p.stem[-10:].replace("_", "-"): p
                for p in (self.root / split / "mainline_states_masked").rglob("*.parquet")
            }
        target_file = data / "task1" / road / split / "sample_submission_state.csv"
        targets = []
        for part in pd.read_csv(target_file, usecols=KEYS, dtype=str, chunksize=250000):
            selected = part.loc[part.timestamp.str[:10].isin(dates),KEYS]
            if len(selected):
                targets.append(selected)
        if not targets:
            raise ValueError(f"No Task1 targets for {road} and {dates}")
        targets = pd.concat(targets, ignore_index=True)
        self.keys = {d: g.reset_index(drop=True) for d, g in targets.groupby(targets.timestamp.str[:10])}
        if set(self.keys) != set(dates) or targets.duplicated(KEYS).any():
            raise ValueError(f"Incomplete or duplicate template targets for {road}")
        links = pd.read_csv(self.root / "network/links.csv", dtype={"link_id":str}).drop_duplicates("link_id")
        lengths = links.set_index("link_id").reindex(self.exp.links).length_km.to_numpy()
        physical = (lengths * self.exp.lanes).astype("float32") / self.exp.lanes
        self.physical = np.broadcast_to((physical * self.exp.lanes)[None], (288, self.n))
        self.topology = pd.read_csv(self.root/"network/lwr_mainline_topology.csv",dtype=str).fillna("").set_index("link_id")
        self.parameters = network_parameters(self.root).set_index("link_id").reindex(self.exp.links)
        self.params = [self.parameters[c].to_numpy() for c in ("free_speed_kmh","capacity_vph","critical_density","k_jam","lanes")]
        if not np.isfinite(self.params).all():
            raise ValueError(f"Invalid road parameters for {road}")
        self.attachments = pd.read_csv(self.root/"network/ramp_attachment_map.csv",dtype=str).rename(
            columns={"nearest_mainline_link_id":"link_id"})

    def ramp_path(self, date):
        return self.root/self.split/"ramp_states"/f"year_month={date[:7]}"/f"synthetic_ramp_{date.replace('-','_')}.parquet"

    def coordinates(self, f):
        stamp = pd.to_datetime(f.timestamp, utc=True)
        time = (stamp.dt.hour*12+stamp.dt.minute//5).to_numpy()
        node = f.station_id.map(self.exp.si).to_numpy(int)
        if not np.array_equal(f.link_id.to_numpy(), np.asarray(self.exp.links)[node]):
            raise ValueError("Station and link IDs do not match the published network")
        return time,node

    def mode(self, date, source, valid):
        # The published quality flags survive Task1 masking. Observed and target
        # endpoints are also intersected with the original known-state mask.
        rows = source.loc[source.is_score_eligible.astype(bool),["timestamp","link_id","station_id"]].copy()
        rows.timestamp = pd.to_datetime(rows.timestamp, utc=True)
        ids = set(rows.link_id)
        internal = set(self.topology.index)
        observable = {k for k,row in self.topology.iterrows()
                      if (set((row.incoming_link_ids+";"+row.outgoing_link_ids).split(";"))&internal)<=ids}
        rows = rows[rows.link_id.isin(observable)].drop_duplicates(["timestamp","link_id"]).sort_values(["link_id","timestamp"])
        rows = rows[(rows.groupby("link_id").timestamp.shift(-1)-rows.timestamp).eq(pd.Timedelta(minutes=5))]
        ramp = pd.read_parquet(self.ramp_path(date))
        ramp = ramp[ramp.is_score_eligible.astype(bool)&ramp.pct_observed.ge(75)&ramp.flow_vph.notna()].copy()
        ramp.timestamp = pd.to_datetime(ramp.timestamp,utc=True)
        ramp = ramp.merge(self.attachments[["ramp_link_id","link_id"]],on="ramp_link_id")
        ramp["on"] = ramp.ramp_type.str.upper().isin(["OR","ON"])
        ramp["off"] = ramp.ramp_type.str.upper().isin(["FR","OFF"])
        rows = rows.merge(ramp.groupby(["timestamp","link_id"])[["on","off"]].max().reset_index(),
                          on=["timestamp","link_id"],how="left")
        for side in ("on","off"):
            rows[side] = rows[side].eq(True)|rows.link_id.map(self.topology[side+"_ramp_link_ids"]).eq("")
        time,node = self.coordinates(rows[rows.on&rows.off])
        mode = np.zeros_like(valid)
        mode[time,node] = True
        return mode & valid

    def day(self,date,prediction):
        keys = self.keys[date]
        if not keys.equals(prediction[KEYS].reset_index(drop=True)):
            raise ValueError(f"{self.road} {date} prediction keys differ from the official template")
        if not np.isfinite(prediction[CHANNELS]).all().all() or (prediction[CHANNELS]<0).any().any():
            raise ValueError(f"{self.road} {date} has invalid state predictions")
        time,node = self.coordinates(keys)
        masked = pd.read_parquet(self.exp.masked_files[date])
        exp = self.exp
        target = np.zeros((288,self.n),dtype=bool)
        target[time,node] = True
        v = exp.grid(masked,"speed_kmh").T/25
        q = exp.grid(masked,"flow_vph").T/(600*exp.lanes[None])
        if np.isfinite(v[target]).any() or np.isfinite(q[target]).any():
            raise ValueError("A Task1 target is visible in the masked input")
        observed = (np.isfinite(v)&np.isfinite(q)&(exp.grid(masked,"pct_observed").T>=75)
                    &(exp.grid(masked,"is_score_eligible").T>0))
        values = np.where(observed[...,None],np.stack((v,q),axis=-1),0.)
        frozen = values.copy()
        frozen[time,node,0] = prediction.speed_kmh.to_numpy()/25
        frozen[time,node,1] = prediction.flow_vph.to_numpy()/(600*exp.lanes[node])
        known = target|observed
        tick = np.broadcast_to(np.arange(288)[:,None],known.shape)
        col = np.arange(self.n)
        prev = np.maximum.accumulate(np.where(observed,tick,-1),axis=0)
        nxt = np.minimum.accumulate(np.where(observed,tick,288)[::-1],axis=0)[::-1]
        before = np.where((prev>=0)[...,None],values[prev.clip(0,287),col],np.nan)
        after = np.where((nxt<288)[...,None],values[nxt.clip(0,287),col],np.nan)
        up,up_n = neighbors(values,observed,exp.neighbors["up"])
        down,down_n = neighbors(values,observed,exp.neighbors["down"])
        physical = self.physical
        cols = [np.where(known,frozen[...,k],np.nan) for k in range(2)]
        cols += [np.where(known,stock(frozen,physical)/10,np.nan),target,observed,known,
                 np.where(prev>=0,np.minimum(tick-prev,24)/24,np.nan),
                 np.where(nxt<288,np.minimum(nxt-tick,24)/24,np.nan),
                 before[...,0],before[...,1],after[...,0],after[...,1],
                 up[...,0],up[...,1],down[...,0],down[...,1],up_n/4,down_n/4,
                 np.sin(tick*(2*np.pi/288)),np.cos(tick*(2*np.pi/288)),
                 np.full_like(tick,pd.Timestamp(date).dayofweek>=5),
                 np.broadcast_to(exp.lanes[None]/5,known.shape),
                 np.broadcast_to((physical/exp.lanes),known.shape),
                 np.where(known,frozen[...,0]-before[...,0],np.nan),
                 np.where(known,frozen[...,1]-after[...,1],np.nan)]
        numeric = np.stack(cols,axis=-1)
        finite = np.isfinite(numeric)
        features = np.concatenate((np.where(finite,numeric.clip(-8,8),0),finite),axis=-1).astype("float32")
        valid = known[:-1]&known[1:]
        regime, = keys.mask_regime.unique()
        return dict(date=date,regime=int(regime[1:])-1,x=features,b=frozen,y=frozen,
                    mask=target,physical=physical,delta=np.diff(stock(frozen,physical),axis=0),
                    valid=valid,pairs=valid&(target[:-1]|target[1:]),mode=self.mode(date,masked,valid))

    def attach_labels(self, day):
        date = day["date"]
        truth = pd.read_parquet(self.exp.truth_files[date])
        time,node = self.coordinates(self.keys[date])
        speed = self.exp.grid(truth,"speed_kmh").T/25
        flow = self.exp.grid(truth,"flow_vph").T/(600*self.exp.lanes[None])
        label = day["b"].copy()
        label[time,node,0] = speed[time,node]
        label[time,node,1] = flow[time,node]
        if not np.isfinite(label).all() or not (self.exp.grid(truth,"pct_observed").T[time,node]>=75).all():
            raise ValueError(f"{self.road} {date} has invalid reference labels")
        quality = truth.dropna(subset=CHANNELS)
        mode = self.mode(date,quality,day["valid"])
        if not np.array_equal(mode,day["mode"]):
            raise ValueError(f"{self.road} {date} masked and unmasked Mode A eligibility differ")
        day["y"] = label
        day["delta"] = np.diff(stock(label,day["physical"]),axis=0)
        return day
