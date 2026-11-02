"""Drop-in frozen Task1 parent plus verified two-seed temporal correction."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
import state_quality_spatial_v3_runtime as runtime
PROFILE=json.loads((HERE/'profile.json').read_text(encoding='utf8'));PARENT=Path(os.environ.get('TFB_FROZEN_PARENT_ROOT',str((HERE/PROFILE['parent_bundle']).resolve())));_PARENT=None;_KEYS=None;_MODELS={};_ROUTES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _parent():
    global _PARENT,_KEYS
    if _PARENT is None:
        sys.path.insert(0,str(PARENT));spec=importlib.util.spec_from_file_location('frozen_selected_parent_quality_v3',PARENT/'predict_selected.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);_PARENT=m;_KEYS=m.keys_for
    return _PARENT
def load_routes():return _parent().load_routes()
def keys_for(data,panel,split,date):
    _parent();return _KEYS(data,panel,split,date)
def _corrector(device):
    if device not in _MODELS:
        path=HERE/PROFILE['checkpoint'];assert sha(path)==PROFILE['checkpoint_sha256'];_MODELS[device]=runtime.load_model(path,device)
    return _MODELS[device]
def _verify_route(panel,route):
    if panel not in _ROUTES:
        for name,p in route['paths'].items():assert sha(p)==route['sha256'][name]
        if route.get('history_sha256'):assert sha(route['history'])==route['history_sha256']
        _ROUTES[panel]=True
def predict_from_parent(panel,date,route,data,output,pred,keys=None,device='cpu'):
    parent=_parent();_verify_route(panel,route);split='train' if date<='2031-02-28' else 'validation' if date<='2031-03-31' else 'private';keys=keys if keys is not None else keys_for(data,panel,split,date);assert pred[parent.base.KEYS].reset_index(drop=True).equals(keys[parent.base.KEYS].reset_index(drop=True));reader=pd.read_parquet
    def guarded(path,*a,**kw):
        if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Inference cannot read unmasked targets')
        return reader(path,*a,**kw)
    pd.read_parquet=guarded
    try:
        e=parent.base.Experiment(SimpleNamespace(release_root=data,panel=panel,output=output/'metadata'/panel,seed=20261102));e.truth_files={}
        if split!='train':e.masked_files={p.stem[-10:].replace('_','-'):p for p in (e.panel/split/'mainline_states_masked').rglob('*.parquet')}
        h=np.load(route['paths'].get('history',route.get('history')));e.history=h['mean'] if isinstance(h,np.lib.npyio.NpzFile) else h;arrays=runtime.inputs(e,date,keys,pred);out=runtime.predict(_corrector(device),arrays,device);ii,jj=arrays['ii'],arrays['jj'];result=keys[parent.base.KEYS].copy();result['speed_kmh']=out[:,0][ii,jj]*25;result['flow_vph']=out[:,1][ii,jj]*600*e.lanes[ii];assert np.isfinite(result[['speed_kmh','flow_vph']].to_numpy()).all() and (result[['speed_kmh','flow_vph']]>=0).all().all()
    finally:pd.read_parquet=reader
    output.mkdir(parents=True,exist_ok=True);result.to_parquet(output/(panel+'_'+date+'.parquet'),index=False);return result
def predict(panel,date,route,data,output,device='cpu'):
    parent=_parent();parent.keys_for=keys_for;reader=pd.read_parquet
    def guarded(path,*a,**kw):
        if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Parent inference cannot read unmasked targets')
        return reader(path,*a,**kw)
    pd.read_parquet=guarded
    try:pred=parent.predict(panel,date,route,data,output/'_frozen_parent',device)
    finally:pd.read_parquet=reader
    return predict_from_parent(panel,date,route,data,output,pred,device=device)
