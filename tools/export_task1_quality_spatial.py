"""Export corrected official Task1 targets; no scoring and no Kaggle calls."""
import argparse,os,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1]/'experiments/task1/best_quality_spatial_094312_20261102');ap.add_argument('--parent-root',type=Path);ap.add_argument('--parent-state-cache',type=Path);ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=ap.parse_args();a.data_root=a.data_root.resolve();a.output=a.output.resolve();a.package=a.package.resolve();a.output.mkdir(parents=True,exist_ok=True);assert not a.output.is_relative_to(a.data_root)
    if a.parent_root:os.environ['TFB_FROZEN_PARENT_ROOT']=str(a.parent_root.resolve())
    spec=importlib.util.spec_from_file_location('quality_spatial_official_export',a.package/'predict_selected.py');model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model);parent=model._parent();routes=model.load_routes();torch.set_num_threads(2);started=time.time();reports=[]
    for split,start,end in [('validation','2031-03-01','2031-03-31'),('private','2031-04-01','2031-04-30')]:
        days=pd.date_range(start,end).strftime('%Y-%m-%d').tolist();dest=a.output/'submissions'/split;dest.mkdir(parents=True,exist_ok=True);signature=dict(package=str(a.package),checkpoint_sha256=model.PROFILE['checkpoint_sha256'],wrapper_sha256=sha(a.package/'predict_selected.py'),data_root=str(a.data_root),split=split,days=days,source='verified quality/spatial v3,unchanged frozen parent')
        lock=dest/'input-signature.json'
        if lock.exists():assert json.loads(lock.read_text())==signature
        else:lock.write_text(json.dumps(signature,indent=2),encoding='utf8')
        if a.parent_state_cache:
            cache=a.parent_state_cache.resolve()/split;sig=json.loads((cache/'input-signature.json').read_text());part=json.loads((cache/'part-manifest.json').read_text());assert not sig['smoke'] and part['complete'];assert sig['routes_sha256']==sha(model.PARENT/'selected-road-models.json') and sig['predict_source_sha256']==sha(model.PARENT/'predict_selected.py');assert Path(sig['data_root']).resolve()==a.data_root and sig['split']==split and sig['days']==days
        for panel,route in routes.items():
            pieces=[]
            for chunk in pd.read_csv(a.data_root/'task1'/panel/split/'sample_submission_state.csv',dtype=str,usecols=parent.base.KEYS,chunksize=250000):pieces.append(chunk)
            keys=pd.concat(pieces,ignore_index=True);daily={d:g.reset_index(drop=True) for d,g in keys.groupby(keys.timestamp.str[:10])};model.keys_for=lambda data,road,which,date:daily[date].copy();n=0
            for date in days:
                file=dest/(panel+'_'+date+'.parquet')
                if file.exists():pred=pd.read_parquet(file)
                elif a.parent_state_cache:
                    frozen=pd.read_parquet(cache/(panel+'_'+date+'.parquet'));pred=model.predict_from_parent(panel,date,route,a.data_root,dest,frozen,keys=daily[date],device=a.device)
                else:pred=model.predict(panel,date,route,a.data_root,dest,a.device)
                assert pred[parent.base.KEYS].reset_index(drop=True).equals(daily[date]);assert np.isfinite(pred[['speed_kmh','flow_vph']].to_numpy()).all() and (pred[['speed_kmh','flow_vph']]>=0).all().all();n+=len(pred)
            output=dest/(panel+'.csv');first=True
            for date in days:
                pred=pd.read_parquet(dest/(panel+'_'+date+'.parquet'));pred[parent.base.KEYS+['speed_kmh','flow_vph']].to_csv(output,mode='w' if first else 'a',header=first,index=False);first=False
            assert n==len(keys);reports.append(dict(panel=panel,split=split,rows=n,sha256=sha(output)));print(json.dumps(dict(seconds=round(time.time()-started,1),panel=panel,split=split,rows=n)),flush=True)
        (dest/'part-manifest.json').write_text(json.dumps(dict(complete=True,days=days,split=split,panels=list(routes),scope='CorrectedTask1only;notKaggleupload'),indent=2),encoding='utf8')
    assert sum(r['rows'] for r in reports)==6735795;report=dict(passed=True,rows=6735795,files=reports,checkpoint_sha256=model.PROFILE['checkpoint_sha256'],parent_cache_manifest_verified=bool(a.parent_state_cache),Kaggle_upload=False,official_total=None);(a.output/'export_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('COMPLETE',flush=True)
if __name__=='__main__':main()
