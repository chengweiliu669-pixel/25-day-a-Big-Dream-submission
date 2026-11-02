"""Export actual official templates for frozen R127 and both horizon thresholds.

Inference consumes released histories and masked T only. Never reports hidden
IoU. The two horizon variants share every probability and every model weight.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parents[1];REPO=B;EX=REPO/'experiments/task2';HORIZON=EX/'best_overall_082723_horizon_20261031';SMOOTH=EX/'best_overall_082815_smooth_20261101';START=time.time()
sys.path.insert(0,str(HORIZON/'code'))
from hybrid_horizon import Hybrid,verify_manifest
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',choices=['validation','private','both'],default='both');a=ap.parse_args();a.data_root=a.data_root.resolve();a.package=a.package.resolve();a.output=a.output.resolve();a.output.mkdir(exist_ok=True,parents=True)
    assert not a.output.is_relative_to(a.data_root);verify_manifest(HORIZON);model=Hybrid(HORIZON)
    sys.path.insert(0,str(EX/'r127_reproduction_20261027'));from repro_common import setup
    frozen,_,_,audit=setup(a.package,a.data_root,a.output/'backend',True,extra_allowed=(HORIZON,SMOOTH,EX/'best_overall_082639_jaccard_20261031',EX/'best_overall_081796_20261030',a.output));model.attach_parent(frozen)
    from task2q.data import PANELS,released_sample
    from task2q.submission import validate_queue
    smooth=np.array(json.loads((SMOOTH/'profile.json').read_text())['ongoing_threshold']);reports={};checks=[];input_hashes={};files=[]
    for split in ['validation','private'] if a.split=='both' else [a.split]:
        outputs={name:[] for name in ['horizon082723','smooth082815','R127']};differences=[]
        for panel in PANELS:
            directory=a.data_root/'task2'/panel/split;index=pd.read_csv(directory/'window_index.csv');template=pd.read_csv(directory/'sample_submission_queue.csv');parts={name:[] for name in outputs};seen=set()
            for name in ['window_index.csv','sample_submission_queue.csv','window_history.parquet']:input_hashes[str((directory/name).relative_to(a.data_root))]=sha(directory/name)
            for row in index.itertuples():
                original_reader=pd.read_parquet
                def guarded(p,*args,**kw):
                    if '/mainline_states/' in str(p).replace('\\','/'):raise RuntimeError('Official inference must not read unmasked state')
                    return original_reader(p,*args,**kw)
                pd.read_parquet=guarded
                try:
                    s=released_sample(a.data_root,panel,split,row.window_id);p=model.probability(s);q=model.decode(s,p);qs=p>=smooth[:,None] if s['condition']==1 else q.copy();pr=frozen.probability(s);qr=pr>=np.array([.25 if s['condition']==1 else .5])[:,None]
                    if s['condition'] not in seen:
                        other=copy.deepcopy(s);other['y'][:]=1;other['m'][:]=True;other['speed_y'][:]=999;other['has_future_queue']=True;np.testing.assert_array_equal(p,model.probability(other));seen.add(s['condition']);checks.append(dict(panel=panel,split=split,condition=int(s['condition']),future_target_mutation_does_not_change_predictions=True))
                finally:pd.read_parquet=original_reader
                keys=template[template.window_id==row.window_id].copy();lookup={str(l):i for i,l in enumerate(s['network']['link_ids'])};delta=(pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300;assert np.all(delta==np.round(delta));h=delta.to_numpy(int)-1;assert (h>=0).all() and (h<6).all();links=keys.link_id.astype(str).map(lookup);assert links.notna().all();node=links.to_numpy(int)
                for name,binary in [('horizon082723',q),('smooth082815',qs),('R127',qr)]:
                    f=keys.copy();f['queue_pred']=binary[h,node].astype(int);parts[name].append(f)
                differences.append(dict(panel=panel,window_id=row.window_id,condition=int(s['condition']),horizon_R127_disagreements=int((q!=qr).sum()),smooth_horizon_disagreements=int((qs!=q).sum()),horizon_positive=int(q.sum()),smooth_positive=int(qs.sum()),R127_positive=int(qr.sum())))
            for name in outputs:
                frame=pd.concat(parts[name],ignore_index=True);validate_queue(frame,template);frame['panel']=panel;outputs[name].append(frame)
            print(json.dumps(dict(seconds=round(time.time()-START,1),stage='PANEL',split=split,panel=panel,windows=len(index))),flush=True)
        for name,frames in outputs.items():
            result=pd.concat(frames,ignore_index=True);assert len(result)==87000;path=a.output/f'{name}_{split}_queue.csv';result.to_csv(path,index=False);files.append(dict(file=path.name,rows=len(result),sha256=sha(path),model=name,split=split))
        reports[split]=dict(windows=len(differences),rows=87000,comparison=differences)
    save(a.output/'comparison.json',reports);save(a.output/'input_hashes.json',input_hashes);save(a.output/'verification.json',dict(passed=True,files=files,official_S_queue=None,official_total=None,models='All frozen; no retraining or threshold tuning',horizon_package=HORIZON.name,smooth_package=SMOOTH.name,causality_checks=checks,unmasked_state_reads_blocked=True,template_horizon_and_node_mapping_verified=True,R127_archive_native_model_guard=True,native_files=sorted(set(audit['native_models'])),seconds=time.time()-START));print('COMPLETE',flush=True)
if __name__=='__main__':main()
