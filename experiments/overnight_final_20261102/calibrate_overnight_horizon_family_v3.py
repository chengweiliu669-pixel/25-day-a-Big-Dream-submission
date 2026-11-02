"""Fair reuse of the already established smooth January threshold family.
Includes all completed plans' October-selected candidates, never February-filtered.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np
import train_ongoing_horizon as t
B=t.BASE;OUT=B/'outputs/overnight_horizon_family_v3_20261102';ROOT=B/'outputs/overnight_20261102';START=time.time()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):t.write(OUT/name,x)
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)
def main():
    OUT.mkdir(exist_ok=False);journal=t.read(ROOT/'journal.json');plans=t.read(ROOT/'plans.json');candidates=[];source_hashes={}
    for plan in plans:
        if journal.get(plan,{}).get('status')!='complete':continue
        source=B/'outputs'/f'ongoing_night_{plan}_20261102';assert t.read(source/'acceptance.json')['passed']
        for p,h in t.read(source/'frozen_before_February.json').items():assert sha(B/p)==h,p
        source_hashes[str((source/'frozen_before_February.json').relative_to(B))]=sha(source/'frozen_before_February.json')
        for r in t.read(source/'selected_candidates.json'):
            cfg=r['config'];candidates.append(dict(id=plan+'__'+cfg['name'],source=str(source.relative_to(B)),config=cfg,October_IoU=r['October']['onset_IoU'],October_eligible=r['eligible']))
    assert candidates;recommended=max(candidates,key=lambda c:(c['October_eligible'],c['October_IoU'],c['id']))['id']
    protocol=dict(membership='Every October-selected candidate in completed predeclared plans at snapshot;no filtering by January/February scores',candidates=candidates,recommended_model_by_October_only=recommended,threshold_family='clip(a+b*(h-17.5)/25,.05,.75);exact pre-existing smooth threshold family',intercepts=np.round(np.arange(.05,.7501,.025),3).tolist(),slopes=[0.,-.05,.05,-.1,.1,-.15,.15],January='Choose threshold parameters only,per already fixed model;maxmacroIoU,tie smaller abs(slope),then lower intercept',February='Freeze all profiles and dependencies before evaluation;report every candidate,both fixed populations. No retuning,seed or road selection. Descriptive point leaders are not unbiased model selection.',onset='All early_full weights and original thresholds unchanged',Kaggle_upload=False,official_total=None)
    save('protocol.json',protocol);t.check();d,m=t.speed.query('january');selected={};grid=[]
    for c in candidates:
        pfile=B/c['source']/f"january_{c['config']['name']}_probability.npy";p=np.load(pfile);source_hashes[str(pfile.relative_to(B))]=sha(pfile);best=None
        for slope in protocol['slopes']:
            for intercept in protocol['intercepts']:
                threshold=np.clip(intercept+slope*(np.arange(5,31,5)-17.5)/25,.05,.75);score=t.old.score(p,d,m,threshold);item=dict(id=c['id'],intercept=intercept,slope=slope,thresholds=threshold.tolist(),**t.old.compact(score));grid.append(item);key=(score['onset_IoU'],-abs(slope),-intercept)
                if best is None or key>best[0]:best=(key,item,score)
        selected[c['id']]=dict(source=c,profile=best[1],January_score=best[2]);log(stage='JANUARY',**best[1])
    save('January_grid.json',grid);save('January_selection.json',selected)
    frozen={**source_hashes,**{str(p.relative_to(B)):sha(p) for p in [Path(__file__),OUT/'protocol.json',OUT/'January_selection.json']}};save('frozen_before_February.json',frozen)
    for p,h in frozen.items():assert sha(B/p)==h
    incumbent=t.read(B/'outputs/ongoing_smooth_threshold_20261101/February_results.json');results={};retained=[]
    for pop in ['new_phase_population','old410']:
        d,m=t.speed.query(pop);rows={};oldinc=incumbent[pop]
        for c in candidates:
            p=np.load(B/c['source']/f"{pop}_{c['config']['name']}_probability.npy");threshold=np.array(selected[c['id']]['profile']['thresholds']);score=t.old.score(p,d,m,threshold);overall=(score['onset_IoU']+oldinc['fixed_onset_IoU'])/2;row=dict(score=score,thresholds=threshold.tolist(),fixed_onset_IoU=oldinc['fixed_onset_IoU'],overall_IoU=overall,IoU_vs_retained_smooth=t.old.prior.REF.bootstrap(score,oldinc['score']),exceeds_retained_smooth=overall>oldinc['overall_IoU'],is_preselected_recommendation=c['id']==recommended,source=c);rows[c['id']]=row;log(stage='FEBRUARY',population=pop,id=c['id'],overall=overall,**t.old.compact(score))
            if row['exceeds_retained_smooth']:retained.append(dict(population=pop,id=c['id'],overall_IoU=overall,profile=selected[c['id']]['profile'],source=c))
        results[pop]=dict(incumbent=oldinc,candidates=rows)
    save('February_results.json',results);save('retained_candidates.json',retained)
    for p,h in frozen.items():assert sha(B/p)==h
    t.check();save('verification.json',dict(passed=True,all_sources_unchanged=True,all_threshold_profiles_frozen_before_February=True,January_thresholds_only=True,February_parameter_selection=False,onset_unchanged=True,model_recommendation_chosen_on_October=recommended,independent_raw_validation_still_required=True,Kaggle_upload=False));save('manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'});log(stage='COMPLETE',retained=len(retained),recommended=recommended)
if __name__=='__main__':main()
