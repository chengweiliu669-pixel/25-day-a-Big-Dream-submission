"""Train a fresh copy of the retained recipe, with fixed routes and January-only selection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import reproduce as repro
import bridge_heads as bridge
import edge_heads as edge
import point_heads as point
import stack
import stack_training as fitting
import training_masks
from public_features import Corridor
from release_data import Panel, KEYS, CHANNELS


HERE = Path(__file__).resolve().parent
FIT = (pd.date_range("2030-12-14","2030-12-31").strftime("%Y-%m-%d").tolist()
       + pd.date_range("2031-01-05","2031-01-20").strftime("%Y-%m-%d").tolist())
DEV = pd.date_range("2031-01-21","2031-01-31").strftime("%Y-%m-%d").tolist()
AUGMENTATION = repro.read(HERE/"training-config/severe_augmentation.json")
PARAMETERS = repro.read(HERE/"training-config/multimonth_fit.json")["params"]
PROTOCOL = repro.read(HERE/"training-config/fitting.json")
ARMS = ("control","context_only","derived","quality_gated")
ROAD = repro.D7


def original_case(panel,date,route,out,device):
    with repro.masked_inputs_only():
        return repro.base_and_features(panel,date,route,out,device)


def augmented_case(panel,date,route,out,device,condition,seed):
    masked_path = panel.exp.masked_files[date]
    raw = pd.read_parquet(masked_path)
    training_masks.SEED = seed
    auxiliary = training_masks.selector(raw,panel.road,date,condition)
    auxiliary["panel"] = panel.road
    keys = panel.keys[date]
    auxiliary["mask_regime"] = keys.mask_regime.iloc[0]
    auxiliary = auxiliary[KEYS]
    targets = pd.concat((keys,auxiliary)).sort_values(KEYS).reset_index(drop=True)
    if targets.duplicated(KEYS).any():
        raise ValueError("Additional masks overlap published targets")
    row_keys = ["timestamp","station_id","link_id"]
    hide = pd.MultiIndex.from_frame(raw[row_keys].astype(str)).isin(pd.MultiIndex.from_frame(auxiliary[row_keys]))
    if hide.sum()!=len(auxiliary):
        raise ValueError("Additional mask selection could not be aligned")
    augmented = raw.copy()
    for col in ("speed_kmh","flow_vph","occupancy","density_occ_linear_vehpkm"):
        if col in augmented:
            augmented.loc[hide,col] = np.nan

    with repro.masked_inputs_only():
        masked_reader = pd.read_parquet

        def patched(path,*a,**kw):
            if Path(path).resolve()==masked_path.resolve():
                cols = kw.get("columns",a[0] if a else None)
                return augmented.copy() if cols is None else augmented[list(cols)].copy()
            return masked_reader(path,*a,**kw)

        pd.read_parquet = patched
        base,x = repro.base_and_features(panel,date,route,out,device,targets)
    loc = pd.MultiIndex.from_frame(base[KEYS]).get_indexer(pd.MultiIndex.from_frame(auxiliary[KEYS]))
    if (loc<0).any():
        raise ValueError("Additional targets were omitted")
    return base.iloc[loc].reset_index(drop=True),x[loc]


def supervised(panel,date,case,kind):
    if date>="2031-02-01":
        raise ValueError("Fitting labels must precede February")
    base,x = case
    keys = ["timestamp","station_id","link_id"]
    truth = pd.read_parquet(panel.exp.truth_files[date],columns=keys+CHANNELS)
    truth[keys] = truth[keys].astype(str)
    matched = base[keys].merge(truth,on=keys,how="left",validate="one_to_one")
    divisor = np.column_stack((np.full(len(base),25.),600*base.link_id.map(panel.parameters.lanes)))
    y = (matched[CHANNELS].to_numpy()-base[CHANNELS].to_numpy())/divisor
    if not np.isfinite(y).all():
        raise ValueError(f"Missing training label on {panel.road} {date}")
    return x,y,np.full(len(y),f"{kind}/{date[:7]}/{base.mask_regime.iloc[0]}")


def arrays(cases):
    x,y,group = map(np.concatenate,zip(*cases))
    _,idx,n = np.unique(group,return_inverse=True,return_counts=True)
    w = np.where(np.char.startswith(group,"original/"),2.,1.)/n[idx]
    return x,y,w*len(w)/w.sum()


def fit_residual(train,dev,folder,width,augmented):
    folder.mkdir(parents=True,exist_ok=True)
    x,y,w = arrays(train)
    z,t,u = arrays(dev)
    params = dict(PARAMETERS)
    if augmented:
        params["seed"] = AUGMENTATION["model_seed"]
    fitted = {}
    for j,name in enumerate(("speed","flow_per_lane")):
        tr = lgb.Dataset(x[:,:width],label=y[:,j],weight=w)
        va = lgb.Dataset(z[:,:width],label=t[:,j],weight=u,reference=tr)
        model = lgb.train(params,tr,num_boost_round=1000,valid_sets=[va],
                          callbacks=[lgb.early_stopping(60,verbose=False)])
        model.save_model(str(folder/(name+".txt")))
        fitted[name] = model
    return fitted


def prepare_days(panel,dates,original,regressors,mixture,cache):
    cache.mkdir(parents=True,exist_ok=True)
    engine = Corridor(panel.root,cache)
    for date in dates:
        pred = repro.apply_residual(original[date],mixture,regressors)
        pred.to_parquet(cache/f"{panel.road}_{date}.parquet",index=False)
        day = panel.attach_labels(panel.day(date,pred))
        context = engine.build(panel.exp.masked_files[date],panel.ramp_path(date),date)
        yield day,context


def fit_physics(panel,days,source,folder):
    for d,context in days:
        stack.prep(d)
        stack.features(d,context,panel)
        target = d["mode"]&d["pairs"]
        d["edge_data"] = {
            arm:np.diff((24*d["physical"])[...,None]*d[arm],axis=0)[target].astype("float64")
            for arm in ARMS
        }
        d["response"] = (d["delta"]-np.diff(stack.p.stock(d["b"],d["physical"]),axis=0))[target].astype("float64")
        d["truth_rhs"] = float(abs(d["delta"][d["mode"]]).sum())
    ds = [d for d,_ in days]
    config = deepcopy(source)
    prior = config["prior"]
    prior["coef"] = fitting.fit(ds,prior["arm"],prior["penalty"]).tolist()
    stack.baseline_stage(ds,prior,np.asarray(prior["coef"]))
    fraction,kind,scale = PROTOCOL["hotspot"][panel.road]
    hot = fitting.hotspots(ds,fraction)
    shared,local = fitting.fit_hot(ds,hot)
    config["hot"] = dict(hot=hot.tolist(),shared=shared.tolist(),
                         local=local.tolist() if kind=="by_link" else None,scale=scale)
    stack.boundary_stage(ds,prior,stack.cascade_frozen(prior,config["hot"]))
    if config["boundary"]:
        rule = config["boundary"]
        coef = fitting.boundary_fit(ds,rule["arm"])
        rule["coef"] = coef.tolist()
        for d in ds:
            d["reference"] = stack.boundary_predict(d,rule["arm"],coef,rule["scale"])
    if config["focus"]:
        focus = config["focus"]
        hist = fitting.historical(ds)
        np.savez_compressed(folder/"focus-history.npz",exact=hist[0],near=hist[1],count=hist[2])
        tree = None
        if focus["arm"]!="profile":
            tree = fitting.pack_tree(ds)
            rounds = int(focus["arm"][4:])
            tree.save_model(str(folder/"focus-tree.txt"),num_iteration=rounds)
        for d in ds:
            x,change = stack.design_rmse(d,hist)
            if tree is not None:
                mask = d["mask"]
                change = np.zeros_like(change)
                change[mask] = tree.predict(x[mask],num_iteration=rounds,num_threads=2)
            d["reference"] = stack.allocate(d,change,focus["scale"],focus["alpha"])
    if config["tail"]:
        hist = fitting.historical(ds)
        tree,risk,phase = fitting.train_tail(ds,stack.adjacency(panel))
        tree.save_model(str(folder/"tail-tree.txt"))
        risk.save_model(str(folder/"risk-tree.txt"))
        np.savez_compressed(folder/"tail-history.npz",exact=hist[0],near=hist[1],count=hist[2])
        config["tail"]["phase"] = phase.tolist()
    config["artifacts"] = {p.relative_to(folder).as_posix():repro.sha(p) for p in folder.rglob("*") if p.is_file()}
    return config


def fit_stack(panel,original,labels,extra,source,weights,cache,heldout=()):
    train = [labels[d] for d in FIT if d not in heldout]
    dates = [d for d in DEV if d not in heldout]
    dev = [labels[d] for d in dates]
    folder = weights/"stack"/panel.road
    mixture = source["mixture"]
    parent = fit_residual(train,dev,folder/"parent",mixture["parent_features"],False)
    aug = None
    if mixture["gamma"]:
        aug = fit_residual(train+[case for d,case in extra["train"] if d not in heldout],
                           dev+[case for d,case in extra["development"] if d not in heldout],
                           folder/"augmented",61,True)
    models = parent,aug
    days = list(prepare_days(panel,dates,original,models,mixture,cache))
    return fit_physics(panel,days,source,folder),models


def save_heads(folder,heads):
    folder.mkdir(parents=True,exist_ok=True)
    results = {}
    for name,model in heads.items():
        path = folder/(name+".txt")
        model.save_model(str(path))
        results[path.name] = repro.sha(path)
    return results


def fit_d7(panel,source,original,labels,extra,weights,out,config):
    graph = edge.neighborhood(stack.adjacency(panel))
    records = []
    for i,held in enumerate(PROTOCOL["d7_folds"],1):
        place = out/"date-folds"/f"fold-{i}"
        frozen,models = fit_stack(panel,original,labels,extra,source,place,place/"context",held)
        stack_model = repro.FrozenStack(panel,place,frozen,None)
        for d,context in prepare_days(panel,held,original,models,source["mixture"],place/"context"):
            d["reference"] = stack_model.correct(d,context)
            x = d["context_only"][...,[17,18,19,23,24,27,28,29,30,31]]
            records.append((d,x))
    params = panel.params
    edge_model = edge.fit_model([edge.examples(d,edge.features(d,x,graph,params)) for d,x in records])
    edge_files = save_heads(weights/"edge"/ROAD,edge_model)
    config["edge"]["model"] = {Path(f).stem:dict(file=f,sha256=h) for f,h in edge_files.items()}
    heads,cal = bridge.fit_model([bridge.examples(d,x,graph,params) for d,x in records])
    config["bridge"].update(heads=save_heads(weights/"bridge"/ROAD,heads),calibration=cal)
    m = point.fit_calibration([d for d,_ in records],params)
    folder = weights/"point"/ROAD
    artifacts = save_heads(folder,{"point":m["tree"]})
    season,spread,basis = m["factors"]
    file = folder/"parameters.npz"
    np.savez_compressed(file,seasonal=season,spread=spread,basis=basis,cost=m["cost"],coefs=m["coefs"],
                        stock=m["stock"],hot=m["hot"])
    artifacts[file.name] = repro.sha(file)
    config["point"]["artifacts"] = artifacts
    return config


def fit_road(panel,route,source,weights,out,device):
    original = {d:original_case(panel,d,route,out/"base-cache"/d,device) for d in FIT+DEV}
    labels = {d:supervised(panel,d,c,"original") for d,c in original.items()}
    extra = {"train":[],"development":[]}
    if source["mixture"]["gamma"]:
        for condition in AUGMENTATION["conditions"]:
            for phase in extra:
                seed = AUGMENTATION["training_seed" if phase=="train" else "development_seed"]
                for date in AUGMENTATION["plans"][panel.road][phase]:
                    case = augmented_case(panel,date,route,out/"base-cache"/condition/date,device,condition,seed)
                    extra[phase].append((date,supervised(panel,date,case,condition)))
    cfg,_ = fit_stack(panel,original,labels,extra,source,weights,out/"context")
    return cfg,original,labels,extra


def install_base(args,out,weights):
    if args.published_base:
        manifest = repro.read(HERE/"weights.json")
        repro.verify(args.published_base,manifest)
        shutil.copytree(args.published_base/"base",weights/"base")
        routes = HERE/"base/selected-road-models.json"
    else:
        base_output = out/"base-training"
        subprocess.run([sys.executable,"-X","utf8","-u",str(HERE/"base/reproduction/pipeline.py"),
                        "train-all","--data-root",str(args.data_root),"--work-dir",str(base_output),
                        "--device",args.device],check=True)
        bundle = base_output/"trained-bundle"
        shutil.copytree(bundle/"models",weights/"base/models")
        routes = bundle/"selected-road-models.json"
    dest = out/"base-routes.json"
    shutil.copy2(routes,dest)
    return repro.models_at(weights,dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--published-base",type=Path,help="Refit downstream layers from a verified release base instead of starting base models from random initialization.")
    parser.add_argument("--road",choices=list(repro.read(HERE/"frozen.json")["roads"]),help="Diagnostic single-road fit.")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    out = args.work_dir.resolve()
    if out.is_relative_to(HERE) or out.is_relative_to(args.data_root):
        parser.error("Use a new work directory outside the source package and dataset")
    if not (args.data_root/"corridors").is_dir():
        parser.error("Extract the official public dataset first")
    out.mkdir(parents=True,exist_ok=False)
    weights = out/"weights"
    config = repro.read(HERE/"frozen.json")
    roads = [args.road] if args.road else list(config["roads"])
    routes = install_base(args,out,weights)
    torch.set_num_threads(4)
    for road in roads:
        print("Refitting",road,"through January",flush=True)
        work = out/"fit"/road
        panel = Panel(args.data_root,road,"train",FIT+DEV,work/"metadata")
        source = config["roads"][road]
        cfg,original,labels,extra = fit_road(panel,routes[road],source,weights,work,args.device)
        config["roads"][road] = cfg
        if road==ROAD:
            config["d7"] = fit_d7(panel,source,original,labels,extra,weights,work,config["d7"])
        print("Fitted",road,flush=True)
    config["roads"] = {road:config["roads"][road] for road in roads}
    if ROAD not in roads:
        config["d7"] = {}
    manifest = {p.relative_to(weights).as_posix():repro.sha(p) for p in weights.rglob("*") if p.is_file()}
    repro.save(out/"weights.json",dict(files=manifest))
    repro.save(out/"frozen.json",config)
    print(f"Frozen new training run with {len(roads)} roads at {out}",flush=True)


if __name__=="__main__":
    main()
