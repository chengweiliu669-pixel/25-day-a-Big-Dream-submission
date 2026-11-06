"""Reproduce the accepted Task1/Task3 version from public inputs and frozen weights."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.request import Request, urlopen
from zipfile import ZipFile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "base"))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import bridge_heads as bridge
import edge_heads as edge
import point_heads as point
import predict_selected as predictor
import stack
from context_features import extra_features
from public_features import Corridor
from release_data import Panel, KEYS, CHANNELS
from residual_features import build_features


D7 = "D7_I10_W"
METRICS = ("state_score","fd","normalized_error_macro","rmse_macro","rmse_pooled")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def save(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def verify(weights, manifest):
    for name, digest in manifest["files"].items():
        file = weights / name
        if not file.is_file() or sha(file)!=digest:
            raise ValueError(f"Missing or changed weight: {file}")
    return len(manifest["files"])


def fetch(weights, manifest, archive=None):
    if archive is None:
        archive = weights.parent / manifest["asset"]
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.is_file():
            url = Request(manifest["url"],headers={"User-Agent":"trafficflowbench-reproduction"})
            temp = archive.with_suffix(".download")
            with urlopen(url,timeout=120) as source, temp.open("wb") as dest:
                shutil.copyfileobj(source,dest)
            temp.replace(archive)
    if sha(archive)!=manifest["sha256"]:
        raise ValueError(f"Archive SHA256 does not match: {archive}")
    with ZipFile(archive) as packed:
        names = packed.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(manifest["files"]):
            raise ValueError("Unexpected paths in weights archive")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe archive path: {name}")
            dest = weights / path
            if not dest.resolve().is_relative_to(weights.resolve()):
                raise ValueError(f"Weight output escapes the destination: {name}")
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp = dest.with_suffix(dest.suffix+".tmp")
            with packed.open(name) as source,temp.open("wb") as out:
                shutil.copyfileobj(source,out)
            if sha(temp)!=manifest["files"][name]:
                raise ValueError(f"Wrong weight digest: {name}")
            temp.replace(dest)
    return verify(weights,manifest)


@contextmanager
def masked_inputs_only():
    original = pd.read_parquet

    def guarded(path,*args,**kwargs):
        if "mainline_states" in str(path).replace("\\","/").split("/"):
            raise RuntimeError("An unmasked label was requested during prediction")
        return original(path,*args,**kwargs)

    pd.read_parquet = guarded
    try:
        yield
    finally:
        pd.read_parquet = original


def models_at(weights, routes_file=None):
    routes = read(routes_file or HERE/"base/selected-road-models.json")["routes"]
    for route in routes.values():
        route["paths"] = {k:str(weights/"base"/p) for k,p in route["paths"].items()}
        if "history" in route:
            route["history"] = str(weights/"base"/route["history"])
    return routes


def load_heads(folder, names):
    return {Path(name).stem:lgb.Booster(model_file=str(folder/name)) for name in names}


def load_d7(weights, configs):
    previous,mid,latest = (configs[k] for k in ("edge","bridge","point"))
    if previous["method"]!="edge" or latest["method"]!="conditional":
        raise ValueError("Unexpected D7_I10_W release method")
    old_heads = load_heads(weights/"edge"/D7,[a["file"] for a in previous["model"].values()])
    middle_heads = load_heads(weights/"bridge"/D7,mid["heads"])
    point_dir = weights/"point"/D7
    with np.load(point_dir/"parameters.npz",allow_pickle=False) as f:
        calibration = dict(f)
    calibration["factors"] = tuple(calibration.pop(k) for k in ("seasonal","spread","basis"))
    calibration["tree"] = lgb.Booster(model_file=str(point_dir/"point.txt"))
    return old_heads,previous["january"]["arm"],(middle_heads,mid["calibration"]),mid["selected"]["arm"],calibration,latest["january"]["arm"]


def apply_d7(day, graph, params, loaded):
    heads,legacy_arm,mid,arm,model,point_arm = loaded
    context = day["context_only"][...,[17,18,19,23,24,27,28,29,30,31]]
    estimates = bridge.forecast(day,context,graph,params,mid)
    fields = edge.predict_model(day,edge.features(day,context,graph,params),heads)
    old = edge.reconcile(day,fields,*legacy_arm[:2])
    change = bridge.correction(day,estimates,arm,old)
    parent = bridge.project(day,estimates,change,arm[3],arm[4],params)
    alt = point.factor_prediction(day,model["factors"])
    state = point.stock_state(day,parent,alt,model,point_arm[0],params)
    return point.scale_state(day,state,point.shifts(day,state,alt,model,params),point_arm[1],params)


def base_and_features(panel,date,route,cache,device,keys=None):
    keys_for = predictor.keys_for
    targets = panel.keys[date] if keys is None else keys
    predictor.keys_for = lambda *_: targets.copy()
    try:
        base = predictor.predict(panel.road,date,route,panel.exp.root,cache,device)
    finally:
        predictor.keys_for = keys_for
    base["lanes"] = base.link_id.map(panel.parameters.lanes)
    x = np.column_stack((build_features(panel.exp.root,panel.road,date,base,panel.split),
                         extra_features(panel.exp.root,panel.road,date,base,panel.split))).astype("float32")
    if x.shape != (len(base),61):
        raise ValueError("Residual feature shape differs from the frozen training contract")
    return base,x


def apply_residual(case,config,models):
    base,x = case
    original = base[CHANNELS].to_numpy()
    divisor = np.column_stack((np.full(len(base),25.),600*base.lanes.to_numpy()))
    parent,aug = models
    width = config["parent_features"]
    first = np.maximum(original+config["parent_alpha"]*np.column_stack(
        [parent[n].predict(x[:,:width],num_threads=4) for n in ("speed","flow_per_lane")])*divisor,0)
    if config["gamma"]:
        second = np.maximum(original+np.column_stack(
            [aug[n].predict(x,num_threads=4) for n in ("speed","flow_per_lane")])*divisor,0)
        first = first+config["gamma"]*(second-first)
    frame = base[KEYS].copy()
    frame[CHANNELS] = first
    return frame


class FrozenStack:
    def __init__(self,panel,weights,config,d7):
        self.panel = panel
        self.config = config
        folder = weights/"stack"/panel.road
        self.regressors = (load_heads(folder/"parent",("speed.txt","flow_per_lane.txt")),
                          load_heads(folder/"augmented",("speed.txt","flow_per_lane.txt"))
                          if config["mixture"]["gamma"] else None)
        self.graph = stack.adjacency(panel)
        self.d7 = load_d7(weights,d7) if panel.road==D7 and d7 is not None else None
        self.focus = self.tail = None
        if config["focus"]:
            with np.load(folder/"focus-history.npz",allow_pickle=False) as f:
                profile = tuple(f[k] for k in ("exact","near","count"))
            tree = (lgb.Booster(model_file=str(folder/"focus-tree.txt"))
                    if config["focus"]["arm"]!="profile" else None)
            self.focus = profile,tree
        if config["tail"]:
            with np.load(folder/"tail-history.npz",allow_pickle=False) as f:
                profile = tuple(f[k] for k in ("exact","near","count"))
            fit = (lgb.Booster(model_file=str(folder/"tail-tree.txt")),
                   lgb.Booster(model_file=str(folder/"risk-tree.txt")),np.asarray(config["tail"]["phase"]))
            self.tail = profile,fit
        if self.d7:
            self.graph_d7 = edge.neighborhood(self.graph)

    def correct(self, day, context):
        c = self.config
        stack.prep(day)
        stack.features(day,context,self.panel)
        stack.boundary_stage([day],c["prior"],stack.cascade_frozen(c["prior"],c["hot"]))
        if c["boundary"]:
            b = c["boundary"]
            day["reference"] = stack.boundary_predict(day,b["arm"],np.asarray(b["coef"]),b["scale"])
        if self.focus:
            profile,tree = self.focus
            x,change = stack.design_rmse(day,profile)
            if tree is not None:
                m = day["mask"]
                change = np.zeros_like(change)
                change[m] = tree.predict(x[m],num_threads=2)
            f = c["focus"]
            day["reference"] = stack.allocate(day,change,f["scale"],f["alpha"])
        if self.tail:
            profile,fit = self.tail
            arm,budget,scale,alpha = c["tail"]["selected"]
            risk,delta = stack.signals(day,profile,self.graph,fit)
            selected = stack.masks_by_budget(day,risk)[budget]
            day["reference"] = stack.allocate(day,np.where(selected,delta[arm],0),scale,alpha)
        if self.d7:
            return apply_d7(day,self.graph_d7,self.panel.params,self.d7)
        return day["reference"]


def run_predictions(data,weights,out,config,dates,split,roads,device,routes_file=None):
    if device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    torch.set_num_threads(4)
    predictor.sha = lru_cache(maxsize=None)(sha)
    routes = models_at(weights,routes_file)
    for road in roads:
        dest = out/"predictions"/road
        dest.mkdir(parents=True,exist_ok=True)
        with masked_inputs_only():
            panel = Panel(data,road,split,dates,out/"metadata"/road)
            stack_model = FrozenStack(panel,weights,config["roads"][road],config["d7"])
            cache = out/"_base"/road
            residual_dir = out/"_residual"/road
            residual_dir.mkdir(parents=True,exist_ok=True)
            engine = Corridor(panel.root,residual_dir)
            for date in dates:
                artifact = dest/f"{date}.parquet"
                if artifact.exists():
                    raise FileExistsError(f"Choose a new output directory; already exists: {artifact}")
                case = base_and_features(panel,date,routes[road],cache,device)
                pred = apply_residual(case,stack_model.config["mixture"],stack_model.regressors)
                pred.to_parquet(residual_dir/f"{road}_{date}.parquet",index=False)
                day = panel.day(date,pred)
                context = engine.build(panel.exp.masked_files[date],panel.ramp_path(date),date)
                values = stack_model.correct(day,context)
                if not np.array_equal(values[~day["mask"]],day["b"][~day["mask"]]):
                    raise ValueError(f"The frozen stack changed a non-target cell on {road} {date}")
                keys = panel.keys[date].copy()
                time,node = panel.coordinates(keys)
                keys["speed_kmh"] = values[time,node,0]*25
                keys["flow_vph"] = values[time,node,1]*600*panel.exp.lanes[node]
                if not np.isfinite(keys[CHANNELS]).all().all() or (keys[CHANNELS]<0).any().any():
                    raise ValueError(f"Invalid output on {road} {date}")
                keys.to_parquet(artifact,index=False)
            print(f"Predicted {road}: {len(dates)} days, {sum(len(x) for x in panel.keys.values())} targets",flush=True)


def evaluate(data,out,config,roads):
    expected = read(HERE/"expected-february.json")
    data_manifest = read(HERE/"february-inputs.json")
    results, differences = {}, {}
    for road in roads:
        for name,expected_sha in data_manifest[road].items():
            if sha(data/name)!=expected_sha:
                raise ValueError(f"February input fingerprint differs: {name}")
        panel = Panel(data,road,"train",config["evaluation_dates"],out/"metadata"/road)
        rows = []
        for date in config["evaluation_dates"]:
            pred = pd.read_parquet(out/"predictions"/road/f"{date}.parquet")
            day = panel.attach_labels(panel.day(date,pred))
            rows.append(stack.scored(day,day["b"],panel.params))
        results[road] = stack.scored_summary(rows)
        differences[road] = {k:results[road][k]-expected[road][k] for k in METRICS}
        print(f"Scored {road}: state={results[road]['state_score']:.9f} "
              f"pooled_rmse={results[road]['rmse_pooled']:.9f}",flush=True)
    mean = {k:float(np.mean([r[k] for r in results.values()])) for k in METRICS}
    mean["lwr_proxy"] = 1-mean["normalized_error_macro"]
    mean["task3_proxy"] = (mean["fd"]+2*mean["lwr_proxy"])/3
    save(out/"february-metrics.json",dict(roads=results,mean=mean,difference_from_reference=differences))
    print(json.dumps(mean),flush=True)
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fetch","verify","predict","evaluate","reproduce"))
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--weights-dir",type=Path,default=HERE/"weights")
    parser.add_argument("--trained-run",type=Path,help="Use a fresh bundle produced by train_frozen.py.")
    parser.add_argument("--archive",type=Path,help="Use a local, SHA256-verified copy of the release archive.")
    parser.add_argument("--output",type=Path,default=Path("runs/task1-task3-frozen"))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--split",choices=("february","validation","private"),default="february")
    parser.add_argument("--road",action="append")
    parser.add_argument("--date",action="append",help="Only for a partial prediction check.")
    args = parser.parse_args()
    trained = args.trained_run
    if trained and args.stage=="fetch":
        parser.error("A trained run already contains its weights")
    manifest = read((trained or HERE)/"weights.json")
    if trained:
        args.weights_dir = trained/"weights"
    if args.stage=="fetch":
        print(f"Verified {fetch(args.weights_dir,manifest,args.archive)} weight files")
        return
    if args.stage!="evaluate":
        print(f"Verified {verify(args.weights_dir,manifest)} weight files")
    if args.stage=="verify":
        return
    if args.data_root is None or not (args.data_root/"corridors").is_dir():
        parser.error("--data-root must be the extracted Kaggle public data root")
    if args.stage in ("reproduce","evaluate") and (args.split!="february" or args.date):
        parser.error("February scoring uses the complete fixed date range")
    config = read((trained or HERE)/"frozen.json")
    roads = args.road or list(config["roads"])
    if len(set(roads))!=len(roads) or not set(roads)<=set(config["roads"]):
        parser.error("Select distinct roads present in frozen.json")
    args.output.mkdir(parents=True,exist_ok=True)
    split = "train" if args.split=="february" else args.split
    dates = (config["evaluation_dates"] if split=="train" else
             pd.date_range("2031-03-01","2031-03-31").strftime("%Y-%m-%d").tolist() if split=="validation" else
             pd.date_range("2031-04-01","2031-04-30").strftime("%Y-%m-%d").tolist())
    if args.date and not set(args.date)<=set(dates):
        parser.error("--date must belong to --split")
    if args.stage in ("predict","reproduce"):
        routes = trained/"base-routes.json" if trained else None
        run_predictions(args.data_root,args.weights_dir,args.output,config,args.date or dates,split,roads,args.device,routes)
    if args.stage in ("evaluate","reproduce"):
        difference = evaluate(args.data_root,args.output,config,roads)
        # An arbitrary device may differ in the last bits; report every metric delta.
        tolerance = {"state_score":2e-6,"fd":2e-6,"normalized_error_macro":2e-5,
                     "rmse_macro":5e-5,"rmse_pooled":5e-5}
        if trained:
            print("Recomputed metrics for the newly trained weights",flush=True)
            return
        if any(abs(v)>tolerance[k] for row in difference.values() for k,v in row.items()):
            raise SystemExit("February reproduction exceeded the published tolerance; see february-metrics.json")
        print("February reference verified",flush=True)


if __name__ == "__main__":
    main()
