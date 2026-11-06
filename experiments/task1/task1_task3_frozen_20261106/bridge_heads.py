"""D7_I10_W frozen bridge heads."""
import numpy as np
import lightgbm as lgb
import edge_heads as edge

GROUPS = (48, 72)


def groups(day, params, t, n):
    m = day["mask"]
    obs = day["x"][..., 4] > 0
    adjacent = np.pad(obs[1:], ((0, 1), (0, 0))) | np.pad(obs[:-1], ((1, 0), (0, 0)))
    gap = np.select([day["keys"]["gap"] == "1", day["keys"]["gap"] == "2-3",
                     day["keys"]["gap"] == "4-6"], [0, 1, 2], default=3)
    slow = day["reference"][..., 0]*25 < .65*params[0]
    p = 16*day["regime"] + 4*gap + 2*adjacent + slow
    edge_type = np.where(m[t, n] & m[t+1, n], 2, np.where(m[t, n], 1, 0))
    edge_gap = np.maximum(np.where(m[t,n], gap[t,n], 0), np.where(m[t+1,n], gap[t+1,n], 0))
    e = 36*(day["physical"][t,n] >= 2.) + 12*day["regime"] + 3*edge_gap + edge_type
    return p[m].astype(int), e.astype(int)


def encode(day, context, graph, params):
    packed = edge.features(day, context, graph, params)
    _, _, t, n = packed
    gp, ge = groups(day, params, t, n)
    leverage = 24*day["physical"]
    return packed, (gp, ge), (leverage[day["mask"]], leverage[t,n])


def examples(day, context, graph, params):
    encoded = encode(day, context, graph, params)
    packed, (gp, ge), (lp, le) = encoded
    _, _, t, n = packed
    data = edge.examples(day, packed)
    use = day["mode"][t,n]
    degree = np.pad(day["mode"], ((1, 0), (0, 0))).astype(float)
    degree[:-1] += day["mode"]
    data.update(date=day["date"], pg=gp, eg=ge[use],
                pl=lp, el=le[use], degree=degree[day["mask"]],
                ratio=edge.state_grid(day["reference"])[..., 2][day["mask"]])
    return data


def fit_heads(parts):
    def col(key):
        return np.concatenate([p[key] for p in parts])

    point, y = col("point"), col("y_point")
    edge_x, z = col("edge"), col("y_edge")
    lp, le, ratio = col("pl"), col("el"), col("ratio")
    counts = {r:sum(len(p["point"]) for p in parts if p["regime"] == r) for r in range(3)}
    regimes = np.concatenate([np.full(len(p["point"]), p["regime"]) for p in parts])
    w = np.concatenate([np.full(len(p["point"]), 1/counts[p["regime"]]) for p in parts])
    # In normalized speed/flow coordinates this is the local gradient ratio
    # of the official S-state loss. Each R group has its own error scale.
    cost = np.ones(3)
    for r in range(3):
        if counts[r]:
            v_rms, q_rms = np.sqrt(np.mean(y[regimes == r, :2]**2, axis=0))
            cost[r] = np.clip((.46*v_rms)/(.54*max(1e-6, q_rms)), .4, 3.)
    c = cost[regimes]
    ray = (y[:,0] + c*ratio*y[:,1])/(1+c*ratio**2)

    def fit(x, target, weight, n):
        model = lgb.Dataset(x, label=target, weight=weight/weight.mean())
        return lgb.train(edge.SETTINGS, model, num_boost_round=n)

    # Scaling by sqrt(link exposure), with exposure as the row weight, gives
    # squared error in vehicles after converting the prediction back.
    root_p, root_e = np.sqrt(lp), np.sqrt(le)
    degree = np.maximum(.1, col("degree"))
    heads = dict(level=fit(point, y[:,2]*root_p, lp*degree, 140),
                 edge=fit(edge_x, z*root_e, le, 155),
                 risk=fit(edge_x, np.log1p(abs(z*le)), np.ones(len(z)), 85),
                 speed=fit(point, y[:,0], w, 115),
                 flow=fit(point, y[:,1], w, 115),
                 ray=fit(point, ray, w*(1+c*ratio**2), 120))
    return heads, cost


def estimate_precision(groups, errors, size):
    sse = np.bincount(groups, weights=errors**2, minlength=size)
    n = np.bincount(groups, minlength=size)
    global_var = max(1e-10, float(np.mean(errors**2)))
    posterior = (sse+150*global_var)/(n+150)
    return np.clip(global_var/posterior, .30, 3.5), global_var


def shrink(groups, prediction, target):
    denom = np.bincount(groups, weights=prediction**2, minlength=GROUPS[0])
    dot = np.bincount(groups, weights=prediction*target, minlength=GROUPS[0])
    overall = np.clip(dot.sum()/max(1e-12, denom.sum()), 0., 1.25)
    prior = max(1e-10, denom.sum()*.035)
    return np.clip((dot+overall*prior)/(denom+prior), 0., 1.25)


def fit_model(parts):
    # Each outer training set contributes a distinct date calibration slice.
    # Those dates are held out from the pilot whose errors determine trust.
    dates = sorted({p["date"] for p in parts})
    cal_days = dates[::4]
    pilot, cost = fit_heads([p for p in parts if p["date"] not in cal_days])
    calibrate = [p for p in parts if p["date"] in cal_days]
    x = np.concatenate([p["point"] for p in calibrate])
    e = np.concatenate([p["edge"] for p in calibrate])
    y = np.concatenate([p["y_point"] for p in calibrate])
    z = np.concatenate([p["y_edge"] for p in calibrate])
    pg = np.concatenate([p["pg"] for p in calibrate])
    eg = np.concatenate([p["eg"] for p in calibrate])
    rp = np.sqrt(np.concatenate([p["pl"] for p in calibrate]))
    re = np.sqrt(np.concatenate([p["el"] for p in calibrate]))
    ratio = np.concatenate([p["ratio"] for p in calibrate])
    w = cost[pg//16]
    targets = dict(speed=y[:,0], flow=y[:,1],
                   ray=(y[:,0]+w*ratio*y[:,1])/(1+w*ratio**2))
    factors = {k:shrink(pg, pilot[k].predict(x, num_threads=4), v) for k,v in targets.items()}
    pr, pv = estimate_precision(pg, pilot["level"].predict(x, num_threads=4)-y[:,2]*rp, GROUPS[0])
    er, ev = estimate_precision(eg, pilot["edge"].predict(e, num_threads=4)-z*re, GROUPS[1])
    heads, cost = fit_heads(parts)
    return heads, dict(cost=cost.tolist(), point_precision=pr.tolist(), edge_precision=er.tolist(),
                       edge_to_level_precision=float(np.clip(pv/ev,.2,2.)),
                       trust={k:v.tolist() for k,v in factors.items()}, calibration_days=cal_days)


def forecast(day, context, graph, params, models):
    heads, cal = models
    (x, e, t, n), (gp, ge), (lp, le) = encode(day, context, graph, params)
    m = day["mask"]
    out = {k:np.zeros(m.shape) for k in ("level", "speed", "flow", "ray", "precision")}
    out["level"][m] = heads["level"].predict(x, num_threads=4)*np.sqrt(lp)
    out["precision"][m] = np.array(cal["point_precision"])[gp]
    for key in ("speed","flow","ray"):
        out[key][m] = heads[key].predict(x, num_threads=4)*np.array(cal["trust"][key])[gp]
    edge_shape = day["mode"].shape
    for key in ("edge", "risk", "edge_precision"):
        out[key] = np.zeros(edge_shape)
    out["edge"][t,n] = heads["edge"].predict(e, num_threads=4)*np.sqrt(le)
    out["risk"][t,n] = heads["risk"].predict(e, num_threads=4)
    out["edge_precision"][t,n] = np.array(cal["edge_precision"])[ge]*cal["edge_to_level_precision"]
    out["cost"] = cal["cost"][day["regime"]]
    return out


def bridge(day, estimates, fraction, strength):
    m = day["mask"]
    available = day["valid"] & day["pairs"]
    risk = estimates["risk"]
    selected = (available & (risk >= np.quantile(risk[available], 1-fraction))
                if available.any() else available)
    active = m & (np.pad(selected, ((1, 0), (0, 0))) | np.pad(selected, ((0, 1), (0, 0))))
    level = np.where(active, estimates["level"], 0.)
    w = strength*estimates["edge_precision"]*available*(active[:-1] | active[1:])
    diagonal = np.where(active, estimates["precision"], 1.)
    rhs = diagonal*level
    off = -w*(active[:-1]&active[1:])
    diagonal[:-1] += w*active[:-1]
    diagonal[1:] += w*active[1:]
    rhs[:-1] -= w*estimates["edge"]*active[:-1]
    rhs[1:] += w*estimates["edge"]*active[1:]
    for t in range(1, len(diagonal)):
        factor = off[t-1]/diagonal[t-1]
        diagonal[t] -= factor*off[t-1]
        rhs[t] -= factor*rhs[t-1]
    solution = rhs/diagonal
    for t in range(len(diagonal)-2, -1, -1):
        solution[t] -= off[t]*solution[t+1]/diagonal[t]
    return np.divide(solution, 24*day["physical"], out=np.zeros_like(solution),
                     where=day["physical"]>0.)


def project(day, estimates, change, scale, mode, params):
    base = day["reference"]
    v, q, ratio = edge.state_grid(base).transpose(2,0,1)
    max_ratio = np.maximum(ratio, params[3]/params[4]/24)
    r = np.clip(ratio+scale*change, 0., max_ratio)
    m = day["mask"]
    active = m & (abs(r-ratio) > 1e-10)
    w = estimates["cost"]
    v_hat = v+estimates["speed"]
    q_hat = q+estimates["flow"]
    v_next = (v_hat+w*r*q_hat)/(1+w*r*r)
    v_next = np.where(active, v_next, v if mode=="local" else v+.6*estimates["ray"])
    edit = active if mode=="local" else (active | (m & (abs(estimates["ray"]) > 1e-10)))
    if mode=="fd":
        tolerance = abs(q-edge.fd_flow(ratio, params))
        expected = edge.fd_flow(r, params)
        den = np.maximum(r, 1e-8)
        bounded = np.clip(v_next, (expected-tolerance)/den, (expected+tolerance)/den)
        v_next = .55*v_next+.45*bounded
    v_next = v_next.clip(.04, 6.)
    out = base.copy()
    out[...,0][edit] = v_next[edit]
    out[...,1][edit] = (r*v_next)[edit]
    return out


def correction(day, estimates, arm, old):
    source, fraction, strength, scale, mode = arm
    if source == "legacy":
        return old
    pooled = bridge(day, estimates, fraction, strength)
    return .65*pooled+.35*old if source == "mixed" else pooled
