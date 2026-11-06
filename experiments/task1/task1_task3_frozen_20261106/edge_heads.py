"""D7_I10_W frozen edge heads and reconciliation."""
import numpy as np
import lightgbm as lgb
from scipy.sparse import csr_matrix

SETTINGS = dict(objective="regression", verbosity=-1, learning_rate=.055,
                num_leaves=31, min_data_in_leaf=45, lambda_l2=6,
                feature_fraction=.85, seed=811, deterministic=True,
                force_col_wise=True, num_threads=4)


def state_grid(values):
    v, q = values[..., 0], values[..., 1]
    return np.stack((v, q, q / v.clip(.04)), axis=-1)


def shifted(values, known, lag):
    value = np.roll(values, lag, axis=0)
    valid = np.roll(known, lag, axis=0).copy()
    if lag > 0:
        valid[:lag] = False
    elif lag < 0:
        valid[lag:] = False
    return np.where(valid[..., None], value, np.nan)


def neighborhood(graph):
    matrices = [csr_matrix(x.T) for x in graph]
    return tuple((x, x @ x) for x in matrices)


def features(day, context, graph, params):
    b, x, m = day["reference"], day["x"], day["mask"]
    known = x[..., 5] > 0
    seen = x[..., 4] > 0
    state = state_grid(b)
    nv, nq = x[..., 8:10], x[..., 10:12]
    left = np.where(x[..., 31:32] > 0, state_grid(nv), np.nan)
    right = np.where(x[..., 32:33] > 0, state_grid(nq), np.nan)
    lengths = day["physical"]
    tick = np.broadcast_to(np.arange(m.shape[0])[:, None], m.shape)
    node = np.broadcast_to(np.arange(m.shape[1]), m.shape)
    _, _, kcrit, kjam, lanes = params
    z = np.stack((tick/288., node/m.shape[1], np.full_like(tick, day["regime"]),
                  np.log1p(lengths), np.broadcast_to(kcrit/lanes/24, m.shape),
                  np.broadcast_to(kjam/lanes/24, m.shape)), axis=-1)
    parts = [state, x[..., 4:24], z, context, left-state, right-state, right-left]
    for lag in (-4, -2, -1, 1, 2, 4):
        parts.append(shifted(state, known, lag)-state)
        parts.append(np.nan_to_num(shifted(seen[..., None], known, lag), nan=-1))
    for direct, extended in graph:
        for mat, active in ((direct, seen), (extended, seen), (direct, known)):
            count = (mat @ active.astype("float32").T).T
            total = np.stack([(mat @ (s * active).astype("float32").T).T for s in state.transpose(2, 0, 1)], axis=-1)
            mean = np.divide(total, count[..., None], out=np.full_like(total, np.nan),
                             where=count[..., None] > 0)
            parts.append(mean-state)
            if mat is direct and active is seen:
                parts.extend((shifted(mean, count > 0, -1)-state, shifted(mean, count > 0, 1)-state))
    dense = np.concatenate(parts, axis=-1).astype("float32")
    t, n = np.where(day["valid"] & day["pairs"])
    edge = np.column_stack((dense[t, n], dense[t+1, n], state[t+1, n]-state[t, n]))
    return dense[m], edge, t, n


def examples(day, x):
    point, edge, t, n = x
    actual = state_grid(day["y"])
    base = state_grid(day["reference"])
    innovations = actual-base
    eligible = day["mode"][t, n]
    signed = innovations[t+1, n, 2] - innovations[t, n, 2]
    lever = day["physical"][t, n]
    point_error = np.zeros(day["mask"].shape)
    error = np.where(eligible, abs(signed)*lever*24, 0.)
    np.maximum.at(point_error, (t, n), error)
    np.maximum.at(point_error, (t+1, n), error)
    return dict(point=point, y_point=innovations[day["mask"]],
                edge=edge[eligible], y_edge=signed[eligible],
                lever_edge=lever[eligible], lever_point=day["physical"][day["mask"]],
                magnitude=error[eligible], point_magnitude=point_error[day["mask"]],
                regime=day["regime"])


def fit_model(parts):
    def column(key):
        return np.concatenate([p[key] for p in parts])

    x, y = column("point"), column("y_point")
    edge, target = column("edge"), column("y_edge")
    magnitude = column("magnitude")
    point_magnitude = column("point_magnitude")
    edge_leverage = column("lever_edge")
    point_leverage = column("lever_point")
    per_regime = {r: [p for p in parts if p["regime"] == r] for r in {p["regime"] for p in parts}}
    n_point = {r: sum(len(p["point"]) for p in group) for r, group in per_regime.items()}
    n_edge = {r: sum(len(p["edge"]) for p in group) for r, group in per_regime.items()}
    w_point = np.concatenate([np.full(len(p["point"]), len(x)/n_point[p["regime"]]) for p in parts])
    w_edge = np.concatenate([np.full(len(p["edge"]), len(edge)/n_edge[p["regime"]]) for p in parts])
    cutoff = np.quantile(magnitude, .99)
    w_stock = (w_point * np.clip(point_leverage/np.median(edge_leverage), .3, 16.)
               * (1 + 2*(point_magnitude >= cutoff)))
    w_delta = (w_edge * np.clip(edge_leverage/np.median(edge_leverage), .3, 16.)
               * (1 + 2*(magnitude >= cutoff)))

    def fit(a, b, weight, rounds):
        dataset = lgb.Dataset(a, label=b, weight=weight/weight.mean())
        return lgb.train(SETTINGS, dataset, num_boost_round=rounds)

    return dict(speed=fit(x, y[:, 0], w_point, 105),
                flow=fit(x, y[:, 1], w_point, 105),
                stock=fit(x, y[:, 2], w_stock, 140),
                edge=fit(edge, target, w_delta, 145),
                risk=fit(edge, np.log1p(magnitude), w_edge, 75))


def predict_model(day, x, models):
    point, edge, t, n = x
    mask = day["mask"]
    fields = {key: np.zeros(mask.shape) for key in ("speed", "flow", "stock")}
    for key, field in fields.items():
        field[mask] = models[key].predict(point, num_threads=4)
    fields["edge"] = np.zeros(day["mode"].shape)
    fields["risk"] = np.full(day["mode"].shape, -np.inf)
    if len(edge):
        for key in ("edge", "risk"):
            fields[key][t, n] = models[key].predict(edge, num_threads=4)
    return fields


def reconcile(day, prediction, fraction, strength):
    mask = day["mask"]
    edges = day["valid"] & day["pairs"]
    active = mask.copy()
    if fraction < 1:
        risk = prediction["risk"]
        choose = edges & (risk >= np.quantile(risk[edges], 1-fraction)) if edges.any() else edges
        touch = np.pad(choose, ((1, 0), (0, 0))) | np.pad(choose, ((0, 1), (0, 0)))
        active &= touch
    prior = np.where(active, prediction["stock"], 0.)
    if strength == 0:
        return prior

    # Solve the normal equations in parallel over stations. A reliable observation
    # has a fixed zero correction, including at the entrance and exit of a gap.
    w = strength * edges * (active[:-1] | active[1:])
    off = -w * (active[:-1] & active[1:])
    diagonal = np.ones(mask.shape)
    diagonal[:-1] += w * active[:-1]
    diagonal[1:] += w * active[1:]
    rhs = prior.copy()
    rhs[:-1] -= w*prediction["edge"]*active[:-1]
    rhs[1:] += w*prediction["edge"]*active[1:]
    for t in range(1, len(diagonal)):
        factor = off[t-1] / diagonal[t-1]
        diagonal[t] -= factor * off[t-1]
        rhs[t] -= factor * rhs[t-1]
    change = rhs / diagonal
    for t in range(len(diagonal)-2, -1, -1):
        change[t] -= off[t]*change[t+1]/diagonal[t]
    assert np.all(change[~mask] == 0.)
    return change


def fd_flow(ratio, params):
    vf, capacity, critical, jam, lanes = params
    density = 24*ratio
    return np.where(density <= critical/lanes,
                    vf*density,
                    capacity/np.maximum(jam-critical, 1e-9)*np.maximum(jam/lanes-density, 0.))/600
