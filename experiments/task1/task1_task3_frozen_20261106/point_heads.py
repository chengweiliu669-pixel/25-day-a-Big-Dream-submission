"""D7_I10_W frozen point calibration."""
from datetime import date
import numpy as np
import lightgbm as lgb
import edge_heads as edge

GROUPS = 48


def smooth(x):
    axis = np.arange(len(x))
    prefix = np.concatenate((np.zeros_like(x[:1]), x.cumsum(axis=0)), axis=0)
    return prefix[np.minimum(axis+4, len(x))]-prefix[np.maximum(axis-3, 0)]


def fit_factors(training):
    shape = training[0]["y"].shape[:2]
    sums = np.zeros((2,*shape,3))
    count = np.zeros((2,*shape,1))
    for day in training:
        weekend = int(date.fromisoformat(day["date"]).weekday() >= 5)
        valid = day["x"][..., 5:6] > 0
        sums[weekend] += edge.state_grid(day["y"])*valid
        count[weekend] += valid
    global_sum, global_n = smooth(sums.sum(axis=0)), smooth(count.sum(axis=0))
    per_link = global_sum.sum(axis=0)/np.maximum(1.,global_n.sum(axis=0))
    base = np.divide(global_sum, np.maximum(1.,global_n))
    base = np.where(global_n > 0, base, per_link)
    profiles = np.stack([(smooth(s)+6*base)/(smooth(n)+6) for s,n in zip(sums,count)])
    residual, seen = [], []
    for day in training:
        weekend = int(date.fromisoformat(day["date"]).weekday() >= 5)
        valid = day["x"][..., 5:6] > 0
        residual.append((edge.state_grid(day["y"])-profiles[weekend])*valid)
        seen.append(np.broadcast_to(valid, (*shape,3)))
    matrix = np.concatenate(residual).reshape(-1,shape[1]*3).astype("float32")
    observed = np.concatenate(seen).reshape(matrix.shape).astype("float32")
    spread = np.sqrt(np.sum(matrix*matrix, axis=0)/np.maximum(1., observed.sum(axis=0)))
    spread = np.maximum(spread, np.tile([.012,.008,.004], shape[1]))
    standardized = (matrix/spread).astype("float32")
    denom = observed.T@observed
    cov = (standardized.T@standardized)/np.maximum(1.,denom)
    cov *= denom/(denom+150.)
    values, vectors = np.linalg.eigh(cov)
    keep = np.argsort(values)[-20:]
    basis = vectors[:,keep]*np.sqrt(np.maximum(.05, values[keep]-.35))
    return profiles, spread, basis


def anchors(day, observed_values, fade=5.):
    t = np.arange(len(observed_values))[:,None]
    n = np.arange(observed_values.shape[1])
    d0,d1 = day["dist"]
    left = np.maximum(t-np.nan_to_num(d0, nan=0).astype(int),0)
    right = np.minimum(t+np.nan_to_num(d1, nan=0).astype(int),len(t)-1)
    a = np.exp(-np.nan_to_num(d0, nan=1e6)/fade)[...,None]
    b = np.exp(-np.nan_to_num(d1, nan=1e6)/fade)[...,None]
    return a*observed_values[left,n]+b*observed_values[right,n], a+b


def factor_prediction(day, factors):
    profiles, spread, basis = factors
    weekend = int(date.fromisoformat(day["date"]).weekday() >= 5)
    expected = profiles[weekend]
    visible = day["x"][...,4] > 0
    state = edge.state_grid(day["reference"])
    delta = ((state-expected)*visible[...,None]).reshape(len(state),-1)/spread
    n = state.shape[1]
    k = basis.shape[-1]
    per_node = basis.reshape(n,3,k)
    gram = np.einsum("nck,ncl->nkl",per_node,per_node).reshape(n,k*k)
    precision = (visible.astype("float32")@gram).reshape(len(state),k,k)
    score = np.linalg.solve(precision+.9*np.eye(k), (delta@basis)[...,None])[...,0]
    predicted = expected+((score@basis.T)*spread).reshape(state.shape)
    correction, support = anchors(day, (state-predicted)*visible[...,None])
    result = predicted+correction/(support+.45)
    return np.maximum(result, [ .04, 0., 0. ])


def ratio_basis(day, alt):
    v,q,r = edge.state_grid(day["reference"]).transpose(2,0,1)
    innovations = np.stack((alt[...,2]-r, alt[...,1]/v.clip(.04)-r,
                            q/alt[...,0].clip(.04)-r),axis=-1)
    return np.clip(innovations,-1.25,1.25)*day["mask"][...,None]


def point_groups(day, base, params):
    m = day["mask"]
    gap = np.select([day["keys"]["gap"]=="1",day["keys"]["gap"]=="2-3",
                     day["keys"]["gap"]=="4-6"],[0,1,2],default=3)
    both = np.isfinite(day["dist"][0]) & np.isfinite(day["dist"][1])
    slow = base[...,0]*25 < .65*params[0]
    return (16*day["regime"]+4*gap+2*both+slow)[m]


def point_features(day, base, alt, cost, params):
    v,q,r = edge.state_grid(base).transpose(2,0,1)
    m = day["mask"]
    projected_q_weight = cost*r/(1+cost*r*r)
    projected_v_weight = 1/(1+cost*r*r)
    fd = edge.fd_flow(r,params)
    v_fd = np.divide(fd,r.clip(1e-8))
    v_fd = np.where(r > 1e-6, v_fd, params[0]/25)
    v_fd = np.where((fd <= .005)&(r >= params[2]/params[4]/24),v,v_fd)
    observed = edge.state_grid(day["reference"])
    near, weight = anchors(day,observed)
    near = near/(weight+.30)
    step = np.stack((np.full(v.shape,.02), (v_fd-v).clip(-.75,.75),
                     (alt[...,0]-v)*projected_v_weight,
                     (alt[...,1]-q)*projected_q_weight,
                     (near[...,0]-v*weight[...,0]/(weight[...,0]+.30))*projected_v_weight,
                     (near[...,1]-q*weight[...,0]/(weight[...,0]+.30))*projected_q_weight),
                    axis=-1)
    x = np.concatenate((step, alt, np.stack((v,q,r),axis=-1),
                        day["x"][...,4:24], weight),axis=-1)
    return step[m], x[m].astype("float32"), point_groups(day,base,params)


def solve(gram, rhs, shrink, prior):
    regularizer = shrink*np.diag(gram)+1e-7
    return np.linalg.solve(gram+np.diag(regularizer),rhs+regularizer*prior)


def fit_calibration(days, params):
    # Build factor predictions from January dates other than the example date.
    dates = sorted(days,key=lambda d:d["date"])
    examples = []
    for known, other in ((dates[::2], dates[1::2]),(dates[1::2],dates[::2])):
        factors = fit_factors(known)
        examples.extend((d,factor_prediction(d,factors)) for d in other)
    costs = []
    points = []
    for r in range(3):
        squared = np.concatenate([(d["y"]-d["reference"])[d["mask"]] for d in days if d["regime"]==r])
        v,q = np.sqrt(np.mean(squared*squared,axis=0))
        costs.append(float(np.clip(.46*v/(.54*max(q,1e-6)),.4,3.)))
        points.append(len(squared))
    p_parts, x_parts, g_parts, w_parts, y_parts = [],[],[],[],[]
    n = days[0]["mask"].shape[1]
    gram = np.zeros((n,3,3))
    rhs = np.zeros((n,3))
    count = np.zeros(n,dtype=int)
    for d,alt in examples:
        r = d["regime"]
        m = d["mask"]
        base = d["reference"]
        cost = costs[r]
        p,x,group = point_features(d,base,alt,cost,params)
        ratio = edge.state_grid(base)[...,2][m]
        err = (d["y"]-base)[m]
        w = (1+cost*ratio*ratio)/points[r]
        target = (err[:,0]+cost*ratio*err[:,1])/(1+cost*ratio*ratio)
        p_parts.append(p)
        x_parts.append(x)
        g_parts.append(group)
        w_parts.append(w)
        y_parts.append(target)
        eligible = d["mode"]&d["pairs"]
        gradient = np.diff(ratio_basis(d,alt),axis=0)*24*d["physical"][:-1,...,None]
        response = (d["delta"]-np.diff(edge.state_grid(base)[...,2],axis=0)*24*d["physical"][:-1])
        gradient *= eligible[...,None]
        response *= eligible
        gram += np.einsum("tnx,tny->nxy",gradient,gradient)
        rhs += np.einsum("tnx,tn->nx",gradient,response)
        count += eligible.sum(axis=0)
    p,x,g,w,y = map(np.concatenate,(p_parts,x_parts,g_parts,w_parts,y_parts))
    fit_p = solve((p.T*w)@p,p.T@(w*y),.08,np.zeros(p.shape[-1]))
    coefs = np.tile(fit_p,(GROUPS,1))
    for group in np.unique(g):
        use = g==group
        if use.sum() >= 300:
            local = p[use]
            coefs[group] = solve((local.T*w[use])@local,local.T@(w[use]*y[use]),.30,fit_p)
    tree = lgb.train(dict(edge.SETTINGS, num_leaves=25, min_data_in_leaf=75),
                     lgb.Dataset(x,label=y,weight=w/w.mean()),num_boost_round=120)
    common = solve(gram.sum(axis=0),rhs.sum(axis=0),.35,np.zeros(3))
    stock = np.tile(common,(n,1))
    for j in np.flatnonzero(count>=160):
        stock[j] = solve(gram[j],rhs[j],.55,common)
    stock = np.clip(stock,-1.5,1.5)
    explained = 2*np.sum(stock*rhs,axis=1)-np.einsum("ni,nij,nj->n",stock,gram,stock)
    order = np.argsort(-explained,kind="stable")
    n_hot = max(3,1+np.searchsorted(np.cumsum(np.maximum(0,explained[order])),
                                   .78*np.maximum(0,explained).sum()))
    hot = np.isin(np.arange(n),order[:n_hot])&(explained>0)
    return dict(factors=fit_factors(dates),cost=np.asarray(costs),coefs=coefs,tree=tree,
                stock=stock,hot=hot)


def stock_state(day, previous, alt, model, arm, params):
    kind, dose = arm
    if kind=="previous":
        return previous
    old = edge.state_grid(previous)[...,2]
    base = edge.state_grid(day["reference"])[...,2]
    adjustment = np.einsum("tnk,nk->tn",ratio_basis(day,alt),model["stock"])
    if kind=="add_hot":
        r = old+dose*adjustment*model["hot"]
    elif kind=="replace_hot":
        r = old+dose*(base+adjustment-old)*model["hot"]
    else:
        r = old+dose*(base+adjustment-old)
    r = np.clip(r,0.,np.maximum(old,params[3]/params[4]/24))
    output = previous.copy()
    m = day["mask"]
    output[...,1][m] = (r*previous[...,0].clip(.04))[m]
    return output


def shifts(day, candidate, alt, model, params):
    design,x,group = point_features(day,candidate,alt,model["cost"][day["regime"]],params)
    return {"ridge": np.einsum("nk,nk->n",design,model["coefs"][group]),
            "tree": model["tree"].predict(x,num_threads=4)}


def scale_state(day, candidate, predicted, arm, params):
    mode, scale, fd = arm
    if mode=="off":
        return candidate
    m = day["mask"]
    v,q,r = edge.state_grid(candidate).transpose(2,0,1)
    v_next = v[m]+scale*predicted[mode].clip(-.25,.25)
    if fd:
        ref = edge.fd_flow(r,params)[m]
        allowed = abs(ref-q[m])
        denominator = r[m].clip(1e-8)
        bound = np.clip(v_next,(ref-allowed)/denominator,(ref+allowed)/denominator)
        bound = np.where((r[m] > 1e-6)&(ref > .005),bound,v_next)
        v_next += fd*(bound-v_next)
    v_next = v_next.clip(.04,6.)
    out = candidate.copy()
    out[...,0][m] = v_next
    out[...,1][m] = r[m]*v_next
    assert np.allclose(edge.state_grid(out)[...,2][m],r[m],atol=1e-10,rtol=0.)
    return out
