"""Stock arithmetic from the retained Task1 residual pilot."""
import numpy as np


def stock(x, physical):
    return 24 * x[..., 1] / x[..., 0].clip(.04) * physical


def neighbors(values, valid, groups):
    adj = np.zeros((len(groups), len(groups)), dtype='float32')
    for node, ids in enumerate(groups):
        adj[node, ids] = 1
    count = valid.astype('float32') @ adj.T
    total = np.stack([(values[..., k] * valid) @ adj.T for k in range(2)], axis=-1)
    return np.divide(total, count[..., None], out=np.full_like(total, np.nan),
                     where=count[..., None] > 0), count
