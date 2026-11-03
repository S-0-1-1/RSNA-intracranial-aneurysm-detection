import numpy as np

from aneurysm.labels import LABELS, N_LOC


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float('nan')
    order = np.argsort(p, kind='mergesort')
    sp = p[order]
    r = np.arange(1, len(p) + 1, dtype=np.float64)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks = np.empty(len(p))
    ranks[order] = r
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def competition_metric(y, p):
    y, p = np.asarray(y), np.asarray(p)
    aucs = np.array([auc(y[:, k], p[:, k]) for k in range(len(LABELS))])
    w = np.array([1.0] * N_LOC + [float(N_LOC)])
    ok = ~np.isnan(aucs)
    score = float((aucs[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float('nan')
    return score, aucs
