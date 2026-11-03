import math

import numpy as np
import torch
import torch.nn.functional as F

from aneurysm.labels import FLIP_LR, LOCATIONS, MODALITIES, N_LOC, N_TERR, TERR_FLIP
from aneurysm.storage import read_volume


def brain_values(img, dist, min_count=500):
    v = img[(dist == 0) & (img > 0)]
    return v if v.size >= min_count else img[img > 0]


def strip_and_stats(img, dist, r_mm, dist_unit_mm, min_count=500):
    lim = int(math.floor(r_mm / dist_unit_mm + 1e-6))
    out = img * (dist <= lim)
    v = brain_values(img[::4, ::4, ::4], dist[::4, ::4, ::4], min_count)
    if v.size < min_count:
        v = brain_values(img, dist, min_count)
    if v.size < min_count:
        raise ValueError(f'stored volume has only {v.size} non-zero voxels')
    v = v.astype(np.float32)
    return out, float(v.mean()), float(max(v.std(), 1.0))


def iterative_stratification(Y, n_folds, seed):
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=np.int64)
    n = Y.shape[0]
    desired_label = np.outer(np.full(n_folds, 1.0 / n_folds), Y.sum(0))
    desired_total = np.full(n_folds, n / n_folds)
    fold = np.full(n, -1)
    remaining = np.ones(n, bool)
    norm = 1.0 / np.maximum(Y.sum(0), 1)
    while remaining.any():
        counts = Y[remaining].sum(0)
        if counts.max() == 0:
            for i in rng.permutation(np.flatnonzero(remaining)):
                f = int(np.argmax(desired_total + rng.random(n_folds) * 1e-9))
                fold[i] = f
                desired_total[f] -= 1
            break
        lab = int(np.argmin(np.where(counts > 0, counts, np.iinfo(np.int64).max)))
        for i in rng.permutation(np.flatnonzero(remaining & (Y[:, lab] > 0))):
            cand = np.flatnonzero(desired_label[:, lab] == desired_label[:, lab].max())
            if len(cand) > 1:
                need = desired_label[cand] @ (Y[i] * norm)
                cand = cand[need >= need.max() - 1e-12]
            if len(cand) > 1:
                cand = cand[desired_total[cand] == desired_total[cand].max()]
            f = int(rng.choice(cand))
            fold[i] = f
            remaining[i] = False
            desired_label[f] -= Y[i]
            desired_total[f] -= 1
    return fold


def make_folds(df, n_folds, seed, groups=None):
    loc = df[LOCATIONS].to_numpy(np.int64)
    ap = df[['Aneurysm Present']].to_numpy(np.int64)
    mod = np.stack([(df['Modality'] == m).to_numpy(np.int64) for m in MODALITIES], 1)
    Y = np.concatenate([loc, ap, mod, mod * ap], 1)
    if groups is None:
        return iterative_stratification(Y, n_folds, seed)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    Yg = np.zeros((len(uniq), Y.shape[1]), np.int64)
    np.maximum.at(Yg, inv, Y)
    return iterative_stratification(Yg, n_folds, seed)[inv]


def rot_matrix(a):
    cx, sx, cy, sy, cz, sz = math.cos(a[0]), math.sin(a[0]), math.cos(a[1]), math.sin(a[1]), math.cos(a[2]), math.sin(a[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Tables:
    def __init__(self, series, points, cfg):
        self.uid = series['uid'].to_numpy()
        self.path = series['path'].to_numpy()
        self.mod = series['Modality'].map({m: i for i, m in enumerate(MODALITIES)}).to_numpy()
        if np.isnan(self.mod.astype(float)).any():
            raise ValueError(f'unknown modality in {sorted(set(series["Modality"]) - set(MODALITIES))}')
        self.mod = self.mod.astype(np.int64)
        self.spacing = series['spacing_mm'].to_numpy(np.float64)
        self.terr = series['terr_ok'].to_numpy(bool)
        self.gap = series['gap_median'].fillna(0.0).to_numpy(np.float64)
        self.n = len(series)
        pos = {u: i for i, u in enumerate(self.uid)}
        pts = points[points['uid'].isin(pos)].reset_index(drop=True)
        self.pt_series = pts['uid'].map(pos).to_numpy(np.int64)
        self.pt_xyz = pts[['a0', 'a1', 'a2']].to_numpy(np.float64)
        self.pt_loc = pts['loc_idx'].to_numpy(np.int64)
        self.pt_dist = pts['dist_mm'].fillna(1e9).to_numpy(np.float64)
        self.pt_valid = (pts['inside_cube'].to_numpy(bool) & pts['in_fov'].to_numpy(bool)
                         & (self.pt_dist <= cfg['keep_mm'] - 1.0))
        self.series_pts = [[] for _ in range(self.n)]
        for j in np.flatnonzero(self.pt_valid):
            self.series_pts[self.pt_series[j]].append(j)
        self.series_pts = [np.array(v, np.int64) for v in self.series_pts]
        valid = np.flatnonzero(self.pt_valid)
        pair = np.array([self.pt_loc[valid] == k for k in range(N_LOC)]).sum(1).astype(np.float64)
        pooled = pair + pair[FLIP_LR] if cfg['p_flip'] > 0 else pair
        mod_pos = np.bincount(self.mod[self.pt_series[valid]], minlength=len(MODALITIES)).astype(np.float64)
        w = np.zeros(len(self.pt_loc))
        if len(valid):
            w[valid] = (np.maximum(pooled[self.pt_loc[valid]], 1.0) ** -cfg['loc_power']
                        * np.maximum(mod_pos[self.mod[self.pt_series[valid]]], 1.0) ** -cfg['mod_power'])
        self.pos_idx = valid
        self.pos_p = w[valid] / w[valid].sum() if len(valid) else np.zeros(0)
        mod_all = np.bincount(self.mod, minlength=len(MODALITIES)).astype(np.float64)
        ws = np.maximum(mod_all[self.mod], 1.0) ** -cfg['mod_power']
        self.ser_p = ws / ws.sum()
        self.neg_xyz = self.pt_xyz[valid] if len(valid) else np.full((1, 3), (cfg['cube'] - 1) / 2.0)


def draw_sample(rng, T, cfg):
    n, P = cfg['cube'], cfg['patch']
    m = (n - 1) / 2.0
    positive = rng.random() < cfg['pos_frac']
    if positive and len(T.pos_idx):
        j = int(T.pos_idx[rng.choice(len(T.pos_p), p=T.pos_p)])
        si = int(T.pt_series[j])
    else:
        j = -1
        si = int(rng.choice(T.n, p=T.ser_p))
    flip = bool(rng.random() < cfg['p_flip'])
    ang = np.deg2rad(rng.uniform(-cfg['rot_deg'], cfg['rot_deg'], 3)) if rng.random() < cfg['p_rot'] else np.zeros(3)
    s = math.exp(rng.uniform(math.log(cfg['scale'][0]), math.log(cfg['scale'][1]))) if rng.random() < cfg['p_scale'] else 1.0
    M = s * rot_matrix(ang) @ np.diag([-1.0 if flip else 1.0, 1.0, 1.0])
    Minv = np.linalg.inv(M)
    pts = T.series_pts[si]
    yv = m + (T.pt_xyz[pts] - m) @ Minv.T
    locs = T.pt_loc[pts]
    if flip:
        locs = np.asarray(FLIP_LR)[locs]
    core = P / 2 - cfg['border'] - 2.0
    if j >= 0:
        yj = m + (T.pt_xyz[j] - m) @ Minv.T
        jit = min(cfg['pos_jitter'], core - 2.0)
        c = yj + rng.uniform(-jit, jit, 3)
    elif rng.random() < cfg['hard_neg']:
        k = int(rng.integers(len(T.neg_xyz)))
        c = m + (T.neg_xyz[k] - m) @ Minv.T + rng.normal(0.0, cfg['neg_sigma'], 3)
    else:
        c = m + rng.uniform(-cfg['neg_box'], cfg['neg_box'], 3)
    c = np.clip(c, P / 2 - cfg['edge'], n - P / 2 + cfg['edge'])
    if j >= 0:
        c = np.clip(c, yj - core, yj + core)
    origin = np.round(c - P / 2).astype(np.int64)
    r = rng.uniform(*cfg['strip_mm'])
    if len(pts):
        r = max(r, float(T.pt_dist[pts].max()) + 2.0)
    r = min(r, cfg['keep_mm'])
    ct = T.mod[si] == 0
    aug = dict(
        bias=float(rng.uniform(0.05, cfg['bias_max'])) if rng.random() < (cfg['p_bias_ct'] if ct else cfg['p_bias_mr']) else 0.0,
        bias_grid=rng.normal(size=(4, 4, 4)).astype(np.float32),
        bright=float(rng.uniform(*cfg['bright'])) if rng.random() < cfg['p_bright'] else 1.0,
        contrast=float(rng.uniform(*cfg['contrast'])) if rng.random() < cfg['p_contrast'] else 1.0,
        gamma=float(math.exp(rng.uniform(math.log(cfg['gamma'][0]), math.log(cfg['gamma'][1])))) if rng.random() < cfg['p_gamma'] else 1.0,
        lowres=float(rng.uniform(*cfg['lowres'])) if rng.random() < cfg['p_lowres'] and T.gap[si] <= cfg['lowres_max_gap'] else 1.0,
        lowres_axis=int(rng.integers(-1, 3)),
        noise=float(rng.uniform(*cfg['noise'])) if rng.random() < cfg['p_noise'] else 0.0,
        seed=int(rng.integers(2 ** 62)),
    )
    try:
        vol = read_volume(T.path[si], ('img', 'dist', 'seg') if T.terr[si] else ('img', 'dist'))
        img, mu, sd = strip_and_stats(vol['img'], vol['dist'], r, cfg['dist_unit_mm'])
    except Exception as e:
        raise RuntimeError(f'series {T.uid[si]} ({T.path[si]}): {e}') from e
    seg = vol.get('seg')
    return dict(
        img=torch.from_numpy(img), seg=None if seg is None else torch.from_numpy(seg.copy()),
        mu=mu, sd=sd, M=M.astype(np.float32), origin=origin, pts=yv.astype(np.float32), locs=locs.astype(np.int64),
        vox_mm=float(T.spacing[si] * s), mod=int(T.mod[si]), flip=flip, aug=aug,
    )


class PatchStream(torch.utils.data.IterableDataset):
    def __init__(self, tables, cfg, seed):
        super().__init__()
        self.T = tables
        self.cfg = cfg
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([int(x) for x in self.seed] + [0 if info is None else info.id])
        while True:
            yield draw_sample(rng, self.T, self.cfg)


def collate_list(batch):
    return batch


class BatchBuilder:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        P = cfg['patch']
        j = torch.arange(P, device=device, dtype=torch.float32)
        self.base = torch.stack(torch.meshgrid(j, j, j, indexing='ij'), -1).reshape(-1, 3)
        self.terr_flip = torch.tensor(TERR_FLIP, device=device)

    def sample_grid(self, M, origin):
        n, P = self.cfg['cube'], self.cfg['patch']
        m = (n - 1) / 2.0
        src = (self.base + (origin - m)) @ M.T + m
        return (src / (n - 1) * 2.0 - 1.0).flip(-1).reshape(1, P, P, P, 3)

    def intensity(self, v, aug, gen):
        fg = (v > 0).float()
        dev = v.device
        if aug['bias'] > 0:
            g = torch.as_tensor(aug['bias_grid'], device=dev)[None, None]
            field = F.interpolate(g, size=v.shape, mode='trilinear', align_corners=True)[0, 0]
            v = v * torch.exp(aug['bias'] * field)
        if aug['bright'] != 1.0:
            v = v * aug['bright']
        if aug['contrast'] != 1.0:
            mu = (v * fg).sum() / fg.sum().clamp(min=1.0)
            v = (v - mu) * aug['contrast'] + mu
        v = v.clamp(0.0, 1.0)
        if aug['gamma'] != 1.0:
            v = v.pow(aug['gamma'])
        if aug['lowres'] > 1.0:
            size = list(v.shape)
            axes = range(3) if aug['lowres_axis'] < 0 else [aug['lowres_axis']]
            small = list(size)
            for a in axes:
                small[a] = max(8, int(round(size[a] / aug['lowres'])))
            v = F.interpolate(F.interpolate(v[None, None], size=small, mode='trilinear', align_corners=False),
                              size=size, mode='trilinear', align_corners=False)[0, 0]
        if aug['noise'] > 0:
            v = v + aug['noise'] * torch.randn(v.shape, device=dev, generator=gen)
        return v.clamp(0.0, 1.0) * fg

    def territory(self, seg, grid, flip):
        lab = F.grid_sample(seg.to(self.device, non_blocking=True).float()[None, None], grid, mode='nearest',
                            padding_mode='zeros', align_corners=True)[0, 0].round().long().clamp(0, N_TERR - 1)
        if flip:
            lab = self.terr_flip[lab]
        oh = torch.stack([(lab == k) for k in range(1, N_TERR)]).float()
        oh = F.max_pool3d(oh[None], 2)[0]
        for _ in range(self.cfg['terr_dilate']):
            oh = F.max_pool3d(oh[None], 3, 1, 1)[0]
        anyv = oh.amax(0) > 0
        return torch.where(anyv, oh.argmax(0) + 1, torch.zeros_like(anyv, dtype=torch.long))

    def axis_centers(self, origin, size, stride):
        i = torch.arange(size, device=self.device, dtype=torch.float32)
        return [origin[a] + stride * i + (stride - 1) / 2.0 for a in range(3)]

    def coords(self, origin, size, stride):
        n = self.cfg['cube']
        m = (n - 1) / 2.0
        c = self.axis_centers(origin, size, stride)
        g = torch.meshgrid(*[(x - m) / (n / 2.0) for x in c], indexing='ij')
        return torch.stack(g, 0)

    def gauss(self, pts, origin, size, stride, sigma_vox):
        c = self.axis_centers(origin, size, stride)
        hm = torch.zeros((size, size, size), device=self.device)
        d2s = []
        for p in pts:
            d2 = ((c[0] - p[0]) ** 2)[:, None, None] + ((c[1] - p[1]) ** 2)[None, :, None] + ((c[2] - p[2]) ** 2)[None, None, :]
            hm = torch.maximum(hm, torch.exp(-d2 / (2.0 * sigma_vox ** 2)))
            d2s.append(d2)
        return hm, d2s

    def __call__(self, samples):
        cfg, dev = self.cfg, self.device
        P = cfg['patch']
        Po, Pa = P // 2, P // 4
        xs, cs, hms, auxs, terrs, tws, yap, wap, yloc, wloc, mods = [], [], [], [], [], [], [], [], [], [], []
        for s in samples:
            gen = torch.Generator(device=dev)
            gen.manual_seed(s['aug']['seed'])
            M = torch.as_tensor(s['M'], device=dev)
            origin = torch.as_tensor(s['origin'], device=dev, dtype=torch.float32)
            grid = self.sample_grid(M, origin)
            vol = s['img'].to(dev, non_blocking=True).float()[None, None]
            v = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0] / 255.0
            v = self.intensity(v, s['aug'], gen)
            xs.append(((v * 255.0 - s['mu']) / s['sd'])[None])
            cs.append(self.coords(origin, Po, 2))
            pts = torch.as_tensor(s['pts'], device=dev)
            locs = torch.as_tensor(s['locs'], device=dev)
            sigma = cfg['hm_sigma_mm'] / s['vox_mm']
            rb = cfg['ball_mm'] / s['vox_mm']
            hm, d2s = self.gauss(pts, origin, Po, 2, sigma)
            hm_a, _ = self.gauss(pts, origin, Pa, 4, sigma)
            if s['seg'] is not None:
                terr = self.territory(s['seg'], grid, s['flip'])
                tw = torch.ones((Po, Po, Po), device=dev)
            else:
                terr = torch.zeros((Po, Po, Po), device=dev, dtype=torch.long)
                tw = torch.zeros((Po, Po, Po), device=dev)
            for d2, k in zip(d2s, locs):
                inside = d2 <= rb ** 2
                terr = torch.where(inside, k + 1, terr)
                tw = torch.where(inside, torch.ones_like(tw), tw)
            center = origin + P / 2.0 - 0.5
            e = (pts - center).abs().amax(1) if len(pts) else torch.zeros(0, device=dev)
            core = e <= P / 2.0 - cfg['border']
            amb = (e > P / 2.0 - cfg['border']) & (e <= P / 2.0 + cfg['border'])
            ya = core.any().float()
            wa = 1.0 - ((~core.any()) & amb.any()).float()
            yl = torch.zeros(N_LOC, device=dev)
            wl = torch.ones(N_LOC, device=dev)
            if len(pts):
                yl = yl.index_put((locs[core],), torch.ones_like(locs[core], dtype=torch.float32), accumulate=False)
                amb_only = torch.zeros(N_LOC, device=dev).index_put((locs[amb],), torch.ones_like(locs[amb], dtype=torch.float32))
                wl = 1.0 - ((amb_only > 0) & (yl == 0)).float()
            hms.append(hm[None])
            auxs.append(hm_a[None])
            terrs.append(terr)
            tws.append(tw)
            yap.append(ya)
            wap.append(wa)
            yloc.append(yl)
            wloc.append(wl)
            mods.append(s['mod'])
        return dict(
            x=torch.stack(xs), coords=torch.stack(cs), hm=torch.stack(hms), aux=torch.stack(auxs),
            terr=torch.stack(terrs), terr_w=torch.stack(tws), y_ap=torch.stack(yap), w_ap=torch.stack(wap),
            y_loc=torch.stack(yloc), w_loc=torch.stack(wloc), mod=torch.tensor(mods, device=dev),
        )


class VolumeSet(torch.utils.data.Dataset):
    def __init__(self, uids, paths, strip_mm, dist_unit_mm):
        self.uids = list(uids)
        self.paths = list(paths)
        self.strip_mm = strip_mm
        self.dist_unit_mm = dist_unit_mm

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        try:
            vol = read_volume(self.paths[i], ('img', 'dist'))
            img, mu, sd = strip_and_stats(vol['img'], vol['dist'], self.strip_mm, self.dist_unit_mm)
        except Exception as e:
            raise RuntimeError(f'series {self.uids[i]} ({self.paths[i]}): {e}') from e
        return self.uids[i], torch.from_numpy(img), mu, sd
