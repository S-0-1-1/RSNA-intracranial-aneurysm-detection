import math
import os

import cc3d
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def volume_corners(plan):
    K, R, C = len(plan['d']), plan['rows'], plan['cols']
    pts = []
    for k in (0, K - 1):
        for r in (0, R - 1):
            for c in (0, C - 1):
                pts.append(plan['o'][k] + plan['d'][k] * plan['n'] + c * plan['dc'] * plan['u'] + r * plan['dr'] * plan['v'])
    return np.array(pts)


def axis_grid(lo, hi, spacing):
    s0, s1, s2 = spacing
    E = np.array([[0.0, 0.0, s2], [0.0, -s1, 0.0], [s0, 0.0, 0.0]])
    origin = np.array([lo[0], hi[1], lo[2]], np.float64)
    shape = (int(math.ceil((hi[2] - lo[2]) / s0)) + 1,
             int(math.ceil((hi[1] - lo[1]) / s1)) + 1,
             int(math.ceil((hi[0] - lo[0]) / s2)) + 1)
    return dict(origin=origin, E=E, shape=shape)


def cube_grid(center, side, n):
    s = side / n
    E = np.array([[s, 0.0, 0.0], [0.0, 0.0, s], [0.0, -s, 0.0]])
    origin = np.asarray(center, np.float64) - E @ np.full(3, (n - 1) / 2.0)
    return dict(origin=origin, E=E, shape=(n, n, n))


def grid_spacing(grid):
    return np.linalg.norm(grid['E'], axis=0)


def grid_corners(grid):
    pts = []
    for i in (0, grid['shape'][0] - 1):
        for j in (0, grid['shape'][1] - 1):
            for k in (0, grid['shape'][2] - 1):
                pts.append(grid['origin'] + grid['E'] @ np.array([i, j, k], np.float64))
    return np.array(pts)


def world_to_grid(grid, p):
    return (np.linalg.inv(grid['E']) @ (np.asarray(p, np.float64) - grid['origin']).T).T


def chunk_world(grid, a, b, ref, device):
    E = torch.as_tensor(grid['E'], dtype=torch.float32, device=device)
    o = torch.as_tensor(grid['origin'] - ref, dtype=torch.float32, device=device)
    i0 = torch.arange(a, b, dtype=torch.float32, device=device)
    i1 = torch.arange(grid['shape'][1], dtype=torch.float32, device=device)
    i2 = torch.arange(grid['shape'][2], dtype=torch.float32, device=device)
    return (o + i0[:, None, None, None] * E[:, 0] + i1[None, :, None, None] * E[:, 1]
            + i2[None, None, :, None] * E[:, 2])


class StackGeometry:
    def __init__(self, plan, device):
        self.ref = plan['o'][0] + plan['d'][0] * plan['n']
        n = plan['n']
        self.device = device
        self.n = torch.as_tensor(n, dtype=torch.float32, device=device)
        self.u = torch.as_tensor(plan['u'], dtype=torch.float32, device=device)
        self.v = torch.as_tensor(plan['v'], dtype=torch.float32, device=device)
        self.dk = torch.as_tensor(plan['d'] - np.dot(self.ref, n), dtype=torch.float32, device=device)
        self.ok = torch.as_tensor(plan['o'] - (self.ref - np.dot(self.ref, n) * n), dtype=torch.float32, device=device)
        self.dr, self.dc = plan['dr'], plan['dc']
        self.K, self.R, self.C = len(plan['d']), plan['rows'], plan['cols']
        self.gap = float(np.median(np.diff(plan['d'])))

    def coords(self, p):
        d = p @ self.n
        idx = torch.searchsorted(self.dk, d.contiguous()).clamp(1, self.K - 1)
        d0, d1 = self.dk[idx - 1], self.dk[idx]
        w = (d - d0) / (d1 - d0)
        kf = (idx - 1).to(torch.float32) + w
        o0, o1 = self.ok[idx - 1], self.ok[idx]
        q = p - (o0 + w[..., None] * (o1 - o0)) - d[..., None] * self.n
        return kf, q @ self.v / self.dr, q @ self.u / self.dc


def gaussian_blur(x, sigmas):
    benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        for axis, s in zip((2, 3, 4), sigmas):
            if s <= 0.15 or x.shape[axis] < 2:
                continue
            r = int(math.ceil(3.0 * s))
            t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
            k = torch.exp(-0.5 * (t / s) ** 2)
            k = k / k.sum()
            shape = [1, 1, 1, 1, 1]
            shape[axis] = 2 * r + 1
            pad = [0, 0, 0, 0, 0, 0]
            j = {4: 0, 3: 2, 2: 4}[axis]
            pad[j] = pad[j + 1] = r
            x = F.conv3d(F.pad(x, pad, mode='replicate'), k.view(shape))
    finally:
        torch.backends.cudnn.benchmark = benchmark
    return x


def sample_stack(vol, geo, grid, fill, antialias, chunk=16, mode='bilinear'):
    corners = torch.as_tensor(grid_corners(grid) - geo.ref, dtype=torch.float32, device=geo.device)
    kf, rf, cf = geo.coords(corners)
    sp = float(grid_spacing(grid).max())
    if antialias:
        sig = [max(0.0, (sp / geo.gap - 1) / 2), max(0.0, (sp / geo.dr - 1) / 2), max(0.0, (sp / geo.dc - 1) / 2)]
    else:
        sig = [0.0, 0.0, 0.0]
    bounds = []
    for c, size, s in zip((kf, rf, cf), (geo.K, geo.R, geo.C), sig):
        m = int(math.ceil(3 * s)) + 2
        lo = max(0, int(math.floor(float(c.min()))) - m)
        hi = min(size - 1, int(math.ceil(float(c.max()))) + m)
        bounds.append((lo, hi))
    out = torch.full(tuple(grid['shape']), float(fill), dtype=torch.float32, device=geo.device)
    valid = torch.zeros(tuple(grid['shape']), dtype=torch.bool, device=geo.device)
    if any(lo > hi for lo, hi in bounds):
        return out, valid
    bounds = [(lo, max(hi, lo + 1)) if size > 1 else (lo, hi) for (lo, hi), size in zip(bounds, (geo.K, geo.R, geo.C))]
    bounds = [(min(lo, size - 2), min(hi, size - 1)) for (lo, hi), size in zip(bounds, (geo.K, geo.R, geo.C))]
    (k0, k1), (r0, r1), (c0, c1) = bounds
    sub = vol[k0:k1 + 1, r0:r1 + 1, c0:c1 + 1][None, None].float()
    sub = gaussian_blur(sub, sig)
    D, H, W = sub.shape[2:]
    for a in range(0, grid['shape'][0], chunk):
        b = min(a + chunk, grid['shape'][0])
        p = chunk_world(grid, a, b, geo.ref, geo.device)
        kk, rr, cc = geo.coords(p)
        ok = ((kk >= -0.5) & (kk <= geo.K - 0.5) & (rr >= -0.5) & (rr <= geo.R - 0.5)
              & (cc >= -0.5) & (cc <= geo.C - 0.5))
        g = torch.stack([2 * (cc - c0) / (W - 1) - 1, 2 * (rr - r0) / (H - 1) - 1, 2 * (kk - k0) / (D - 1) - 1], -1)
        val = F.grid_sample(sub, g[None], mode=mode, padding_mode='border', align_corners=True)[0, 0]
        out[a:b] = torch.where(ok, val, torch.full_like(val, float(fill)))
        valid[a:b] = ok
    return out, valid


def sample_regular(src, src_grid, grid, mode, fill=0.0, chunk=16):
    device = src.device
    inv = torch.as_tensor(np.linalg.inv(src_grid['E']), dtype=torch.float32, device=device)
    ref = src_grid['origin']
    D, H, W = src.shape
    s = src[None, None].float()
    out = torch.empty(tuple(grid['shape']), dtype=torch.float32, device=device)
    for a in range(0, grid['shape'][0], chunk):
        b = min(a + chunk, grid['shape'][0])
        p = chunk_world(grid, a, b, ref, device)
        idx = p @ inv.T
        i0, i1, i2 = idx[..., 0], idx[..., 1], idx[..., 2]
        ok = (i0 >= -0.5) & (i0 <= D - 0.5) & (i1 >= -0.5) & (i1 <= H - 0.5) & (i2 >= -0.5) & (i2 <= W - 0.5)
        g = torch.stack([2 * i2 / max(W - 1, 1) - 1, 2 * i1 / max(H - 1, 1) - 1, 2 * i0 / max(D - 1, 1) - 1], -1)
        val = F.grid_sample(s, g[None], mode=mode, padding_mode='border', align_corners=True)[0, 0]
        out[a:b] = torch.where(ok, val, torch.full_like(val, float(fill)))
    return out


def largest_component(mask):
    labels = cc3d.connected_components(np.asarray(mask, bool), connectivity=6)
    counts = np.bincount(labels.ravel())
    if counts.size <= 1:
        return np.asarray(mask, bool)
    counts[0] = 0
    return labels == int(np.argmax(counts))


def air_value(modality, cfg):
    return cfg['ct_clip'][0] if modality == 'CT' else 0.0


def head_box(vol, geo, plan, cfg):
    corners = volume_corners(plan)
    lo, hi = corners.min(0), corners.max(0)
    s = cfg['overview_mm']
    grid = axis_grid(lo, hi, (s, s, s))
    img, valid = sample_stack(vol, geo, grid, air_value(plan['modality'], cfg), antialias=False)
    a = img.cpu().numpy()
    m = valid.cpu().numpy()
    if plan['modality'] == 'CT':
        body = (a > -500.0) & m
    else:
        vals = a[m]
        body = (a > 0.1 * float(np.percentile(vals, 99.5))) & m if vals.size else m
    if not body.any():
        return lo, hi, lo, hi
    body = largest_component(body)
    z = grid['origin'][2] + np.arange(grid['shape'][0]) * s
    z_top = z[np.nonzero(body.any(axis=(1, 2)))[0].max()]
    head = body & (z >= z_top - cfg['head_xy_mm'])[:, None, None]
    w = grid['origin'] + np.argwhere(head) @ grid['E'].T
    xy_lo = np.maximum(w.min(0) - cfg['body_margin_mm'], lo)
    xy_hi = np.minimum(w.max(0) + cfg['body_margin_mm'], hi)
    b_lo = np.array([xy_lo[0], xy_lo[1], max(lo[2], z_top - cfg['head_mm'])])
    b_hi = np.array([xy_hi[0], xy_hi[1], min(hi[2], z_top + cfg['body_margin_mm'])])
    r_lo = np.array([xy_lo[0], xy_lo[1], max(lo[2], z_top - cfg['head_xy_mm'])])
    r_hi = np.array([xy_hi[0], xy_hi[1], min(hi[2], z_top)])
    return b_lo, b_hi, r_lo, r_hi


def hdbet_weights():
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.paths import folder_with_parameter_files
    maybe_download_parameters()
    return folder_with_parameter_files


def load_hdbet(device, weights_dir):
    os.environ.setdefault('nnUNet_raw', '/tmp/nnunet_raw')
    os.environ.setdefault('nnUNet_preprocessed', '/tmp/nnunet_preprocessed')
    os.environ.setdefault('nnUNet_results', '/tmp/nnunet_results')
    os.environ['nnUNet_compile'] = 'F'
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
        device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(str(weights_dir), 'all')
    return predictor


def hdbet_spacing(predictor):
    tb = list(predictor.plans_manager.transpose_backward)
    sp = list(predictor.configuration_manager.spacing)
    return tuple(float(sp[tb[i]]) for i in range(3))


def run_hdbet(predictor, arr, sp, tta):
    default = predictor.use_mirroring
    predictor.use_mirroring = tta
    seg = predictor.predict_single_npy_array(arr[None], {'spacing': list(sp)}, None, None, False)
    predictor.use_mirroring = default
    return largest_component(np.asarray(seg) > 0)


def brain_mask(vol, geo, plan, predictor, cfg):
    lo, hi, r_lo, r_hi = head_box(vol, geo, plan, cfg)
    sp = hdbet_spacing(predictor)
    grid = axis_grid(lo, hi, sp)
    img, _ = sample_stack(vol, geo, grid, air_value(plan['modality'], cfg), antialias=True)
    arr = img.cpu().numpy()
    voxel_cm3 = float(np.prod(sp)) / 1000.0
    for method, tta in (('hdbet', False), ('hdbet_tta', True)):
        mask = run_hdbet(predictor, arr, sp, tta)
        if mask.sum() * voxel_cm3 >= cfg['min_brain_cm3']:
            return mask, grid, method, (r_lo, r_hi)
    return None, grid, 'none', (r_lo, r_hi)


def process_volume(vol_np, plan, predictor, device, cfg):
    modality = plan['modality']
    if modality not in ('CT', 'MR'):
        raise ValueError(f"unsupported Modality tag '{modality}'")
    vol = torch.from_numpy(vol_np).to(device)
    if modality == 'CT':
        vol.clamp_(cfg['ct_clip'][0], cfg['ct_clip'][1])
    geo = StackGeometry(plan, device)
    mask, hgrid, method, roi = brain_mask(vol, geo, plan, predictor, cfg)
    sp = grid_spacing(hgrid)
    if mask is None:
        if cfg['no_brain'] != 'roi':
            raise ValueError(f"{plan['series_dir']}: HD-BET found no brain, also with test-time mirroring")
        brain_cm3 = 0.0
        b_lo, b_hi = roi
    else:
        brain_cm3 = float(mask.sum() * np.prod(sp) / 1000.0)
        idx = np.argwhere(mask)
        w = hgrid['origin'] + np.stack([idx.min(0), idx.max(0)]) @ hgrid['E'].T
        b_lo, b_hi = w.min(0), w.max(0)
    extent = b_hi - b_lo
    side = float(np.clip(extent.max() + 2 * cfg['margin_mm'], cfg['cube_min_mm'], cfg['cube_max_mm']))
    n = cfg['size']
    cube = cube_grid((b_lo + b_hi) / 2.0, side, n)
    s = side / n
    img, valid = sample_stack(vol, geo, cube, air_value(modality, cfg), antialias=True)
    del vol
    if mask is None:
        inside = valid.clone()
        if modality == 'MR':
            inside &= img > 0.1 * float(torch.quantile(img[valid][:16_000_000], 0.995))
        dist = torch.zeros_like(img)
    else:
        m_src = torch.from_numpy(mask.astype(np.float32)).to(device)
        inside = sample_regular(m_src, hgrid, cube, 'bilinear') >= 0.5
        half = inside[::2, ::2, ::2].cpu().numpy()
        if not half.any():
            raise ValueError(f"{plan['series_dir']}: brain mask lies outside the cube")
        dh = ndi.distance_transform_edt(~half, sampling=(2 * s, 2 * s, 2 * s)).astype(np.float32)
        hgrid_half = dict(origin=cube['origin'], E=cube['E'] * 2.0, shape=half.shape)
        dist = sample_regular(torch.from_numpy(dh).to(device), hgrid_half, cube, 'bilinear')
        dist[inside] = 0.0
    if modality == 'CT':
        lo, hi = cfg['ct_window']
    else:
        vals = img[inside & valid]
        if vals.numel() > 16_000_000:
            vals = vals[torch.randperm(vals.numel(), device=vals.device)[:16_000_000]]
        q = torch.quantile(vals, torch.tensor([cfg['mr_pct'][0] / 100.0, cfg['mr_pct'][1] / 100.0], device=vals.device))
        lo, hi = float(q[0]), float(q[1])
        if hi <= lo:
            raise ValueError(f"{plan['series_dir']}: degenerate MR intensity range")
    u8 = ((img - lo) / (hi - lo) * 255.0).round_().clamp_(0, 255).to(torch.uint8)
    u8[~(valid & (dist <= cfg['keep_mm']))] = 0
    dq = (dist / cfg['dist_unit_mm']).round_().clamp_(0, 254).to(torch.uint8)
    meta = dict(
        cube_center=cube['origin'] + cube['E'] @ np.full(3, (n - 1) / 2.0),
        cube_side_mm=side, spacing_mm=s, brain_cm3=brain_cm3, mask_method=method,
        brain_lo=b_lo, brain_hi=b_hi, norm_lo=lo, norm_hi=hi,
        fov_fraction=float(valid.float().mean()), inside_fraction=float(inside.float().mean()),
        ct_intercept=plan.get('ct_intercept', 'n/a'),
    )
    return dict(img=u8.cpu().numpy(), dist=dq.cpu().numpy(), valid=valid.cpu().numpy(), cube=cube, meta=meta, geo=geo)


def map_points(points_world, cube, dist_u8, valid, cfg):
    rows = []
    n = cube['shape'][0]
    for p in np.asarray(points_world).reshape(-1, 3):
        a = world_to_grid(cube, p)
        inside = bool(np.all(a >= -0.5) and np.all(a <= n - 0.5))
        dmm, fov = float('nan'), False
        if inside:
            i = np.clip(np.round(a).astype(int), 0, n - 1)
            dmm = float(dist_u8[i[0], i[1], i[2]]) * cfg['dist_unit_mm']
            fov = bool(valid[i[0], i[1], i[2]])
        rows.append(dict(a0=float(a[0]), a1=float(a[1]), a2=float(a[2]), inside_cube=inside, in_fov=fov, dist_mm=dmm))
    return rows
