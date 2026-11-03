import copy
import itertools
import math
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from aneurysm import config, cube, dicom
from aneurysm.dataset import strip_and_stats
from aneurysm.labels import N_TERR, TERR_FLIP
from aneurysm.model import AneurysmNet, build_resenc_l, readout


def window_origins(zone, n, P, stride):
    out = []
    for a in range(3):
        lo, hi = zone[a]
        length = hi - lo + 1
        if length <= P:
            starts = [2 * int(round(((lo + hi) / 2.0 - P / 2.0) / 2.0))]
        else:
            k = int(math.ceil((length - P) / stride)) + 1
            pos = np.linspace(lo, hi - P + 1, k)
            starts = ([2 * int(math.floor(pos[0] / 2.0))] + [2 * int(round(x / 2.0)) for x in pos[1:-1]]
                      + [2 * int(math.ceil(pos[-1] / 2.0))])
        out.append(sorted({int(np.clip(s, 0, n - P)) for s in starts}))
    return [(a, b, c) for a in out[0] for b in out[1] for c in out[2]]


def gaussian_weight(size, device):
    i = torch.arange(size, device=device, dtype=torch.float32)
    g = torch.exp(-((i - (size - 1) / 2.0) ** 2) / (2.0 * (size / 8.0) ** 2))
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    w = w / w.max()
    return w.clamp(min=1e-3)


@torch.no_grad()
def predict_maps(model, img_u8, mu, sd, zone, cfg, stride, device, tta=False):
    n, P = cfg['cube'], cfg['patch']
    Po, no = P // 2, n // 2
    m = (n - 1) / 2.0
    amp = device.type == 'cuda'
    z = (img_u8.to(device, non_blocking=True).float() - mu) / sd
    if amp:
        z = z.half()
    acc_hm = torch.zeros((no, no, no), device=device)
    acc_t = torch.zeros((N_TERR, no, no, no), device=device)
    wsum = torch.zeros((no, no, no), device=device)
    w = gaussian_weight(Po, device)
    terr_flip = torch.tensor(TERR_FLIP, device=device)
    i = torch.arange(Po, device=device, dtype=torch.float32)
    for flip in ((False, True) if tta else (False,)):
        zz = z.flip(0) if flip else z
        for o in window_origins(zone, n, P, stride):
            src = (n - P - o[0], o[1], o[2]) if flip else o
            x = zz[src[0]:src[0] + P, src[1]:src[1] + P, src[2]:src[2] + P][None, None]
            c = torch.stack(torch.meshgrid(*[(src[a] + 2 * i + 0.5 - m) / (n / 2.0) for a in range(3)], indexing='ij'), 0)[None]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x, c)
            hm = out['hm'][0, 0].float()
            t = out['terr'][0].float()
            if flip:
                hm = hm.flip(0)
                t = t.flip(1)[terr_flip]
            s0, s1, s2 = o[0] // 2, o[1] // 2, o[2] // 2
            acc_hm[s0:s0 + Po, s1:s1 + Po, s2:s2 + Po] += w * hm
            acc_t[:, s0:s0 + Po, s1:s1 + Po, s2:s2 + Po] += w * t
            wsum[s0:s0 + Po, s1:s1 + Po, s2:s2 + Po] += w
    covered = wsum > 0
    hm = torch.where(covered, acc_hm / wsum.clamp(min=1e-6), torch.full_like(acc_hm, -30.0))
    t = acc_t / wsum.clamp(min=1e-6)
    return hm, t, covered


def predict_volume(model, img_u8, mu, sd, zone, cfg, stride, device, tta=False):
    hm, t, covered = predict_maps(model, img_u8, mu, sd, zone, cfg, stride, device, tta)
    p_ap, p_loc = readout(hm[None, None], t[None], cfg['readout_k'], covered[None])
    return np.append(p_loc[0].cpu().numpy().astype(np.float64), float(p_ap[0]))


def split_hdbet(predictor, devices):
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    nets = [predictor.network]
    for d in devices[1:]:
        net = copy.deepcopy(predictor.network)
        net.load_state_dict(predictor.list_of_parameters[0])
        nets.append(net.to(d).eval())
    pool = ThreadPoolExecutor(len(devices) - 1)
    heads = predictor.label_manager.num_segmentation_heads
    patch = tuple(predictor.configuration_manager.patch_size)

    def forward(net, x):
        pred = net(x)
        axes = predictor.allowed_mirroring_axes if predictor.use_mirroring else None
        if axes is not None:
            axes = [a + 2 for a in axes]
            combos = [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]
            for c in combos:
                pred += torch.flip(net(torch.flip(x, c)), c)
            pred /= len(combos) + 1
        return pred

    def run(j, data, slicers):
        dev = devices[j]
        with torch.inference_mode(), torch.autocast(dev.type, enabled=dev.type == 'cuda'):
            d = data.to(dev)
            logits = torch.zeros((heads, *d.shape[1:]), dtype=torch.half, device=dev)
            count = torch.zeros(d.shape[1:], dtype=torch.half, device=dev)
            g = compute_gaussian(patch, sigma_scale=1. / 8, value_scaling_factor=10, device=dev)
            for sl in slicers:
                pred = forward(nets[j], torch.clone(d[sl][None], memory_format=torch.contiguous_format))[0]
                logits[sl] += pred * g
                count[sl[1:]] += g
        return logits, count

    def sliding_window(self, data, slicers, do_on_device=True):
        nd = len(devices)
        jobs = [pool.submit(run, j, data, slicers[j::nd]) for j in range(1, nd) if slicers[j::nd]]
        logits, count = run(0, data, slicers[0::nd])
        for job in jobs:
            part, n = job.result()
            logits += part.to(devices[0])
            count += n.to(devices[0])
        torch.div(logits, count, out=logits)
        if torch.any(torch.isinf(logits)):
            raise RuntimeError('inf in the HD-BET logits')
        return logits

    predictor._internal_predict_sliding_window_return_logits = types.MethodType(sliding_window, predictor)


class Predictor:
    def __init__(self, model_paths, devices, hdbet_dir, prep_cfg, read_threads=8):
        if not model_paths:
            raise FileNotFoundError('no model files given')
        self.devices = [torch.device(d) for d in devices]
        self.prep_cfg = prep_cfg
        self.groups = [[] for _ in self.devices]
        for i, path in enumerate(model_paths):
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            cfg = config.train_cfg(**ckpt['train_cfg'])
            if (cfg['cube'], cfg['keep_mm'], cfg['dist_unit_mm']) != (prep_cfg['size'], prep_cfg['keep_mm'], prep_cfg['dist_unit_mm']):
                raise ValueError(f'{path}: model and preprocessing configs do not match')
            device = self.devices[i % len(self.devices)]
            net = AneurysmNet(build_resenc_l().encoder, cfg['prior_logit'])
            net.load_state_dict(ckpt['model'])
            self.groups[i % len(self.devices)].append((net.to(device).eval(), cfg, ckpt['zone'], ckpt['stride'], device))
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True
        self.hdbet = cube.load_hdbet(self.devices[0], hdbet_dir)
        if len(self.devices) > 1:
            split_hdbet(self.hdbet, self.devices)
        import nnunetv2.inference.predict_from_raw_data as nnunet_predict
        nnunet_predict.empty_cache = lambda device: None
        self.read_pool = ThreadPoolExecutor(read_threads)
        self.gpu_pool = ThreadPoolExecutor(len(self.devices))

    def run_group(self, group, x, mu, sd):
        return [predict_volume(net, x, mu, sd, zone, cfg, stride, device) for net, cfg, zone, stride, device in group]

    def __call__(self, series_dir):
        plan, _, vol = dicom.load_series(series_dir, self.prep_cfg, self.read_pool)
        out = cube.process_volume(vol, plan, self.hdbet, self.devices[0], self.prep_cfg)
        del vol
        img, dist = out['img'], out['dist']
        if np.count_nonzero(img[::8]) / img[::8].size < config.EMPTY_FRAC:
            return None
        x, mu, sd = strip_and_stats(img, dist, self.cfg['strip_inf_mm'], self.cfg['dist_unit_mm'])
        x = torch.from_numpy(x)
        jobs = [self.gpu_pool.submit(self.run_group, g, x, mu, sd) for g in self.groups if g]
        return np.mean([p for job in jobs for p in job.result()], axis=0)
