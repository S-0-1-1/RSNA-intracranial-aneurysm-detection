import itertools
import os

import nibabel as nib
import numpy as np
import torch

from aneurysm.cube import sample_regular, sample_stack
from aneurysm.labels import LOCATIONS

SEG_VALUE_TO_LOCATION = {
    1: 'Other Posterior Circulation',
    2: 'Basilar Tip',
    3: 'Right Posterior Communicating Artery',
    4: 'Left Posterior Communicating Artery',
    5: 'Right Infraclinoid Internal Carotid Artery',
    6: 'Left Infraclinoid Internal Carotid Artery',
    7: 'Right Supraclinoid Internal Carotid Artery',
    8: 'Left Supraclinoid Internal Carotid Artery',
    9: 'Right Middle Cerebral Artery',
    10: 'Left Middle Cerebral Artery',
    11: 'Right Anterior Cerebral Artery',
    12: 'Left Anterior Cerebral Artery',
    13: 'Anterior Communicating Artery',
}
SEG_LUT = np.zeros(256, np.uint8)
for value, name in SEG_VALUE_TO_LOCATION.items():
    SEG_LUT[value] = LOCATIONS.index(name) + 1


def index_segmentations(seg_root):
    out = {}
    if not seg_root or not os.path.isdir(seg_root):
        return out
    for dirpath, _, files in os.walk(seg_root):
        for name in files:
            if name.endswith('.nii') or name.endswith('.nii.gz'):
                uid = name.split('.nii')[0].replace('_cowseg', '')
                out.setdefault(uid, []).append(os.path.relpath(os.path.join(dirpath, name), seg_root))
    return out


def find_seg_files(seg_index, seg_root, uid):
    names = seg_index.get(uid)
    if not names:
        return None, None
    lab = next((n for n in names if 'cowseg' in n), None)
    img = next((n for n in names if 'cowseg' not in n), None)
    return (None if img is None else os.path.join(seg_root, img)), (None if lab is None else os.path.join(seg_root, lab))


def nifti_grid(path):
    im = nib.load(path)
    data = np.asanyarray(im.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ im.affine
    return data, dict(origin=A[:3, 3].copy(), E=A[:3, :3].copy(), shape=data.shape)


def stack_alignment(vol, nii_img, n_samples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    K, R, C = vol.shape
    kk, rr, cc = rng.integers(0, K, n_samples), rng.integers(0, R, n_samples), rng.integers(0, C, n_samples)
    a = vol[kk, rr, cc].astype(np.float64)
    best = None
    for perm in itertools.permutations(range(3)):
        if tuple(nii_img.shape[p] for p in perm) != (K, R, C):
            continue
        t = np.transpose(nii_img, perm)
        for flips in itertools.product((False, True), repeat=3):
            ik = K - 1 - kk if flips[0] else kk
            ir = R - 1 - rr if flips[1] else rr
            ic = C - 1 - cc if flips[2] else cc
            b = t[ik, ir, ic].astype(np.float64)
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if best is None or r > best[0]:
                best = (r, perm, flips)
    return best


def apply_alignment(arr, perm, flips):
    t = np.transpose(arr, perm)
    return np.ascontiguousarray(t[tuple(slice(None, None, -1) if f else slice(None) for f in flips)])


def resample_seg(vol, geo, cube, img_path, lab_path, device, min_corr=0.9):
    lab_data, lab_grid = nifti_grid(lab_path)
    lab = SEG_LUT[np.clip(np.rint(lab_data).astype(np.int64), 0, 255)]
    info = dict(seg_method='affine', seg_corr=float('nan'))
    if img_path is not None:
        nii_img = np.asanyarray(nib.load(img_path).dataobj)
        if nii_img.ndim == 4:
            nii_img = nii_img[..., 0]
        if nii_img.shape == lab.shape:
            best = stack_alignment(vol, nii_img)
            if best is not None:
                info['seg_corr'] = best[0]
                if best[0] >= min_corr:
                    src = torch.from_numpy(apply_alignment(lab, best[1], best[2]).astype(np.float32)).to(device)
                    out, _ = sample_stack(src, geo, cube, 0.0, antialias=False, mode='nearest')
                    info['seg_method'] = 'stack'
                    return out.round_().clamp_(0, 255).to(torch.uint8).cpu().numpy(), info
    src = torch.from_numpy(lab.astype(np.float32)).to(device)
    out = sample_regular(src, lab_grid, cube, 'nearest')
    return out.round_().clamp_(0, 255).to(torch.uint8).cpu().numpy(), info
