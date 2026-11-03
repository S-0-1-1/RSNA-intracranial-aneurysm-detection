import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from aneurysm import config, cube, dicom, segmentation, storage
from aneurysm.labels import LOCATIONS


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v


def save(out_dir, uid, img, dist, seg, row, meta):
    blob = storage.encode_volume(img, dist, seg)
    tmp = out_dir / 'vol' / f'.{uid}.tmp'
    tmp.write_bytes(blob)
    os.replace(tmp, out_dir / 'vol' / f'{uid}.pkl')
    meta.write(json.dumps(jsonable(dict(row, bytes=len(blob)))) + '\n')
    meta.flush()


def main(job_path, out_dir, name):
    with open(job_path) as f:
        job = json.load(f)
    cfg = config.prep_cfg(**job['cfg'])
    out_dir = Path(out_dir)
    series_root = Path(job['series_root'])
    seg_index = segmentation.index_segmentations(job['seg_root'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    hdbet = cube.load_hdbet(device, job['hdbet'])
    series = job['series']

    with ThreadPoolExecutor(4) as files, ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(1) as writer, \
            open(out_dir / 'meta' / f'{name}.jsonl', 'a') as meta:
        loading = deque(reader.submit(dicom.load_series, series_root / uid, cfg, files, points) for uid, points in series[:2])
        saving = deque()
        for i, (uid, points) in enumerate(series):
            if i + 2 < len(series):
                nxt, nxt_points = series[i + 2]
                loading.append(reader.submit(dicom.load_series, series_root / nxt, cfg, files, nxt_points))
            try:
                plan, summary, vol = loading.popleft().result()
                res = cube.process_volume(vol, plan, hdbet, device, cfg)
                seg, seg_info = None, {}
                img_path, lab_path = segmentation.find_seg_files(seg_index, job['seg_root'], uid)
                if lab_path is not None:
                    seg, seg_info = segmentation.resample_seg(vol, res['geo'], res['cube'], img_path, lab_path, device)
            except Exception as e:
                raise RuntimeError(f'series {uid}: {e}') from e
            del vol
            mapped = cube.map_points(plan['points_world'], res['cube'], res['dist'], res['valid'], cfg)
            pts = [dict(uid=uid, location=p['location'], loc_idx=LOCATIONS.index(p['location']), **m)
                   for p, m in zip(points or [], mapped)]
            img = res['img']
            row = dict(uid=uid, **summary, **res['meta'], **seg_info, has_seg=seg is not None,
                       fg_frac=float(np.count_nonzero(img[::8]) / img[::8].size), points=pts)
            saving.append(writer.submit(save, out_dir, uid, img, res['dist'], seg, row, meta))
            while len(saving) > 2:
                saving.popleft().result()
            if (i + 1) % 100 == 0 or i + 1 == len(series):
                print(f'{name}: {i + 1}/{len(series)}', flush=True)
        for f in saving:
            f.result()


if __name__ == '__main__':
    main(*sys.argv[1:])
