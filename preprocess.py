import ast
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
import torch

from aneurysm import config, cube, dicom
from aneurysm.dataset import make_folds

DATA_DIR = Path('data/rsna-intracranial-aneurysm-detection')
OUT_DIR = Path('data/prepared')
N_FOLDS = 5
FOLD_SEED = 42


def read_localizers(path):
    points = {}
    for r in pd.read_csv(path).itertuples(index=False):
        c = ast.literal_eval(r.coordinates)
        points.setdefault(r.SeriesInstanceUID, []).append(dict(
            sop=r.SOPInstanceUID, x=float(c['x']), y=float(c['y']), f=int(c['f']) if 'f' in c else None, location=r.location))
    return points


def read_meta(meta_dir):
    rows = {}
    for path in sorted(meta_dir.glob('*.jsonl')):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                rows[row['uid']] = row
    return rows


def patient_id(series_dir):
    first = dicom.list_series_files(series_dir)[0]
    ds = pydicom.dcmread(first, stop_before_pixels=True, specific_tags=['PatientID', 'StudyInstanceUID'])
    return str(ds.get('PatientID', '')) or str(ds.get('StudyInstanceUID', ''))


def main():
    data_dir, out_dir = DATA_DIR.resolve(), OUT_DIR.resolve()
    series_root = data_dir / 'series'
    (out_dir / 'vol').mkdir(parents=True, exist_ok=True)
    (out_dir / 'meta').mkdir(exist_ok=True)
    train = pd.read_csv(data_dir / 'train.csv')

    cfg = config.prep_cfg()
    single = [series_root / u for u in train.SeriesInstanceUID if len(os.listdir(series_root / u)) == 1]
    cfg['ps_table'] = dicom.multiframe_spacing_table(single)
    dicom.save_table(cfg['ps_table'], out_dir / 'ps_table.json')

    done = read_meta(out_dir / 'meta')
    todo = [u for u in train.SeriesInstanceUID if u not in done]
    if todo:
        localizers = read_localizers(data_dir / 'train_localizers.csv')
        hdbet = cube.hdbet_weights()
        random.Random(0).shuffle(todo)
        n_gpu = max(1, torch.cuda.device_count())
        print(f'preprocessing {len(todo)} series on {n_gpu} device(s)')
        procs = []
        for g in range(n_gpu):
            job = dict(cfg=cfg, hdbet=str(hdbet), series_root=str(series_root), seg_root=str(data_dir / 'segmentations'),
                       series=[(u, localizers.get(u)) for u in todo[g::n_gpu]])
            job_path = out_dir / f'job_{g}.json'
            with open(job_path, 'w') as f:
                json.dump(job, f)
            procs.append(subprocess.Popen([sys.executable, '-m', 'aneurysm.prep_worker', str(job_path), str(out_dir), f'gpu{g}'],
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)), cwd=Path(__file__).resolve().parent))
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f'preprocessing failed, worker exit codes {codes}')
        for p in out_dir.glob('job_*.json'):
            p.unlink()

    rows = read_meta(out_dir / 'meta')
    missing = set(train.SeriesInstanceUID) - set(rows)
    if missing:
        raise SystemExit(f'{len(missing)} series are missing from {out_dir / "meta"}')
    points = pd.DataFrame([p for r in rows.values() for p in r['points']])
    series = pd.DataFrame([{k: v for k, v in r.items() if k != 'points'} for r in rows.values()])
    series = series.merge(train.rename(columns={'SeriesInstanceUID': 'uid'}), on='uid', validate='one_to_one')
    series = series.sort_values('uid').reset_index(drop=True)
    series['file'] = 'vol/' + series.uid + '.pkl'
    series['terr_ok'] = series.seg_method.eq('stack') if 'seg_method' in series else False
    series['empty_vol'] = series.fg_frac < config.EMPTY_FRAC
    with ThreadPoolExecutor(16) as pool:
        series['patient'] = list(pool.map(patient_id, [series_root / u for u in series.uid]))
    groups = series.patient if series.patient.duplicated().any() else None
    series['fold'] = make_folds(series, N_FOLDS, FOLD_SEED, groups)

    series.to_parquet(out_dir / 'series.parquet', index=False)
    points.to_parquet(out_dir / 'points.parquet', index=False)
    with open(out_dir / 'prep_config.json', 'w') as f:
        json.dump({k: v for k, v in cfg.items() if k != 'ps_table'}, f, indent=1)
    print(f'{len(series)} series and {len(points)} aneurysm points written to {out_dir}')


if __name__ == '__main__':
    main()
