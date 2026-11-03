import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
import torch

from aneurysm import config, dicom
from aneurysm.inference import Predictor
from aneurysm.labels import LABELS

HERE = Path(__file__).resolve().parent
TIME_LIMIT_H = 10.5
RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
START = time.time()


def find_competition(root='/kaggle/input'):
    for dirpath, dirnames, filenames in os.walk(root):
        if 'train_localizers.csv' in filenames:
            return Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ('series', 'segmentations')]
    raise FileNotFoundError(f'competition data not found under {root}')


COMP = find_competition()
if importlib.util.find_spec('kaggle_evaluation') is None:
    sys.path.append(str(COMP))
import kaggle_evaluation.rsna_inference_server

prior = pd.read_csv(COMP / 'train.csv')[LABELS].mean().to_numpy()
with open(HERE / 'prep_config.json') as f:
    prep_cfg = config.prep_cfg(**json.load(f))
prep_cfg['ps_table'] = dicom.load_table(HERE / 'ps_table.json')
devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
predictor = Predictor(sorted((HERE / 'models').glob('*.pt')), devices, HERE / 'hdbet', prep_cfg)


def predict(series_path):
    probs = None
    try:
        if time.time() - START < TIME_LIMIT_H * 3600:
            probs = predictor(series_path)
    except Exception:
        if not RERUN:
            raise
        torch.cuda.empty_cache()
    if str(series_path).startswith('/kaggle/shared/'):
        shutil.rmtree(series_path, ignore_errors=True)
    if probs is None:
        probs = prior
    return pl.DataFrame([probs.tolist()], schema=LABELS, orient='row')


server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
if RERUN:
    server.serve()
else:
    server.run_local_gateway()
