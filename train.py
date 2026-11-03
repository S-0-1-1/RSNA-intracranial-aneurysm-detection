import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

from aneurysm import config
from aneurysm.dataset import BatchBuilder, PatchStream, Tables, VolumeSet, collate_list
from aneurysm.inference import predict_volume
from aneurysm.labels import LABELS
from aneurysm.losses import compute_losses
from aneurysm.metric import competition_metric
from aneurysm.model import build_model, param_groups

FOLD = 0
PREP_DIR = Path('data/prepared')
OUT_DIR = Path('runs') / f'fold{FOLD}'
PRETRAINED = 'MIC-DKFZ/nnFoundationCNN'
TOTAL_STEPS = 36000
WARMUP = 1500
BATCH = 2
LR = 1e-3
LR_NEW_MULT = 3.0
MOMENTUM = 0.99
WEIGHT_DECAY = 3e-5
GRAD_CLIP = 12.0
VAL_STRIDE = 96
ZONE_MARGIN = 24
SAVE_EVERY = 1000
LOG_EVERY = 200
WORKERS = max(1, min(3, (os.cpu_count() - 1) // 2))


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR * (1.0 - min(1.0, (step - WARMUP) / (TOTAL_STEPS - WARMUP))) ** 0.9


def inference_zone(tables, cube):
    xyz = tables.pt_xyz[tables.pos_idx]
    lo = np.floor(np.percentile(xyz, 0.5, axis=0)) - ZONE_MARGIN
    hi = np.ceil(np.percentile(xyz, 99.5, axis=0)) + ZONE_MARGIN
    return [[int(max(0, a)), int(min(cube - 1, b))] for a, b in zip(lo, hi)]


def download_pretrained():
    return str(Path(snapshot_download(PRETRAINED, allow_patterns=['*.pth', '*.json'])) / 'checkpoint_final.pth')


def save(obj, path):
    torch.save(obj, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def validate(model, frame, prior, cfg, zone, device, rank, world):
    model.eval()
    live = frame[~frame.empty_vol].iloc[rank::world]
    data = VolumeSet(live.uid, live.path, cfg['strip_inf_mm'], cfg['dist_unit_mm'])
    loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=2, pin_memory=device.type == 'cuda')
    rows = []
    for uid, img, mu, sd in loader:
        probs = predict_volume(model, img, mu, sd, zone, cfg, VAL_STRIDE, device, tta=True)
        rows.append(dict(uid=uid, **dict(zip(LABELS, probs.tolist()))))
    if world > 1:
        parts = [None] * world
        dist.all_gather_object(parts, rows)
        rows = [r for part in parts for r in part]
    if rank != 0:
        return
    rows += [dict(uid=u, **prior) for u in frame.uid[frame.empty_vol]]
    pred = pd.DataFrame(rows)
    truth = frame.set_index('uid').loc[pred.uid]
    score, aucs = competition_metric(truth[LABELS].to_numpy(), pred[LABELS].to_numpy())
    pred.to_parquet(OUT_DIR / 'oof.parquet', index=False)
    with open(OUT_DIR / 'val.json', 'w') as f:
        json.dump(dict(score=score, auc=dict(zip(LABELS, aucs.tolist()))), f, indent=1)
    print(f'fold {FOLD} validation score {score:.4f} (aneurysm present AUC {aucs[-1]:.4f})', flush=True)


def main():
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', local_rank) if cuda else torch.device('cpu')
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config.train_cfg()

    series = pd.read_parquet(PREP_DIR / 'series.parquet')
    series['path'] = [str(PREP_DIR / f) for f in series.file]
    points = pd.read_parquet(PREP_DIR / 'points.parquet')
    train_df = series[(series.fold != FOLD) & ~series.empty_vol].reset_index(drop=True)
    val_df = series[series.fold == FOLD].reset_index(drop=True)
    tables = Tables(train_df, points, cfg)
    zone = inference_zone(tables, cfg['cube'])
    prior = {c: float(train_df[c].mean()) for c in LABELS}

    state_path = OUT_DIR / 'train_state.pth'
    state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else None
    pretrained = download_pretrained() if state is None and PRETRAINED and rank == 0 else None
    model = build_model(pretrained, cfg)
    if state is not None:
        model.load_state_dict(state['model'])
    model.to(device)
    optimizer = torch.optim.SGD(param_groups(model, WEIGHT_DECAY), lr=LR, momentum=MOMENTUM, nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type, enabled=cuda)
    step = 0
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        if cuda:
            scaler.load_state_dict(state['scaler'])
        step = state['step']
        del state

    net = model
    if world > 1:
        net = DDP(model, device_ids=[local_rank] if cuda else None, gradient_as_bucket_view=True)
        if cuda:
            net.register_comm_hook(None, default_hooks.fp16_compress_hook)
    loader = torch.utils.data.DataLoader(PatchStream(tables, cfg, seed=[FOLD, rank, step]), batch_size=BATCH,
                                         num_workers=WORKERS, collate_fn=collate_list, pin_memory=cuda,
                                         persistent_workers=True)
    builder = BatchBuilder(cfg, device)
    batches = iter(loader)
    model.train()
    total, count, bad, t0 = 0.0, 0, 0, time.time()
    while step < TOTAL_STEPS:
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group['lr'] = lr * (LR_NEW_MULT if group['name'] == 'new' else 1.0)
        target = builder(next(batches))
        with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
            out = net(target['x'], target['coords'])
        loss = compute_losses(out, target, cfg)
        if torch.isfinite(loss):
            bad = 0
        else:
            bad += 1
            if bad > 100:
                raise RuntimeError(f'loss is not finite for {bad} steps in a row at step {step}')
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        total += float(loss.detach())
        count += 1
        if rank == 0 and step % LOG_EVERY == 0:
            print(f'step {step}/{TOTAL_STEPS} loss {total / count:.4f} lr {lr:.2e} {(time.time() - t0) / count:.2f}s/step', flush=True)
            total, count, t0 = 0.0, 0, time.time()
        if rank == 0 and (step % SAVE_EVERY == 0 or step == TOTAL_STEPS):
            save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step), state_path)
    del batches, loader

    if rank == 0:
        weights = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
        save(dict(model=weights, train_cfg=cfg, zone=zone, stride=VAL_STRIDE, fold=FOLD, step=step), OUT_DIR / 'model.pt')
    validate(model, val_df, prior, cfg, zone, device, rank, world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
