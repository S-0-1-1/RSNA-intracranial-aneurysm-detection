# RSNA Intracranial Aneurysm Detection

My solution for the [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) competition on Kaggle: a 3D U-Net that finds aneurysms in CTA, MRA and MRI series and assigns them to one of 13 arterial locations.

| | score |
|---|---|
| 5-fold CV (out of fold, 4348 series) | 0.8997 |
| Public leaderboard | 0.90237 |
| Private leaderboard | 0.88294 |

Leaderboard scores are from a late submission.

## Method

**Preprocessing.** Each series is turned into a 256³ cube around the brain.

- The main image stack is rebuilt from the DICOM geometry, using only the tags that are kept in the test set. Missing pixel spacing of multi-frame files is taken from a table built on the training data. CT series with a wrong rescale intercept are detected and fixed.
- The head is located on a coarse 4 mm overview and HD-BET gives the brain mask inside it.
- The cube covers the brain plus a margin (140 to 210 mm, so 0.55 to 0.82 mm voxels) and follows the patient axes.
- CT is windowed to [-100, 900] HU, MR is scaled between the 0.5 and 99.95 percentiles inside the brain, then stored as uint8. Voxels more than 20 mm from the brain are set to zero and the distance to the brain is stored with the image.
- The vessel segmentations that come with part of the training data are resampled into the cube and used as territory labels.

**Model.** U-Net with the ResEnc-L encoder of [nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) (self-supervised pretraining on unlabeled 3D scans) and a new decoder with CBAM attention on the skip connections. It works on 128³ patches and outputs at half resolution:

- an aneurysm heatmap (Gaussian targets with sigma 3 mm, focal loss),
- a vessel territory map with 14 classes (13 locations and background),
- an auxiliary heatmap at quarter resolution and the modality.

A small network on the voxel coordinates adds a position prior to the heatmap and territory logits. Series probabilities are read from the 8 strongest voxels: P(aneurysm) is the mean of the top 8 heatmap probabilities, P(location) is the same on heatmap x territory probability. These readouts get their own BCE loss during training.

**Training.** 5 folds stratified on locations and modality, grouped by patient. SGD with Nesterov momentum 0.99, learning rate 1e-3 (x3 for the new layers), 1500 warmup steps then poly decay, 36k steps with a batch of 2 per GPU on 2x T4, about 10 hours per fold. Half of the patches contain an annotated aneurysm, sampled so that rare locations and modalities are not drowned out; most of the other patches are taken around typical aneurysm positions. Augmentations: left/right flip with location swap, rotation, scaling, random brain margin, bias field, brightness, contrast, gamma, low resolution and noise.

**Inference.** Same preprocessing on the test series, sliding window over the part of the cube where aneurysms are found in training, average of the 5 folds.

## Setup

```
pip install -r requirements.txt
```

Download the competition data into `data/rsna-intracranial-aneurysm-detection` (or change `DATA_DIR` at the top of `preprocess.py`).

## 1. Preprocessing

```
python preprocess.py
```

It runs one process per GPU. If it stops, run it again: finished series are skipped. The output goes to `data/prepared` (about 24 GB):

- `vol/` one file per series with the cube, the distance map and the territory labels
- `series.parquet` labels, folds and preprocessing details for each series
- `points.parquet` aneurysm positions in cube coordinates
- `prep_config.json` and `ps_table.json`, needed again at inference

## 2. Training

Train the folds one by one. Set the fold at the top of `train.py`:

```python
FOLD = 0
```

and run

```
torchrun --standalone --nproc_per_node=2 train.py
```

Repeat with `FOLD = 1`, `2`, `3` and `4`. On Kaggle 2x T4 put `NCCL_P2P_DISABLE=1` in front of the command.

The pretrained weights are downloaded from Hugging Face on the first run. The training state is saved every 1000 steps in `runs/fold0/train_state.pth` and the same command resumes from it. At the end `runs/fold0/model.pt` is written and the held-out fold is evaluated (`val.json`, `oof.parquet`).

## 3. Kaggle submission

```
python export.py
```

This puts the 5 models, the HD-BET weights, the preprocessing files, the code and the wheels of the packages missing on Kaggle into `kaggle_dataset/`. Some of the wheels are compiled, so run it with the same Python version as the Kaggle image (a Kaggle notebook with internet on works). Upload the folder as a Kaggle dataset, then create a notebook with GPU T4 x2, internet off, the competition data and the dataset attached, and run:

```
!pip install -q --no-index --no-deps /kaggle/input/aneurysm-models/wheels/*.whl
%run /kaggle/input/aneurysm-models/submit.py
```

(use the path where Kaggle mounts your dataset). The hidden test set takes about 8 to 9 hours. Series that are still left after 10.5 hours, or that fail, get the training prevalence.

## Code

```
aneurysm/
  dicom.py          reading series, stack geometry, CT intercept check
  cube.py           resampling, HD-BET brain mask, cube and distance map
  segmentation.py   vessel labels to cube
  storage.py        compressed volume files
  dataset.py        folds, patch sampling, augmentations, targets
  model.py          encoder, CBAM decoder, heads, readout
  losses.py
  metric.py         competition metric
  inference.py      sliding window and the series predictor
  prep_worker.py    preprocessing on one GPU
preprocess.py
train.py
export.py
submit.py
```
