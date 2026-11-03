# RSNA Intracranial Aneurysm Detection

My solution for the [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) competition on Kaggle. A 3D U-Net finds aneurysms in CTA, MRA and MRI series and assigns them to one of 13 arterial locations.

This is a late submission, made after the competition ended. It scores 0.88294 on the private leaderboard and 0.90237 on the public leaderboard. The 5-fold cross-validation score is 0.8997.

![Kaggle late submission](kaggle_submission.png)

## Method

- Preprocessing: each series is resampled into a 256³ cube around the brain, found with an HD-BET brain mask. CT and MR intensities are normalized and everything far from the brain is removed.
- Model: a 3D U-Net with the pretrained ResEnc-L encoder of [nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) and a decoder with CBAM attention. It predicts an aneurysm heatmap and a map of the 13 vessel locations, and the series probabilities are read from the strongest voxels.
- Training: 5 folds on 128³ patches, about 10 hours per fold on 2x T4.
- Inference: same preprocessing, sliding window, average of the 5 folds.

## Setup

```
pip install -r requirements.txt
```

Put the competition data in `data/rsna-intracranial-aneurysm-detection`.

## 1. Preprocessing

```
python preprocess.py
```

The output goes to `data/prepared` (about 24 GB). If it stops, run it again and it continues where it stopped.

## 2. Training

Set the fold at the top of `train.py`:

```python
FOLD = 0
```

and run

```
torchrun --standalone --nproc_per_node=2 train.py
```

Repeat with `FOLD = 1`, `2`, `3` and `4`. On Kaggle (2x T4), put `NCCL_P2P_DISABLE=1` in front of the command. The pretrained weights are downloaded on the first run, and each fold is saved in `runs/fold0`, `runs/fold1`, and so on.

## 3. Kaggle submission

```
python export.py
```

Run it in a Kaggle notebook with internet on, so the packages it collects match the Kaggle Python version. Upload the `kaggle_dataset/` folder as a Kaggle dataset. Then, in a notebook with GPU T4 x2, internet off, the competition data and this dataset attached, run:

```
!pip install -q --no-index --no-deps /kaggle/input/aneurysm-models/wheels/*.whl
%run /kaggle/input/aneurysm-models/submit.py
```

Change the path to where Kaggle mounts your dataset.
