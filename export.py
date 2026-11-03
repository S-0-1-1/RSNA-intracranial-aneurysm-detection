import shutil
import subprocess
import sys
from pathlib import Path

from aneurysm.cube import hdbet_weights

PREP_DIR = Path('data/prepared')
RUNS_DIR = Path('runs')
OUT_DIR = Path('kaggle_dataset')
FOLDS = [0, 1, 2, 3, 4]
PACKAGES = [
    'hd-bet==2.0.1', 'nnunetv2==2.8.1', 'acvl-utils==0.2.6', 'dynamic-network-architectures==0.4.4',
    'batchgenerators==0.25.5', 'batchgeneratorsv2==0.3.6', 'yacs', 'connected-components-3d', 'timm<1.0.23', 'einops',
    'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'pylibjpeg-rle', 'imagecodecs', 'zstandard', 'SimpleITK', 'nibabel',
]


def main():
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        shutil.copy(RUNS_DIR / f'fold{fold}' / 'model.pt', OUT_DIR / 'models' / f'fold{fold}.pt')
    shutil.copy(PREP_DIR / 'prep_config.json', OUT_DIR / 'prep_config.json')
    shutil.copy(PREP_DIR / 'ps_table.json', OUT_DIR / 'ps_table.json')
    shutil.copytree(hdbet_weights(), OUT_DIR / 'hdbet', dirs_exist_ok=True)
    shutil.copytree('aneurysm', OUT_DIR / 'aneurysm', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy('submit.py', OUT_DIR / 'submit.py')
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '-q', '--no-deps', '-w', str(OUT_DIR / 'wheels'), *PACKAGES], check=True)


if __name__ == '__main__':
    main()
