from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
RAW_DIR = Path("data/raw")
EXTRACT_DIR = RAW_DIR / "UCI HAR Dataset"

SIGNAL_FILES = (
    "body_acc_x_{split}.txt", "body_acc_y_{split}.txt", "body_acc_z_{split}.txt",
    "body_gyro_x_{split}.txt", "body_gyro_y_{split}.txt", "body_gyro_z_{split}.txt",
    "total_acc_x_{split}.txt", "total_acc_y_{split}.txt", "total_acc_z_{split}.txt",
)

def _extract_nested_archive(archive: Path) -> None:
    with zipfile.ZipFile(archive) as outer:
        outer.extractall(RAW_DIR)
        nested = next((name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip")), None)
    if nested is not None:
        with zipfile.ZipFile(RAW_DIR / nested) as inner:
            inner.extractall(RAW_DIR)

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    required = EXTRACT_DIR / "train" / "Inertial Signals" / "body_acc_x_train.txt"
    if not required.exists():
        archive = RAW_DIR / "uci_har.zip"
        print("Downloading UCI HAR dataset...")
        urllib.request.urlretrieve(DATA_URL, archive)
        print("Extracting UCI HAR dataset...")
        _extract_nested_archive(archive)
    if not required.exists():
        raise FileNotFoundError(f"UCI HAR Inertial Signals were not extracted correctly. Expected: {required}")
    return EXTRACT_DIR

def _load_split(root: Path, split: str):
    signal_dir = root / split / "Inertial Signals"
    channels = [np.loadtxt(signal_dir / filename.format(split=split), dtype=np.float32) for filename in SIGNAL_FILES]
    x = np.stack(channels, axis=-1)
    if x.ndim != 3 or x.shape[1:] != (128, 9):
        raise ValueError(f"Unexpected {split} signal shape: {x.shape}; expected (n, 128, 9)")
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(subjects)

def load_data():
    root = download_dataset()
    return _load_split(root, "train"), _load_split(root, "test")

def make_loader(x, y, batch_size=64, shuffle=True):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)
