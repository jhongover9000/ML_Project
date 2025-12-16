"""
BCIC IV 2a EEG Classification Pipeline
FBCSP (Classic), EEGNet (Deep), ATCNet (Deep)
Evaluation: Within-Subject, Leave-One-Subject-Out, K-Fold

Note: Used AI to clean up the code but it should still work the same
python eval_pipeline.py --model [model to use] --eval [which evaluation method to use] --device [if gpu exists]
"""
from __future__ import annotations

import argparse
import json
import random
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Signal Processing / EEG
from mne.filter import filter_data
from mne.decoding import CSP
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGNetv4, ATCNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.util import set_random_seeds

# ML / Scikit-Learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedShuffleSplit


# ==========================================
# Configuration & Constants
# ==========================================

DEFAULT_SEED = 9070
DATASET_NAME = "BNCI2014_001"
# Standard Motor Imagery bands
BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]

# Default Hyperparameters per model
DEFAULT_CONFIGS = {
    "fbcsp": {
        "k_best": 20,
        "csp_components": 2,
        "mi_offset": 0.5,
    },
    "eegnet": {
        "epochs": 60,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 10,
        "grad_clip": 0.0,
    },
    "atcnet": {
        "epochs": 80,
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-2,
        "patience": 12,
        "grad_clip": 1.0,
    },
}

@dataclass
class SubjectSessions:
    """Holds raw session data for a single subject."""
    sfreq: float
    sessions: Dict[str, Tuple[np.ndarray, np.ndarray]]  # session_name -> (X, y)

@dataclass
class DataSplits:
    """Holds processed train/val/test splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    sfreq: float
    labels: np.ndarray


# ==========================================
# Utilities
# ==========================================

def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Sets seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_random_seeds(seed=seed, cuda=torch.cuda.is_available())


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs broadcast-safe division with zero-protection."""
    return np.divide(
        a.astype(float),
        b.astype(float),
        out=np.zeros_like(a, dtype=float),
        where=(b != 0)
    )


def save_plot(fig: plt.Figure, path: Path) -> None:
    """Helper to save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], title: str, out_prefix: Path) -> None:
    """Plots Loss, Accuracy, and Balanced Accuracy curves."""
    epochs = history.get("epoch", [])
    if not epochs:
        return

    metrics = [
        ("loss", "Loss", ["train_loss"]),
        ("acc", "Accuracy", ["train_acc", "val_acc"]),
        ("balacc", "Balanced Accuracy", ["train_bal_acc", "val_bal_acc"]),
    ]

    for suffix, ylabel, keys in metrics:
        fig, ax = plt.subplots()
        for k in keys:
            if k in history:
                ax.plot(epochs, history[k], label=k)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({ylabel.lower()})")
        ax.legend()
        save_plot(fig, out_prefix.with_name(f"{out_prefix.name}_{suffix}.png"))


# ==========================================
# Data Processing
# ==========================================

def load_subject_sessions(
    subject_id: int,
    low_cut_hz: float,
    high_cut_hz: float,
    start_offset_s: float,
    stop_offset_s: float,
    n_jobs: int = -1,
) -> SubjectSessions:
    """
    Loads BNCI2014-001 data via MOABB, filters, and creates windows.
    """
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[subject_id])
    sfreq = float(dataset.datasets[0].raw.info["sfreq"])

    # Define Preprocessing Pipeline
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Drop EOG/Stim
        Preprocessor(lambda data: data * 1e6),  # Convert V to uV
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),
    ]
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    # Windowing
    windows = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(round(start_offset_s * sfreq)),
        trial_stop_offset_samples=int(round(stop_offset_s * sfreq)),
        preload=True,
    )

    # Split by session
    sessions_np = {}
    for sess_name, ds in windows.split("session").items():
        # Manual extraction to numpy
        xs, ys = [], []
        for i in range(len(ds)):
            x, y, _ = ds[i] # ds[i] returns (x, y, ind)
            xs.append(x)
            ys.append(y)
        sessions_np[str(sess_name)] = (np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64))

    return SubjectSessions(sfreq=sfreq, sessions=sessions_np)


def build_splits(
    cache: Dict[int, SubjectSessions],
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
    use_all_sessions: bool,
    val_frac: float,
    seed: int,
) -> DataSplits:
    """
    Concatenates data from multiple subjects/sessions and creates Train/Val/Test splits.
    """
    train_sessions = ["0train", "1test"] if use_all_sessions else ["0train"]
    test_sessions = ["0train", "1test"] if use_all_sessions else ["1test"]

    def _collect(sub_ids, session_names):
        Xs, ys = [], []
        for sid in sub_ids:
            sess_dict = cache[sid].sessions
            for sname in session_names:
                X, y = sess_dict[sname]
                Xs.append(X)
                ys.append(y)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_full_train, y_full_train = _collect(train_subjects, train_sessions)
    X_test, y_test = _collect(test_subjects, test_sessions)

    # Stratified Train/Val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(X_full_train, y_full_train))

    # Derive global labels
    labels = np.sort(np.unique(np.concatenate([y_full_train, y_test])))
    sfreq = next(iter(cache.values())).sfreq

    return DataSplits(
        X_train=X_full_train[tr_idx], y_train=y_full_train[tr_idx],
        X_val=X_full_train[va_idx], y_val=y_full_train[va_idx],
        X_test=X_test, y_test=y_test,
        sfreq=sfreq, labels=labels
    )


# ==========================================
# Models: Classic (FBCSP)
# ==========================================

class FBCSP_OVR:
    """Multiclass FBCSP using One-vs-Rest CSP per frequency band."""
    def __init__(self, sfreq: float, bands: List[Tuple[float, float]], n_components: int = 2):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.models_: Dict[Tuple[int, int], CSP] = {}
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        X = np.ascontiguousarray(X, dtype=np.float64)
        for bi, (fmin, fmax) in enumerate(self.bands):
            Xb = filter_data(X, self.sfreq, fmin, fmax, verbose=False)
            for c in self.classes_:
                # Binary targets for OVR
                y_bin = (y == c).astype(int)
                csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, cov_est="epoch")
                csp.fit(Xb, y_bin)
                self.models_[(bi, int(c))] = csp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        features = []
        for bi, (fmin, fmax) in enumerate(self.bands):
            Xb = filter_data(X, self.sfreq, fmin, fmax, verbose=False)
            for c in self.classes_:
                features.append(self.models_[(bi, int(c))].transform(Xb))
        return np.concatenate(features, axis=1)


def run_fbcsp_pipeline(splits: DataSplits, config: Dict) -> Dict[str, Any]:
    """Runs the FBCSP + LDA pipeline."""
    # Crop start offset if needed for motor imagery calculation
    start_sample = int(round(config["mi_offset"] * splits.sfreq)) if config.get("mi_offset") else 0
    
    def crop(X):
        return X[:, :, start_sample:] if start_sample > 0 else X

    Xtr, Xva, Xte = crop(splits.X_train), crop(splits.X_val), crop(splits.X_test)

    # Feature Extraction
    fbcsp = FBCSP_OVR(splits.sfreq, BANDS, n_components=config["csp_components"])
    fbcsp.fit(Xtr, splits.y_train)
    
    Ftr = fbcsp.transform(Xtr)
    Fva = fbcsp.transform(Xva)
    Fte = fbcsp.transform(Xte)

    # Feature Selection
    k = min(config["k_best"], Ftr.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    Ftr_s = selector.fit_transform(Ftr, splits.y_train)
    Fva_s = selector.transform(Fva)
    Fte_s = selector.transform(Fte)

    # Classifier
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(Ftr_s, splits.y_train)

    yva_hat = clf.predict(Fva_s)
    yte_hat = clf.predict(Fte_s)

    return {
        "val_acc": accuracy_score(splits.y_val, yva_hat),
        "val_bal_acc": balanced_accuracy_score(splits.y_val, yva_hat),
        "test_acc": accuracy_score(splits.y_test, yte_hat),
        "test_bal_acc": balanced_accuracy_score(splits.y_test, yte_hat),
        "test_cm": confusion_matrix(splits.y_test, yte_hat, labels=splits.labels),
    }


# ==========================================
# Models: Deep Learning
# ==========================================

def build_torch_model(name: str, n_chans: int, n_times: int, n_classes: int, sfreq: float) -> nn.Module:
    if name == "eegnet":
        return EEGNetv4(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    if name == "atcnet":
        return ATCNet(n_chans=n_chans, n_outputs=n_classes, 
                      input_window_seconds=n_times/sfreq, sfreq=sfreq)
    raise ValueError(f"Unknown model: {name}")


class DeepTrainer:
    """Encapsulates training logic, validation, and early stopping."""
    
    def __init__(self, model: nn.Module, device: str, config: Dict):
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer selection based on model nuance
        if isinstance(model, ATCNet):
            self.opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        else:
            self.opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses = []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(X), y)
            loss.backward()
            
            if self.cfg.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            
            self.opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        ys, preds = [], []
        for X, y in loader:
            X = X.to(self.device)
            logits = self.model(X)
            ys.append(y.numpy())
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            
        y_true = np.concatenate(ys)
        y_hat = np.concatenate(preds)
        return (
            float(accuracy_score(y_true, y_hat)), 
            float(balanced_accuracy_score(y_true, y_hat)),
            y_true,
            y_hat
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        history = {k: [] for k in ["epoch", "train_loss", "train_acc", "train_bal_acc", "val_acc", "val_bal_acc"]}
        best_val_acc = -1.0
        best_state = None
        patience_counter = 0
        
        print(f"    Starting training for {self.cfg['epochs']} epochs...")
        for ep in range(1, self.cfg["epochs"] + 1):
            loss = self.train_epoch(train_loader)
            tr_acc, tr_bal, _, _ = self.evaluate(train_loader)
            val_acc, val_bal, _, _ = self.evaluate(val_loader)
            
            # Record
            history["epoch"].append(ep)
            history["train_loss"].append(loss)
            history["train_acc"].append(tr_acc)
            history["train_bal_acc"].append(tr_bal)
            history["val_acc"].append(val_acc)
            history["val_bal_acc"].append(val_bal)
            
            # Checkpoint
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            print(f"\r    Ep {ep:03d}: Loss={loss:.4f} TrAcc={tr_acc:.3f} ValAcc={val_acc:.3f} (Best={best_val_acc:.3f})", end="")
            
            if patience_counter >= self.cfg["patience"]:
                print(f"\n    Early stopping at epoch {ep}")
                break
        print("") # newline
        
        if best_state:
            self.model.load_state_dict(best_state)
            
        return {
            "best_val_acc": best_val_acc,
            "best_val_bal_acc": history["val_bal_acc"][history["val_acc"].index(max(history["val_acc"]))],
            "history": history
        }


def run_deep_pipeline(splits: DataSplits, model_name: str, device: str, config: Dict) -> Dict[str, Any]:
    """Sets up data loaders and runs the DeepTrainer."""
    # Create Datasets/Loaders
    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle)

    train_loader = to_loader(splits.X_train, splits.y_train, shuffle=True)
    val_loader = to_loader(splits.X_val, splits.y_val)
    test_loader = to_loader(splits.X_test, splits.y_test)

    # Build Model
    model = build_torch_model(
        model_name, 
        n_chans=splits.X_train.shape[1], 
        n_times=splits.X_train.shape[2], 
        n_classes=len(splits.labels), 
        sfreq=splits.sfreq
    )

    # Train
    trainer = DeepTrainer(model, device, config)
    res = trainer.fit(train_loader, val_loader)

    # Final Test
    test_acc, test_bal, y_true, y_hat = trainer.evaluate(test_loader)
    
    return {
        **res,
        "test_acc": test_acc,
        "test_bal_acc": test_bal,
        "test_cm": confusion_matrix(y_true, y_hat, labels=splits.labels)
    }


# ==========================================
# Main Execution
# ==========================================

def get_folds(subjects: Sequence[int], mode: str, k: int, seed: int) -> List[Tuple[str, List[int], List[int]]]:
    """Generates fold tuples: (name, train_subjects, test_subjects)."""
    subjects = np.array(subjects, dtype=int)
    
    if mode == "within":
        return [(f"subj{s}", [s], [s]) for s in subjects]
    
    if mode == "loso":
        return [(f"holdout{s}", subjects[subjects != s].tolist(), [s]) for s in subjects]
    
    if mode == "kfold":
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []
        for i, (tr_idx, te_idx) in enumerate(kf.split(subjects), 1):
            folds.append((f"fold{i}", subjects[tr_idx].tolist(), subjects[te_idx].tolist()))
        return folds
    
    raise ValueError(f"Unknown eval mode: {mode}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", choices=["fbcsp", "eegnet", "atcnet"], required=True)
    p.add_argument("--eval", choices=["within", "loso", "kfold"], default="within")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--use_all_sessions", action="store_true", help="Use both train and test sessions for training")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_root", type=str, default="results_bcic2a")
    
    # Override defaults
    p.add_argument("--epochs", type=int, help="Override default epochs")
    p.add_argument("--lr", type=float, help="Override default learning rate")

    args = p.parse_args()
    seed_everything(DEFAULT_SEED)

    # Merge CLI args into config
    model_config = DEFAULT_CONFIGS[args.model].copy()
    if args.epochs: model_config["epochs"] = args.epochs
    if args.lr: model_config["lr"] = args.lr

    # Setup Paths
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.model}_{args.eval}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {args.model.upper()} | Eval: {args.eval} | Output: {out_dir}")
    print(f"Config: {json.dumps(model_config, indent=2)}")

    # Load Data
    print("Loading Subject Data...")
    data_cache = {}
    for sid in args.subjects:
        data_cache[sid] = load_subject_sessions(
            sid, low_cut_hz=4.0, high_cut_hz=38.0, 
            start_offset_s=-0.5, stop_offset_s=0.0
        )

    # Run Folds
    folds = get_folds(args.subjects, args.eval, args.folds, DEFAULT_SEED)
    fold_results = []
    sum_cm = None
    
    
    for fold_name, train_sub, test_sub in folds:
        print(f"\n=== {fold_name} | Train: {len(train_sub)} sub | Test: {len(test_sub)} sub ===")
        
        # Prepare Splits
        # For 'within', we treat the single subject as both train and test list to the builder
        builder_train = [test_sub[0]] if args.eval == "within" else train_sub
        builder_test  = [test_sub[0]] if args.eval == "within" else test_sub
        
        splits = build_splits(
            data_cache, builder_train, builder_test, 
            args.use_all_sessions, val_frac=0.2, seed=DEFAULT_SEED
        )

        # Execute Model
        if args.model == "fbcsp":
            res = run_fbcsp_pipeline(splits, model_config)
        else:
            res = run_deep_pipeline(splits, args.model, args.device, model_config)
            # Plot deep learning curves
            if "history" in res:
                plot_training_curves(res["history"], f"{args.model} {fold_name}", out_dir / f"train_{fold_name}")
                np.save(out_dir / f"hist_{fold_name}.npy", res["history"])

        # Aggregate Results
        cm = res["test_cm"]
        sum_cm = cm if sum_cm is None else sum_cm + cm
        
        row = {
            "fold": fold_name,
            "test_acc": round(res["test_acc"], 4),
            "test_bal": round(res["test_bal_acc"], 4),
        }
        fold_results.append(row)
        print(f"    Result: {row}")

    # Final Summary
    avg_acc = np.mean([r["test_acc"] for r in fold_results])
    print(f"\nAverage Test Accuracy: {avg_acc:.4f}")
    
    # Save Results
    np.save(out_dir / "cm_sum.npy", sum_cm)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "config": model_config,
            "folds": fold_results,
            "avg_test_acc": avg_acc
        }, f, indent=2)


if __name__ == "__main__":
    main()