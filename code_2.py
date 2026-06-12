# -*- coding: utf-8 -*-

import copy
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

REGRESSION_OPTIM_CONFIG: Dict[str, object] = {
    "existence_dropout": 0.70,
    "existence_hidden_ratio": 0.30,
    "existence_layers": 2,
    "existence_lr": 1e-4,
    "initial_existence_weight": 0.60,
    "max_existence_weight": 2.50,
    "min_existence_weight": 0.40,
    "grad_clip_existence": 0.50,
    "existence_weight_decay": 8e-4,

    "regression_dropout": 0.06,
    "regression_hidden_layers": 4,
    "regression_lr": 1.2e-3,
    "initial_regression_weight": 4.00,
    "max_regression_weight": 8.00,
    "min_regression_weight": 1.20,
    "grad_clip_regression": 2.50,
    "regression_weight_decay": 1e-5,

    "label_loss_weights": [1.0, 1.0, 1.0, 4.0, 1.0],
    "label4_index": 3,

    "conv_filters": 32,
    "fc_units": 160,
    "shared_lr": 8e-4,
    "shared_weight_decay": 1e-4,
    "shared_grad_clip": 1.0,

    "batch_size": 32,
    "augment_noise_std": 0.003,
    "regression_focus_epochs": 45,
    "regression_focus_multiplier": 1.25,
    "patience": 50,
    "lr_patience": 10,
    "lr_factor": 0.70,
    "min_r2_threshold": -1.0,
    "huber_beta": 1.0,

    "early_stop_min_delta": 1e-4,
    "score_exist_weight": 0.03,
    "score_reg_loss_weight": 0.01,
    "fold_score_label4_weight": 0.85,
    "fold_score_overall_weight": 0.15,
    "existence_thresholds": [0.50, 0.50, 0.50, 0.40, 0.40],
}

RUN_CONFIG = {
    # choose "train" or "external"
    "mode": "train",

    "eem_file": "reduce_data.csv",
    "loading_file": "reduce_loadings.csv",
    "label_file": "FMAX.csv",

    "external_eem_file": "predict_all_data_processed.csv",
    "external_loading_file": "predict_all_ture_loadings_processed.csv",
    "external_label_file": "external_FMAX.csv",
    "external_output_file": "ture_external_label4_3.csv",

    "model_file": "",

    "output_dir": "outputs_data33_best_label4_external_generalized",
    "n_splits": 5,
    "epochs": 180,
    "num_workers": 16,
    "patience": int(REGRESSION_OPTIM_CONFIG["patience"]),
    "lr_patience": int(REGRESSION_OPTIM_CONFIG["lr_patience"]),
    "seed": 42,
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class NonZeroLabelStandardizer:

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, labels: np.ndarray) -> "NonZeroLabelStandardizer":
        labels = np.asarray(labels, dtype=np.float32)
        n_labels = labels.shape[1]
        mean = np.zeros(n_labels, dtype=np.float32)
        scale = np.ones(n_labels, dtype=np.float32)
        for i in range(n_labels):
            mask = labels[:, i] != 0
            vals = labels[mask, i]
            if len(vals) > 1 and np.std(vals) > self.eps:
                mean[i] = float(np.mean(vals))
                scale[i] = float(np.std(vals))
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("NonZeroLabelStandardizer must be fitted before transform.")
        labels = np.asarray(labels, dtype=np.float32).copy()
        for i in range(labels.shape[1]):
            mask = labels[:, i] != 0
            labels[mask, i] = (labels[mask, i] - self.mean_[i]) / self.scale_[i]
        return labels

    def inverse_transform(self, labels_std: np.ndarray, existence_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("NonZeroLabelStandardizer must be fitted before inverse_transform.")
        labels_std = np.asarray(labels_std, dtype=np.float32)
        out = labels_std * self.scale_.reshape(1, -1) + self.mean_.reshape(1, -1)
        if existence_mask is not None:
            out = out * existence_mask
        return out.astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean_, "scale": self.scale_}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "NonZeroLabelStandardizer":
        obj = cls()
        obj.mean_ = np.asarray(state["mean"], dtype=np.float32)
        obj.scale_ = np.asarray(state["scale"], dtype=np.float32)
        return obj


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.var(y_true) <= 1e-8:
        return 0.0
    return float(r2_score(y_true, y_pred))


def calculate_robust_r2(regression_pred: np.ndarray,
                        regression_target: np.ndarray,
                        existence_target: np.ndarray,
                        min_valid: int = 5) -> float:
    r2_scores: List[float] = []
    valid_samples: List[int] = []
    for i in range(regression_target.shape[1]):
        mask = existence_target[:, i] == 1
        n = int(mask.sum())
        valid_samples.append(n)
        if n <= min_valid:
            r2_scores.append(0.0)
            continue
        y_true = regression_target[mask, i]
        y_pred = regression_pred[mask, i]
        if np.var(y_true) <= 1e-8:
            r2_scores.append(0.0)
            continue
        q1, q3 = np.percentile(y_true, [25, 75])
        iqr = q3 - q1
        if iqr > 1e-8:
            keep = (y_true >= q1 - 1.5 * iqr) & (y_true <= q3 + 1.5 * iqr)
            if keep.sum() > min_valid and np.var(y_true[keep]) > 1e-8:
                y_true = y_true[keep]
                y_pred = y_pred[keep]
        r2_scores.append(safe_r2(y_true, y_pred))
    valid_idx = [i for i, n in enumerate(valid_samples) if n > min_valid]
    if not valid_idx:
        return 0.0
    weights = np.array([valid_samples[i] for i in valid_idx], dtype=np.float32)
    values = np.array([r2_scores[i] for i in valid_idx], dtype=np.float32)
    return float(np.average(values, weights=weights))


def per_label_metrics(pred_raw: np.ndarray,
                      target_raw: np.ndarray,
                      existence_target: np.ndarray,
                      prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_labels = target_raw.shape[1]
    for i in range(n_labels):
        mask = existence_target[:, i] == 1
        n = int(mask.sum())
        name = f"{prefix}label{i + 1}"
        out[f"{name}_n"] = n
        if n > 1 and np.var(target_raw[mask, i]) > 1e-8:
            y_true = target_raw[mask, i]
            y_pred = pred_raw[mask, i]
            out[f"{name}_r2"] = safe_r2(y_true, y_pred)
            out[f"{name}_mae"] = float(mean_absolute_error(y_true, y_pred))
            out[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        else:
            out[f"{name}_r2"] = 0.0
            out[f"{name}_mae"] = 0.0
            out[f"{name}_rmse"] = 0.0
    return out


def regression_metrics(pred_std: np.ndarray,
                       target_std: np.ndarray,
                       existence_target: np.ndarray,
                       pred_raw_gated: Optional[np.ndarray] = None,
                       pred_raw_ungated: Optional[np.ndarray] = None,
                       target_raw: Optional[np.ndarray] = None) -> Dict[str, float]:
    pred_std = np.asarray(pred_std)
    target_std = np.asarray(target_std)
    existence_target = np.asarray(existence_target)
    final_pred_std = pred_std * existence_target
    metrics = {
        "regression_r2_std": calculate_robust_r2(pred_std, target_std, existence_target),
        "final_r2_std": safe_r2(target_std.flatten(), final_pred_std.flatten()),
        "mse_std": float(mean_squared_error(target_std.flatten(), final_pred_std.flatten())),
        "mae_std": float(mean_absolute_error(target_std.flatten(), final_pred_std.flatten())),
    }
    if pred_raw_gated is not None and pred_raw_ungated is not None and target_raw is not None:
        metrics.update({
            "regression_r2_raw_gated": calculate_robust_r2(pred_raw_gated, target_raw, existence_target),
            "regression_r2_raw_ungated": calculate_robust_r2(pred_raw_ungated, target_raw, existence_target),

            "final_r2_raw_gated": safe_r2(target_raw.flatten(), pred_raw_gated.flatten()),
            "final_r2_raw_ungated": safe_r2(target_raw.flatten(), pred_raw_ungated.flatten()),
            "mse_raw_gated": float(mean_squared_error(target_raw.flatten(), pred_raw_gated.flatten())),
            "mae_raw_gated": float(mean_absolute_error(target_raw.flatten(), pred_raw_gated.flatten())),
            "mse_raw_ungated": float(mean_squared_error(target_raw.flatten(), pred_raw_ungated.flatten())),
            "mae_raw_ungated": float(mean_absolute_error(target_raw.flatten(), pred_raw_ungated.flatten())),
        })
        metrics.update(per_label_metrics(pred_raw_gated, target_raw, existence_target, prefix="gated_"))
        metrics.update(per_label_metrics(pred_raw_ungated, target_raw, existence_target, prefix="ungated_"))
    return metrics


def make_existence_predictions(logits: torch.Tensor, thresholds: List[float]) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    th = torch.tensor(thresholds, dtype=probs.dtype, device=probs.device).reshape(1, -1)
    return (probs > th).float()


# read data
def load_spectral_data_fixed_headers(eem_file: str,
                                     loading_file: str,
                                     label_file: str,
                                     excitation_dim: int = 51,
                                     emission_dim: int = 88) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    print(f"EEM={eem_file}, loading={loading_file}, label={label_file}")
    eem_df = pd.read_csv(eem_file, header=None)
    loading_df = pd.read_csv(loading_file, header=None)
    label_df = pd.read_csv(label_file, index_col=0)

    eem_data = eem_df.values.astype(np.float32)
    loading_scores = loading_df.values.astype(np.float32)
    labels = label_df.values.astype(np.float32)

    n = min(len(eem_data), len(loading_scores), len(labels))
    if len(eem_data) != len(loading_scores) or len(eem_data) != len(labels):
        eem_data = eem_data[:n]
        loading_scores = loading_scores[:n]
        labels = labels[:n]

    if eem_data.ndim == 2:
        if eem_data.shape[1] == excitation_dim * emission_dim:
            eem_shape = (excitation_dim, emission_dim)
        else:
            total_features = eem_data.shape[1]
            dims = []
            for i in range(1, int(np.sqrt(total_features)) + 1):
                if total_features % i == 0:
                    dims.append((i, total_features // i))
            if not dims:
                raise ValueError(f"Cannot reshape EEM feature count {total_features} into a two-dimensional matrix")
            eem_shape = min(dims, key=lambda x: abs(x[0] - excitation_dim) + abs(x[1] - emission_dim))
        eem_data = eem_data.reshape(len(eem_data), *eem_shape)
    else:
        eem_shape = (eem_data.shape[1], eem_data.shape[2])

    print(f" n={len(eem_data)}, EEM={eem_shape}, loading_dim={loading_scores.shape[1]}, labels={labels.shape[1]}")
    return eem_data, loading_scores, labels, eem_shape


class BalancedMultiTaskDataset(Dataset):
    def __init__(self,
                 eem_data: np.ndarray,
                 loading_scores: np.ndarray,
                 labels: np.ndarray,
                 augment: bool = False,
                 noise_std: float = 0.003):
        self.eem_data = torch.FloatTensor(eem_data)
        self.loading_scores = torch.FloatTensor(loading_scores)
        self.labels = torch.FloatTensor(labels)
        self.existence_labels = (self.labels != 0).float()
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.eem_data)

    def __getitem__(self, idx: int):
        eem = self.eem_data[idx]
        loading = self.loading_scores[idx]
        label = self.labels[idx]
        existence = self.existence_labels[idx]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                eem = eem + torch.randn_like(eem) * self.noise_std
            if torch.rand(1).item() < 0.3:
                loading = loading + torch.randn_like(loading) * (self.noise_std * 0.5)
        return eem, loading, label, existence


class BalancedMultiTaskModel(nn.Module):
    def __init__(self,
                 eem_shape: Tuple[int, int],
                 loading_dim: int,
                 num_labels: int,
                 conv_filters: int = 32,
                 fc_units: int = 128,
                 existence_dropout: float = 0.70,
                 regression_dropout: float = 0.12,
                 existence_hidden_ratio: float = 0.30,
                 existence_layers: int = 2,
                 regression_hidden_layers: int = 3):
        super().__init__()
        excitation_dim, emission_dim = eem_shape
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.loading_fc = nn.Sequential(
            nn.Linear(loading_dim, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, excitation_dim, emission_dim)
            cnn_size = self.shared_conv(dummy).numel()
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_size + fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
        )

        existence_net: List[nn.Module] = []
        cur = fc_units
        next_dim = max(8, int(fc_units * existence_hidden_ratio))
        existence_net += [nn.Linear(cur, next_dim), nn.BatchNorm1d(next_dim), nn.ReLU(inplace=True), nn.Dropout(existence_dropout)]
        cur = next_dim
        if existence_layers >= 2:
            next_dim = max(8, int(cur * 0.5))
            existence_net += [nn.Linear(cur, next_dim), nn.BatchNorm1d(next_dim), nn.ReLU(inplace=True), nn.Dropout(existence_dropout * 0.9)]
            cur = next_dim
        existence_net.append(nn.Linear(cur, num_labels))
        self.existence_branch = nn.Sequential(*existence_net)
        label4_index = int(REGRESSION_OPTIM_CONFIG.get("label4_index", 3))
        if not 0 <= label4_index < num_labels:
            raise ValueError("label4_index must be within the range of label count")
        self.label4_index = label4_index
        self.other_label_indices = [i for i in range(num_labels) if i != label4_index]

        common_net: List[nn.Module] = []
        cur = fc_units
        for i in range(regression_hidden_layers):
            if i < regression_hidden_layers - 1:
                next_dim = max(40, int(cur / 1.30))
            else:
                next_dim = max(40, fc_units // 2)
            common_net += [
                nn.Linear(cur, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(regression_dropout),
            ]
            cur = next_dim
        common_net.append(nn.Linear(cur, num_labels - 1))
        self.regression_common_branch = nn.Sequential(*common_net)

        label4_net: List[nn.Module] = []
        cur = fc_units
        label4_dims = [max(96, fc_units), max(64, int(fc_units * 0.75)), max(48, fc_units // 2)]
        for next_dim in label4_dims:
            label4_net += [
                nn.Linear(cur, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(regression_dropout),
            ]
            cur = next_dim
        label4_net.append(nn.Linear(cur, 1))
        self.regression_label4_branch = nn.Sequential(*label4_net)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, eem: torch.Tensor, loading: torch.Tensor):
        eem_feat = self.shared_conv(eem.unsqueeze(1)).flatten(1)
        loading_feat = self.loading_fc(loading)
        features = self.shared_fc(torch.cat([eem_feat, loading_feat], dim=1))
        existence_logits = self.existence_branch(features)

        common_pred = self.regression_common_branch(features)
        label4_pred = self.regression_label4_branch(features)
        regression_pred = torch.zeros(
            features.size(0),
            common_pred.size(1) + 1,
            dtype=features.dtype,
            device=features.device,
        )
        regression_pred[:, self.other_label_indices] = common_pred
        regression_pred[:, self.label4_index:self.label4_index + 1] = label4_pred
        return existence_logits, regression_pred


class AdaptiveMultiTaskLoss(nn.Module):
    def __init__(self,
                 initial_existence_weight: float,
                 initial_regression_weight: float,
                 max_existence_weight: float,
                 min_existence_weight: float,
                 max_regression_weight: float,
                 min_regression_weight: float,
                 huber_beta: float,
                 label_loss_weights: List[float],
                 device: torch.device):
        super().__init__()
        self.existence_weight = nn.Parameter(torch.tensor(float(initial_existence_weight), device=device), requires_grad=False)
        self.regression_weight = nn.Parameter(torch.tensor(float(initial_regression_weight), device=device), requires_grad=False)
        self.max_existence_weight = max_existence_weight
        self.min_existence_weight = min_existence_weight
        self.max_regression_weight = max_regression_weight
        self.min_regression_weight = min_regression_weight
        self.device = device
        self.register_buffer("label_loss_weights", torch.tensor(label_loss_weights, dtype=torch.float32, device=device))
        self.existence_criterion = nn.BCEWithLogitsLoss()
        self.regression_criterion = nn.SmoothL1Loss(beta=huber_beta)
        self.existence_train_acc = 0.0
        self.existence_val_acc = 0.0
        self.regression_train_r2 = 0.0
        self.regression_val_r2 = 0.0

    def update_performance_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        self.existence_train_acc = train_metrics.get("existence_accuracy", 0.0)
        self.existence_val_acc = val_metrics.get("existence_accuracy", 0.0)
        self.regression_train_r2 = train_metrics.get("regression_r2_std", 0.0)
        self.regression_val_r2 = val_metrics.get("regression_r2_std", 0.0)

    def adapt_weights(self, epoch: int) -> Tuple[float, float]:
        existence_overfit = (self.existence_train_acc - self.existence_val_acc) > 0.10
        regression_underfit = self.regression_val_r2 < 0.20
        ew = float(self.existence_weight.item())
        rw = float(self.regression_weight.item())
        if regression_underfit:
            ew = max(ew * 0.98, self.min_existence_weight)
            rw = min(rw * 1.05, self.max_regression_weight)
        elif existence_overfit:
            ew = max(ew * 0.95, self.min_existence_weight)
        else:
            if self.existence_val_acc < 0.90:
                ew = min(ew * 1.01, self.max_existence_weight)
                rw = max(rw * 0.995, self.min_regression_weight)
        self.existence_weight.data.fill_(ew)
        self.regression_weight.data.fill_(rw)
        return ew, rw

    def forward(self,
                existence_logits: torch.Tensor,
                regression_pred: torch.Tensor,
                existence_target: torch.Tensor,
                regression_target: torch.Tensor):
        existence_loss = self.existence_criterion(existence_logits, existence_target)
        mask = existence_target == 1
        regression_loss = torch.tensor(0.0, device=self.device)
        weight_sum = torch.tensor(0.0, device=self.device)
        for i in range(regression_target.shape[1]):
            label_mask = mask[:, i]
            if label_mask.sum() > 0:
                loss_i = self.regression_criterion(regression_pred[label_mask, i], regression_target[label_mask, i])
                w = self.label_loss_weights[i]
                regression_loss = regression_loss + w * loss_i
                weight_sum = weight_sum + w
        if weight_sum.item() > 0:
            regression_loss = regression_loss / weight_sum
        total_loss = self.existence_weight * existence_loss + self.regression_weight * regression_loss
        return total_loss, existence_loss, regression_loss


class GroupWiseScheduler:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 lr_patience: int = 10,
                 factor: float = 0.70,
                 regression_focus_epochs: int = 20):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.lr_patience = lr_patience
        self.factor = factor
        self.regression_focus_epochs = regression_focus_epochs
        self.best_score = -float("inf")
        self.counter = 0
        self.existence_factor = 1.0
        self.regression_factor = 1.0
        self.global_factor = 1.0

    def step(self, score: float, existence_overfit: bool, regression_underfit: bool, epoch: int) -> Dict[str, float]:
        if score > self.best_score + 1e-4:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.lr_patience:
                self.global_factor *= self.factor
                self.counter = 0
                print(f"Global learning rate factor reduced to {self.global_factor:.4f}")
        if epoch < self.regression_focus_epochs or regression_underfit:
            self.regression_factor = min(self.regression_factor * 1.02, 1.60)
            self.existence_factor = max(self.existence_factor * 0.99, 0.50)
        elif existence_overfit:
            self.existence_factor = max(self.existence_factor * 0.93, 0.35)
        lrs = {}
        for idx, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{idx}")
            factor = self.global_factor
            if name == "existence":
                factor *= self.existence_factor
            elif name == "regression":
                factor *= self.regression_factor
            group["lr"] = self.base_lrs[idx] * factor
            lrs[name] = group["lr"]
        return lrs


def make_model(eem_shape: Tuple[int, int], loading_dim: int, num_labels: int, device: torch.device) -> BalancedMultiTaskModel:
    cfg = REGRESSION_OPTIM_CONFIG
    return BalancedMultiTaskModel(
        eem_shape=eem_shape,
        loading_dim=loading_dim,
        num_labels=num_labels,
        conv_filters=int(cfg["conv_filters"]),
        fc_units=int(cfg["fc_units"]),
        existence_dropout=float(cfg["existence_dropout"]),
        regression_dropout=float(cfg["regression_dropout"]),
        existence_hidden_ratio=float(cfg["existence_hidden_ratio"]),
        existence_layers=int(cfg["existence_layers"]),
        regression_hidden_layers=int(cfg["regression_hidden_layers"]),
    ).to(device)


def split_params(model: nn.Module):
    shared_params, existence_params, regression_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "existence_branch" in name:
            existence_params.append(param)
        elif "regression_branch" in name:
            regression_params.append(param)
        else:
            shared_params.append(param)
    return shared_params, existence_params, regression_params


def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   device: torch.device,
                   criterion: Optional[AdaptiveMultiTaskLoss] = None,
                   label_scaler: Optional[NonZeroLabelStandardizer] = None,
                   raw_labels: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    thresholds = REGRESSION_OPTIM_CONFIG["existence_thresholds"]
    all_exist_pred, all_exist_target = [], []
    all_reg_pred, all_reg_target = [], []
    all_exist_prob = []
    total_loss = exist_loss_sum = reg_loss_sum = 0.0
    with torch.no_grad():
        for eem, loading, reg_target, exist_target in data_loader:
            eem = eem.to(device)
            loading = loading.to(device)
            reg_target = reg_target.to(device)
            exist_target = exist_target.to(device)
            exist_logits, reg_pred = model(eem, loading)
            if criterion is not None:
                loss, exist_loss, reg_loss = criterion(exist_logits, reg_pred, exist_target, reg_target)
                total_loss += float(loss.item())
                exist_loss_sum += float(exist_loss.item())
                reg_loss_sum += float(reg_loss.item())
            exist_pred = make_existence_predictions(exist_logits, thresholds)
            exist_prob = torch.sigmoid(exist_logits)
            all_exist_pred.append(exist_pred.cpu().numpy())
            all_exist_prob.append(exist_prob.cpu().numpy())
            all_exist_target.append(exist_target.cpu().numpy())
            all_reg_pred.append(reg_pred.cpu().numpy())
            all_reg_target.append(reg_target.cpu().numpy())
    existence_preds = np.vstack(all_exist_pred)
    existence_probs = np.vstack(all_exist_prob)
    existence_targets = np.vstack(all_exist_target)
    regression_preds = np.vstack(all_reg_pred)
    regression_targets = np.vstack(all_reg_target)
    metrics = regression_metrics(regression_preds, regression_targets, existence_targets)
    metrics["existence_accuracy"] = float((existence_preds == existence_targets).mean())
    if label_scaler is not None and raw_labels is not None:
        pred_raw_ungated = label_scaler.inverse_transform(regression_preds, existence_mask=None)
        pred_raw_gated = label_scaler.inverse_transform(regression_preds, existence_mask=existence_preds)
        metrics.update(regression_metrics(
            regression_preds,
            regression_targets,
            existence_targets,
            pred_raw_gated=pred_raw_gated,
            pred_raw_ungated=pred_raw_ungated,
            target_raw=raw_labels,
        ))
    n = max(len(data_loader), 1)
    losses = {"total_loss": total_loss / n, "existence_loss": exist_loss_sum / n, "regression_loss": reg_loss_sum / n}
    arrays = {
        "existence_preds": existence_preds,
        "existence_probs": existence_probs,
        "existence_targets": existence_targets,
        "regression_preds_std": regression_preds,
        "regression_targets_std": regression_targets,
    }
    return metrics, losses, arrays


def train_one_model(model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    num_epochs: int,
                    device: torch.device,
                    patience: int,
                    lr_patience: int) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float]]:
    cfg = REGRESSION_OPTIM_CONFIG
    shared_params, existence_params, regression_params = split_params(model)
    criterion = AdaptiveMultiTaskLoss(
        initial_existence_weight=float(cfg["initial_existence_weight"]),
        initial_regression_weight=float(cfg["initial_regression_weight"]),
        max_existence_weight=float(cfg["max_existence_weight"]),
        min_existence_weight=float(cfg["min_existence_weight"]),
        max_regression_weight=float(cfg["max_regression_weight"]),
        min_regression_weight=float(cfg["min_regression_weight"]),
        huber_beta=float(cfg["huber_beta"]),
        label_loss_weights=list(cfg["label_loss_weights"]),
        device=device,
    )
    optimizer = torch.optim.AdamW([
        {"params": shared_params, "lr": float(cfg["shared_lr"]), "weight_decay": float(cfg["shared_weight_decay"]), "name": "shared"},
        {"params": existence_params, "lr": float(cfg["existence_lr"]), "weight_decay": float(cfg["existence_weight_decay"]), "name": "existence"},
        {"params": regression_params, "lr": float(cfg["regression_lr"]), "weight_decay": float(cfg["regression_weight_decay"]), "name": "regression"},
    ])
    scheduler = GroupWiseScheduler(
        optimizer,
        lr_patience=lr_patience,
        factor=float(cfg["lr_factor"]),
        regression_focus_epochs=int(cfg["regression_focus_epochs"]),
    )
    history: Dict[str, List[float]] = {k: [] for k in [
        "train_total_loss", "val_total_loss", "train_existence_loss", "val_existence_loss",
        "train_regression_loss", "val_regression_loss", "train_existence_accuracy", "val_existence_accuracy",
        "train_regression_r2_std", "val_regression_r2_std", "val_final_r2_std",
        "existence_weight", "regression_weight", "lr_shared", "lr_existence", "lr_regression", "score"
    ]}
    best_score = -float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    thresholds = REGRESSION_OPTIM_CONFIG["existence_thresholds"]

    for epoch in range(num_epochs):
        model.train()
        train_total = train_exist = train_reg = 0.0
        exist_correct = exist_total = 0
        tr_pred, tr_target, tr_exist = [], [], []
        optimizer.zero_grad(set_to_none=True)
        for eem, loading, reg_target, exist_target in train_loader:
            eem = eem.to(device)
            loading = loading.to(device)
            reg_target = reg_target.to(device)
            exist_target = exist_target.to(device)
            exist_logits, reg_pred = model(eem, loading)
            loss, exist_loss, reg_loss = criterion(exist_logits, reg_pred, exist_target, reg_target)
            if epoch < int(cfg["regression_focus_epochs"]):
                loss = loss * float(cfg["regression_focus_multiplier"])
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regression_params, float(cfg["grad_clip_regression"]))
            torch.nn.utils.clip_grad_norm_(existence_params, float(cfg["grad_clip_existence"]))
            torch.nn.utils.clip_grad_norm_(shared_params, float(cfg["shared_grad_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_total += float(loss.item())
            train_exist += float(exist_loss.item())
            train_reg += float(reg_loss.item())
            exist_pred = make_existence_predictions(exist_logits, thresholds)
            exist_correct += int((exist_pred == exist_target).sum().item())
            exist_total += int(exist_target.numel())
            tr_pred.append(reg_pred.detach().cpu().numpy())
            tr_target.append(reg_target.detach().cpu().numpy())
            tr_exist.append(exist_target.detach().cpu().numpy())

        train_n = max(len(train_loader), 1)
        train_exist_acc = exist_correct / max(exist_total, 1)
        train_reg_r2 = calculate_robust_r2(np.vstack(tr_pred), np.vstack(tr_target), np.vstack(tr_exist)) if tr_pred else 0.0
        val_metrics, val_losses, _ = evaluate_model(model, val_loader, device, criterion=criterion)
        criterion.update_performance_metrics(
            {"existence_accuracy": train_exist_acc, "regression_r2_std": train_reg_r2},
            val_metrics,
        )
        ew, rw = criterion.adapt_weights(epoch)
        existence_overfit = (train_exist_acc - val_metrics["existence_accuracy"]) > 0.10
        regression_underfit = val_metrics["regression_r2_std"] < 0.20
        score = (
            val_metrics["regression_r2_std"]
            + float(cfg["score_exist_weight"]) * val_metrics["existence_accuracy"]
            - float(cfg["score_reg_loss_weight"]) * val_losses["regression_loss"]
        )
        lrs = scheduler.step(score, existence_overfit, regression_underfit, epoch)

        history["train_total_loss"].append(train_total / train_n)
        history["train_existence_loss"].append(train_exist / train_n)
        history["train_regression_loss"].append(train_reg / train_n)
        history["train_existence_accuracy"].append(train_exist_acc)
        history["train_regression_r2_std"].append(train_reg_r2)
        history["val_total_loss"].append(val_losses["total_loss"])
        history["val_existence_loss"].append(val_losses["existence_loss"])
        history["val_regression_loss"].append(val_losses["regression_loss"])
        history["val_existence_accuracy"].append(val_metrics["existence_accuracy"])
        history["val_regression_r2_std"].append(val_metrics["regression_r2_std"])
        history["val_final_r2_std"].append(val_metrics["final_r2_std"])
        history["existence_weight"].append(ew)
        history["regression_weight"].append(rw)
        history["lr_shared"].append(lrs.get("shared", 0.0))
        history["lr_existence"].append(lrs.get("existence", 0.0))
        history["lr_regression"].append(lrs.get("regression", 0.0))
        history["score"].append(score)

        if score > best_score + float(cfg["early_stop_min_delta"]):
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{num_epochs} | score={score:.4f} | "
                f"val_r2={val_metrics['regression_r2_std']:.4f} | final_r2={val_metrics['final_r2_std']:.4f} | "
                f"exist_acc={val_metrics['existence_accuracy']:.4f} | lr_reg={lrs.get('regression', 0):.6f} | ew={ew:.3f} rw={rw:.3f}"
            )
        if patience_counter >= patience:
            print(f"early stopping: epoch={epoch + 1}, best_epoch={best_epoch + 1}, best_score={best_score:.4f}")
            break
        if val_metrics["regression_r2_std"] < float(cfg["min_r2_threshold"]) and epoch > 15:
            print(f"Early stopping: The R2 value of the regression is too low {val_metrics['regression_r2_std']:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {"best_score": best_score, "best_epoch": best_epoch}


def compute_fold_score(metrics: Dict[str, float]) -> float:
    cfg = REGRESSION_OPTIM_CONFIG
    label4_idx = int(cfg["label4_index"]) + 1
    label4_r2 = metrics.get(f"ungated_label{label4_idx}_r2", metrics.get(f"gated_label{label4_idx}_r2", 0.0))
    overall = metrics.get("final_r2_raw_ungated", metrics.get("final_r2_raw_gated", 0.0))
    return float(cfg["fold_score_label4_weight"]) * label4_r2 + float(cfg["fold_score_overall_weight"]) * overall


def plot_history(history: Dict[str, List[float]], output_file: str) -> None:
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(history["train_total_loss"], label="train")
        axes[0, 0].plot(history["val_total_loss"], label="val")
        axes[0, 0].set_title("Total loss")
        axes[0, 0].legend()
        axes[0, 1].plot(history["train_regression_r2_std"], label="train")
        axes[0, 1].plot(history["val_regression_r2_std"], label="val")
        axes[0, 1].set_title("Regression R2 std")
        axes[0, 1].legend()
        axes[1, 0].plot(history["val_existence_accuracy"])
        axes[1, 0].set_title("Val existence accuracy")
        axes[1, 1].plot(history["lr_regression"], label="lr_reg")
        axes[1, 1].plot(history["lr_shared"], label="lr_shared")
        axes[1, 1].set_title("Learning rates")
        axes[1, 1].legend()
        fig.tight_layout()
        fig.savefig(output_file, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Drawing failed: {exc}")


def cross_validate_regression_optimized(eem_data: np.ndarray,
                                        loading_scores: np.ndarray,
                                        labels: np.ndarray,
                                        eem_shape: Tuple[int, int],
                                        n_splits: int = 5,
                                        num_epochs: int = 180,
                                        num_workers: int = 4,
                                        patience: int = 35,
                                        lr_patience: int = 10,
                                        output_dir: str = "."):
    seed_everything(int(RUN_CONFIG["seed"]))
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(REGRESSION_OPTIM_CONFIG, indent=2, ensure_ascii=False))
    loading_dim = loading_scores.shape[1]
    num_labels = labels.shape[1]
    if len(REGRESSION_OPTIM_CONFIG["label_loss_weights"]) != num_labels:
        raise ValueError("The length of label_loss_weights must be equal to the number of labels")
    if len(REGRESSION_OPTIM_CONFIG["existence_thresholds"]) != num_labels:
        raise ValueError("The length of the existence_thresholds must be equal to the number of labels")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(RUN_CONFIG["seed"]))
    fold_results = {"fold_histories": [], "fold_metrics": [], "best_epochs": [], "train_indices": [], "val_indices": []}
    cv_metrics: Dict[str, List[float]] = {"fold_score": [], "existence_accuracy": [], "regression_r2_std": [], "regression_r2_raw_gated": [], "regression_r2_raw_ungated": [], "final_r2_raw_gated": [], "final_r2_raw_ungated": [], "label4_r2_ungated": [], "label4_r2_gated": [], "train_time": []}
    best_fold_score = -float("inf")
    best_bundle = None
    best_fold_index = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(eem_data), start=1):
        print("\n" + "=" * 70)
        print(f"start fold {fold}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")
        print("=" * 70)
        X_eem_train, X_eem_val = eem_data[train_idx], eem_data[val_idx]
        X_load_train, X_load_val = loading_scores[train_idx], loading_scores[val_idx]
        y_train_raw, y_val_raw = labels[train_idx], labels[val_idx]

        scaler_eem = StandardScaler()
        X_eem_train_proc = scaler_eem.fit_transform(X_eem_train.reshape(len(X_eem_train), -1)).reshape(X_eem_train.shape)
        X_eem_val_proc = scaler_eem.transform(X_eem_val.reshape(len(X_eem_val), -1)).reshape(X_eem_val.shape)
        scaler_load = StandardScaler()
        X_load_train_proc = scaler_load.fit_transform(X_load_train)
        X_load_val_proc = scaler_load.transform(X_load_val)
        label_scaler = NonZeroLabelStandardizer().fit(y_train_raw)
        y_train = label_scaler.transform(y_train_raw)
        y_val = label_scaler.transform(y_val_raw)

        train_ds = BalancedMultiTaskDataset(X_eem_train_proc, X_load_train_proc, y_train, augment=True, noise_std=float(REGRESSION_OPTIM_CONFIG["augment_noise_std"]))
        val_ds = BalancedMultiTaskDataset(X_eem_val_proc, X_load_val_proc, y_val, augment=False)
        train_loader = DataLoader(train_ds, batch_size=int(REGRESSION_OPTIM_CONFIG["batch_size"]), shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=int(REGRESSION_OPTIM_CONFIG["batch_size"]), shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0))

        model = make_model(eem_shape, loading_dim, num_labels, device)
        start_time = time.time()
        model, history, best_summary = train_one_model(model, train_loader, val_loader, num_epochs, device, patience=patience, lr_patience=lr_patience)
        train_time = time.time() - start_time
        metrics, losses, arrays = evaluate_model(model, val_loader, device, label_scaler=label_scaler, raw_labels=y_val_raw)
        fold_score = compute_fold_score(metrics)

        fold_results["fold_histories"].append(history)
        fold_results["fold_metrics"].append(metrics)
        fold_results["best_epochs"].append(best_summary["best_epoch"])
        fold_results["train_indices"].append(train_idx)
        fold_results["val_indices"].append(val_idx)
        cv_metrics["fold_score"].append(fold_score)
        cv_metrics["existence_accuracy"].append(metrics["existence_accuracy"])
        cv_metrics["regression_r2_std"].append(metrics["regression_r2_std"])
        cv_metrics["regression_r2_raw_gated"].append(metrics.get("regression_r2_raw_gated", 0.0))
        cv_metrics["regression_r2_raw_ungated"].append(metrics.get("regression_r2_raw_ungated", 0.0))
        cv_metrics["final_r2_raw_gated"].append(metrics.get("final_r2_raw_gated", 0.0))
        cv_metrics["final_r2_raw_ungated"].append(metrics.get("final_r2_raw_ungated", 0.0))
        cv_metrics["label4_r2_ungated"].append(metrics.get("ungated_label4_r2", 0.0))
        cv_metrics["label4_r2_gated"].append(metrics.get("gated_label4_r2", 0.0))
        cv_metrics["train_time"].append(train_time)

        print(f"Fold {fold} 完成 | score={fold_score:.4f} | label4 ungated R2={metrics.get('ungated_label4_r2', 0.0):.4f} | label4 gated R2={metrics.get('gated_label4_r2', 0.0):.4f} | regression raw ungated R2={metrics.get('regression_r2_raw_ungated', 0.0):.4f} | final raw ungated R2={metrics.get('final_r2_raw_ungated', 0.0):.4f}")
        plot_history(history, os.path.join(output_dir, f"fold{fold}_history.png"))

        if fold_score > best_fold_score:
            best_fold_score = fold_score
            best_fold_index = fold - 1
            best_bundle = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "eem_shape": eem_shape,
                "loading_dim": loading_dim,
                "num_labels": num_labels,
                "scaler_eem": scaler_eem,
                "scaler_load": scaler_load,
                "label_scaler_state": label_scaler.state_dict(),
                "best_fold_index": best_fold_index,
                "best_fold_score": best_fold_score,
                "best_fold_metrics": metrics,
                "training_config": copy.deepcopy(REGRESSION_OPTIM_CONFIG),
                "run_config": copy.deepcopy(RUN_CONFIG),
                "cv_metrics": cv_metrics,
                "fold_results": fold_results,
            }

    print("\n" + "=" * 70)
    print("Five-fold cross-validation summary")
    print("=" * 70)
    for k, vals in cv_metrics.items():
        if vals:
            print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"best fold: {best_fold_index + 1}, best_fold_score={best_fold_score:.4f}")

    assert best_bundle is not None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"best_model_label4_generalized_{timestamp}.pth")
    torch.save(best_bundle, model_path)
    print(f"the best model has been saved: {model_path}")
    with open(os.path.join(output_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in cv_metrics.items()}, f, indent=2, ensure_ascii=False)
    return best_bundle, model_path


def find_latest_model(output_dir: str) -> str:
    candidates = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.endswith(".pth") and "best_model" in name:
                path = os.path.join(output_dir, name)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(f"The file "best_model*.pth" was not found in {output_dir}, Please set RUN_CONFIG['model_file']")
    return sorted(candidates, reverse=True)[0][1]


def external_validate(model_file: str,
                      eem_file: str,
                      loading_file: str,
                      label_file: str,
                      output_file: str) -> Dict[str, float]:
    if not model_file:
        model_file = find_latest_model(str(RUN_CONFIG["output_dir"]))
    print(f"load the model: {model_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(model_file, map_location=device)
    eem_shape = tuple(bundle["eem_shape"])
    loading_dim = int(bundle["loading_dim"])
    num_labels = int(bundle["num_labels"])
    scaler_eem: StandardScaler = bundle["scaler_eem"]
    scaler_load: StandardScaler = bundle["scaler_load"]
    label_scaler = NonZeroLabelStandardizer.from_state_dict(bundle["label_scaler_state"])

    eem_data, loading_scores, labels_raw, ext_eem_shape = load_spectral_data_fixed_headers(eem_file, loading_file, label_file)
    if tuple(ext_eem_shape) != tuple(eem_shape):
        raise ValueError(f"The external EEM shape {ext_eem_shape} is inconsistent with the shape used for model training {eem_shape}")
    X_eem_proc = scaler_eem.transform(eem_data.reshape(len(eem_data), -1)).reshape(eem_data.shape)
    X_load_proc = scaler_load.transform(loading_scores)
    labels_std = label_scaler.transform(labels_raw)
    ds = BalancedMultiTaskDataset(X_eem_proc, X_load_proc, labels_std, augment=False)
    loader = DataLoader(ds, batch_size=int(REGRESSION_OPTIM_CONFIG["batch_size"]), shuffle=False, num_workers=0)
    model = make_model(eem_shape, loading_dim, num_labels, device)
    model.load_state_dict(bundle["model_state_dict"])
    metrics, losses, arrays = evaluate_model(model, loader, device, label_scaler=label_scaler, raw_labels=labels_raw)

    pred_raw_ungated = label_scaler.inverse_transform(arrays["regression_preds_std"], existence_mask=None)
    pred_raw_gated = label_scaler.inverse_transform(arrays["regression_preds_std"], existence_mask=arrays["existence_preds"])

    out = pd.DataFrame()
    for i in range(num_labels):
        out[f"label{i + 1}_true"] = labels_raw[:, i]
        out[f"label{i + 1}_exist_true"] = arrays["existence_targets"][:, i]
        out[f"label{i + 1}_exist_prob"] = arrays["existence_probs"][:, i]
        out[f"label{i + 1}_exist_pred"] = arrays["existence_preds"][:, i]
        out[f"label{i + 1}_pred_ungated"] = pred_raw_ungated[:, i]
        out[f"label{i + 1}_pred_gated"] = pred_raw_gated[:, i]
    out.to_csv(output_file, index=False, encoding="utf-8-sig")

    metrics_path = os.path.splitext(output_file)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}, f, indent=2, ensure_ascii=False)
    print(f"save: {output_file}")
    print(f"save: {metrics_path}")
    print(f"  regression std R2:          {metrics.get('regression_r2_std', 0.0):.4f}")
    print(f"  regression raw gated R2:    {metrics.get('regression_r2_raw_gated', 0.0):.4f}")
    print(f"  regression raw ungated R2:  {metrics.get('regression_r2_raw_ungated', 0.0):.4f}")
    print(f"  final raw gated R2:         {metrics.get('final_r2_raw_gated', 0.0):.4f}")
    print(f"  final raw ungated R2:       {metrics.get('final_r2_raw_ungated', 0.0):.4f}")
    print(f"  label4 gated R2:    {metrics.get('gated_label4_r2', 0.0):.4f}")
    print(f"  label4 ungated R2:  {metrics.get('ungated_label4_r2', 0.0):.4f}")
    return metrics

def main() -> None:
    os.makedirs(str(RUN_CONFIG["output_dir"]), exist_ok=True)
    seed_everything(int(RUN_CONFIG["seed"]))
    if RUN_CONFIG["mode"] == "train":
        eem_data, loading_scores, labels, eem_shape = load_spectral_data_fixed_headers(
            str(RUN_CONFIG["eem_file"]), str(RUN_CONFIG["loading_file"]), str(RUN_CONFIG["label_file"])
        )
        cross_validate_regression_optimized(
            eem_data=eem_data,
            loading_scores=loading_scores,
            labels=labels,
            eem_shape=eem_shape,
            n_splits=int(RUN_CONFIG["n_splits"]),
            num_epochs=int(RUN_CONFIG["epochs"]),
            num_workers=int(RUN_CONFIG["num_workers"]),
            patience=int(RUN_CONFIG["patience"]),
            lr_patience=int(RUN_CONFIG["lr_patience"]),
            output_dir=str(RUN_CONFIG["output_dir"]),
        )
    elif RUN_CONFIG["mode"] == "external":
        external_validate(
            model_file=str(RUN_CONFIG["model_file"]),
            eem_file=str(RUN_CONFIG["external_eem_file"]),
            loading_file=str(RUN_CONFIG["external_loading_file"]),
            label_file=str(RUN_CONFIG["external_label_file"]),
            output_file=str(RUN_CONFIG["external_output_file"]),
        )
    else:
        raise ValueError("RUN_CONFIG['mode'] 'train' or 'external'")


if __name__ == "__main__":
    main()
