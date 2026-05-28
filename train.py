"""Training and evaluation loops for ServeLSTM."""

from __future__ import annotations

import argparse
import copy
import importlib
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import model as model_module
from model import CLASS_NAMES, NUM_CLASSES


def class_weights(y: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights for imbalanced Flat / Kick / Slice."""
    counts = torch.bincount(y, minlength=NUM_CLASSES).float().clamp(min=1)
    weights = 1.0 / counts
    return weights * (NUM_CLASSES / weights.sum())


def downsample_balanced(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly downsample majority classes to match the minority class count."""
    generator = torch.Generator().manual_seed(seed)
    counts = torch.bincount(y, minlength=NUM_CLASSES)
    target = int(counts.min().item())

    indices: list[torch.Tensor] = []
    for cls in range(NUM_CLASSES):
        cls_idx = (y == cls).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=generator)]
        indices.append(perm[:target])

    idx = torch.cat(indices)
    idx = idx[torch.randperm(len(idx), generator=generator)]
    return X[idx], y[idx]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).argmax(dim=1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds), torch.cat(labels)


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of hyperparameter lists, e.g. {'lr': [1e-3, 5e-4]} -> all combos."""
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]


def run_training(
    *,
    train_path: str = "train_data.pt",
    val_path: str = "val_data.pt",
    checkpoint_path: str | None = "serve_lstm_best.pt",
    batch_size: int = 512,
    epochs: int = 25,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    use_layernorm: bool = False,
    weight_decay: float = 0.0,
    use_class_weights: bool = True,
    use_downsampling: bool = False,
    downsample_seed: int = 42,
    manual_class_weights: list[float] | tuple[float, ...] | None = None,
    patience: int = 5,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict:
    # If model.py was edited in-notebook, make sure we use the latest symbols.
    importlib.reload(model_module)

    # Pandas/grid-search rows can upcast ints to floats (e.g., 256.0).
    # Normalize types here so DataLoader/LSTM always receive valid arg types.
    batch_size = int(batch_size)
    epochs = int(epochs)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    patience = int(patience)
    num_workers = int(num_workers)
    lr = float(lr)
    dropout = float(dropout)
    weight_decay = float(weight_decay)
    if isinstance(use_layernorm, str):
        use_layernorm = use_layernorm.strip().lower() in {"1", "true", "yes", "y"}
    else:
        use_layernorm = bool(use_layernorm)
    if isinstance(bidirectional, str):
        bidirectional = bidirectional.strip().lower() in {"1", "true", "yes", "y"}
    else:
        bidirectional = bool(bidirectional)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = torch.load(train_path, weights_only=True)
    X_val, y_val = torch.load(val_path, weights_only=True)

    if use_downsampling and use_class_weights:
        raise ValueError("use_downsampling and use_class_weights are mutually exclusive.")
    if use_downsampling:
        if verbose:
            before = torch.bincount(y_train, minlength=NUM_CLASSES).tolist()
            print(f"Train counts before downsampling: {dict(zip(CLASS_NAMES, before))}")
        X_train, y_train = downsample_balanced(X_train, y_train, seed=downsample_seed)
        if verbose:
            after = torch.bincount(y_train, minlength=NUM_CLASSES).tolist()
            print(f"Train counts after downsampling:  {dict(zip(CLASS_NAMES, after))}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model_module.build_model(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        use_layernorm=use_layernorm,
        device=device,
    )

    if manual_class_weights is not None:
        if len(manual_class_weights) != NUM_CLASSES:
            raise ValueError(
                f"manual_class_weights must have length {NUM_CLASSES}, got {len(manual_class_weights)}"
            )
        weight = torch.tensor([float(w) for w in manual_class_weights], dtype=torch.float32).to(device)
        if verbose:
            print(
                "Loss class weights (manual):",
                dict(zip(CLASS_NAMES, [round(w, 3) for w in weight.cpu().tolist()])),
            )
    elif use_class_weights:
        weight = class_weights(y_train).to(device)
        if verbose:
            print(
                "Loss class weights (auto):",
                dict(zip(CLASS_NAMES, [round(w, 3) for w in weight.cpu().tolist()])),
            )
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "params": {
            "batch_size": batch_size,
            "lr": lr,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "use_layernorm": use_layernorm,
            "weight_decay": weight_decay,
            "use_class_weights": use_class_weights,
            "use_downsampling": use_downsampling,
            "downsample_seed": downsample_seed,
            "manual_class_weights": list(manual_class_weights) if manual_class_weights is not None else None,
        },
    }
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            if checkpoint_path:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "params": history["params"],
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improve += 1
            if patience and epochs_without_improve >= patience:
                if verbose:
                    print(f"Early stop: no val loss improvement for {patience} epochs.")
                break

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = ckpt["epoch"]
        best_val_loss = ckpt["val_loss"]
        best_val_acc = ckpt["val_acc"]
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(
            f"Best: epoch {best_epoch}, val_loss {best_val_loss:.4f}, val_acc {best_val_acc:.3f}"
        )

    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["best_checkpoint"] = checkpoint_path
    history["model"] = model
    history["device"] = device
    return history


def tune_hyperparameters(
    param_grid: dict[str, list[Any]],
    *,
    search_epochs: int = 12,
    patience: int = 3,
    train_path: str = "train_data.pt",
    val_path: str = "val_data.pt",
    use_class_weights: bool = True,
    use_downsampling: bool = False,
    downsample_seed: int = 42,
    manual_class_weights: list[float] | tuple[float, ...] | None = None,
    verbose_trials: bool = True,
) -> pd.DataFrame:
    """
    Grid search over param_grid. Each combo trains with search_epochs (shorter than final training).

    Returns a DataFrame sorted by best val accuracy (descending). Use the top row's params in run_training
  with full epochs for the final model.
    """
    combos = expand_param_grid(param_grid)
    rows: list[dict[str, Any]] = []

    for i, params in enumerate(combos, start=1):
        if verbose_trials:
            print(f"\n=== Trial {i}/{len(combos)}: {params} ===")

        history = run_training(
            train_path=train_path,
            val_path=val_path,
            checkpoint_path=None,
            epochs=search_epochs,
            patience=patience,
            use_class_weights=use_class_weights,
            use_downsampling=use_downsampling,
            downsample_seed=downsample_seed,
            manual_class_weights=manual_class_weights,
            verbose=verbose_trials,
            **params,
        )

        rows.append(
            {
                **params,
                "best_val_acc": history["best_val_acc"],
                "best_val_loss": history["best_val_loss"],
                "best_epoch": history["best_epoch"],
                "epochs_ran": len(history["val_acc"]),
            }
        )

    results = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    if verbose_trials:
        print("\n=== Top 5 configs by validation accuracy ===")
        print(results.head().to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ServeLSTM")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--downsample", action="store_true", help="Balance train set by downsampling")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_class_weights=not args.no_class_weights and not args.downsample,
        use_downsampling=args.downsample,
    )


if __name__ == "__main__":
    main()
