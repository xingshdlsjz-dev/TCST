import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_models import MODEL_REGISTRY


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "targets": np.asarray(targets),
        "preds": np.asarray(preds),
    }


def train_one(model_name, args, data, metadata, device):
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        input_size=x_train.shape[-1],
        num_classes=len(metadata["activity_names"]),
        seq_len=x_train.shape[1],
        hidden_size=args.hidden_size,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(x_train, y_train, args.batch_size, True)
    val_loader = make_loader(x_val, y_val, args.batch_size, False)
    test_loader = make_loader(x_test, y_test, args.batch_size, False)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    out_dir = args.output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    best_path = out_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val = evaluate(model, val_loader, device, criterion)
        scheduler.step(val["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        print(
            f"{model_name} epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )
        if val["macro_f1"] > best_score:
            best_score, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "metadata": metadata, "args": vars(args)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"{model_name} early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])
    test = evaluate(model, test_loader, device, criterion)
    metrics = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(
            test["targets"],
            test["preds"],
            target_names=metadata["activity_names"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare HAR baseline models on processed PAMAP2 windows.")
    parser.add_argument("--data", type=Path, default=Path("processed/pamap2_paper10_50hz_128_random.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/compare_models_paper10"))
    parser.add_argument("--models", type=str, default="deepconvlstm,cnn1d,lstm,patchtst,itransformer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    raw = np.load(args.data, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    data = {key: raw[key] for key in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]}
    # Subject-split preprocessing can store single-window samples as a
    # sequence of length one. Baseline models consume [N, L, C].
    for split in ["train", "val", "test"]:
        if data[f"x_{split}"].ndim == 4 and data[f"x_{split}"].shape[1] == 1:
            data[f"x_{split}"] = data[f"x_{split}"][:, 0]
            data[f"y_{split}"] = data[f"y_{split}"][:, 0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        all_metrics.append(train_one(model_name, args, data, metadata, device))

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
