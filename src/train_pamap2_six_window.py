"""Fair six-window PAMAP2 comparison for TCST-HAR and all baselines.

All methods receive the same six consecutive 128x27 sensor windows and are
evaluated only on the label of the sixth window. TCST-HAR retains the window
hierarchy; baseline models consume the same observations as a 768x27 sequence.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import MODEL_REGISTRY
from models.train_tcst import TCSTHAR


class HARMamba(nn.Module):
    def __init__(self, input_size, num_classes, d_model=128, dropout=0.2, **kwargs):
        super().__init__()
        self.patch = nn.Conv1d(input_size, d_model, kernel_size=16, stride=8)
        self.mix = nn.Linear(d_model, 2 * d_model)
        self.decay = nn.Parameter(torch.zeros(d_model))
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def scan(self, x, reverse=False):
        if reverse:
            x = x.flip(1)
        value, gate = self.mix(x).chunk(2, -1)
        gate = torch.sigmoid(gate)
        decay = torch.sigmoid(self.decay).view(1, -1)
        state = torch.zeros_like(value[:, 0])
        outputs = []
        for index in range(value.shape[1]):
            state = decay * state + (1 - decay) * value[:, index] * gate[:, index]
            outputs.append(state)
        output = torch.stack(outputs, 1)
        return output.flip(1) if reverse else output

    def forward(self, x):
        token = self.patch(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.norm(self.out(self.scan(token) + self.scan(token, True)))
        return self.head(hidden.mean(1))


class ModernBlock(nn.Module):
    def __init__(self, width, kernel_size, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width, bias=False)
        self.bn = nn.BatchNorm1d(width)
        self.pointwise = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(2 * width, width, 1)
        )
        self.norm = nn.BatchNorm1d(width)
        self.scale = nn.Parameter(torch.ones(1, width, 1) * 0.1)

    def forward(self, x):
        return x + self.scale * self.norm(self.pointwise(torch.nn.functional.gelu(self.bn(self.depthwise(x)))))


class ModernTCN(nn.Module):
    def __init__(self, input_size, num_classes, d_model=192, dropout=0.15, **kwargs):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(input_size, d_model, 1, bias=False), nn.BatchNorm1d(d_model), nn.GELU())
        self.blocks = nn.Sequential(*[ModernBlock(d_model, kernel, dropout) for kernel in (7, 15, 31, 63, 7)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x.transpose(1, 2))))


class SixWindowDataset(Dataset):
    def __init__(self, x, y, hierarchical):
        self.x = x
        self.y = y
        self.hierarchical = hierarchical

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index].astype(np.float32)
        if self.hierarchical:
            # TCST-HAR uses all six labels only for its transition regularizer.
            return torch.from_numpy(x), torch.from_numpy(self.y[index].copy()).long()
        x = x.reshape(-1, x.shape[-1])
        return torch.from_numpy(x), torch.tensor(self.y[index, -1], dtype=torch.long)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(name, channels, num_classes, tcst_variant="full"):
    if name == "tcst_har":
        options = {
            "full": {},
            "no_transition_loss": {},
            "no_causal_mask": {"causal_mask": False},
            "no_transition": {"disable_transition": True},
            "transformer_only": {"disable_transition": True, "disable_state_encoder": True},
        }
        if tcst_variant not in options:
            raise ValueError(f"Unknown TCST variant: {tcst_variant}")
        return TCSTHAR(channels, num_classes, latent_size=192, heads=8, layers=3, dropout=0.15, action_mode="ground_truth", **options[tcst_variant])
    if name == "harmamba":
        return HARMamba(channels, num_classes)
    if name == "moderntcn":
        return ModernTCN(channels, num_classes)
    if name == "resnet1d":
        return MODEL_REGISTRY[name](input_size=channels, num_classes=num_classes, hidden_size=192, layers=5, dropout=0.15)
    return MODEL_REGISTRY[name](
        input_size=channels, num_classes=num_classes, seq_len=768, hidden_size=128, d_model=128,
        heads=4, layers=2, patch_len=16, stride=8, dropout=0.25,
    )


def evaluate(model, loader, device, criterion, tcst):
    model.eval()
    targets, predictions, probabilities, losses = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x, y)[0][:, -1] if tcst else model(x)
            target = y[:, -1] if tcst else y
            losses.append(float(criterion(logits, target).item()))
            targets.append(target.cpu().numpy())
            predictions.append(logits.argmax(-1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions), np.concatenate(probabilities), float(np.mean(losses))


def train_one(name, args, data, metadata, device):
    seed_all(args.seed)
    tcst = name == "tcst_har"
    loaders = []
    for split in ("train", "val", "test"):
        dataset = SixWindowDataset(data[f"x_{split}"], data[f"y_{split}"], hierarchical=tcst)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=0, pin_memory=True))
    train_loader, val_loader, test_loader = loaders
    num_classes = len(metadata["activity_names"])
    channels = data["x_train"].shape[-1]
    model = build_model(name, channels, num_classes, args.tcst_variant).to(device)
    train_labels = data["y_train"][:, -1]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    learning_rate = 1e-2 if name == "deepconvlstm" else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if tcst:
                logits, transition_loss = model(x, y)
                transition_weight = 0.0 if args.tcst_variant == "no_transition_loss" else 0.1
                loss = criterion(logits[:, -1], y[:, -1])
                if transition_loss is not None:
                    loss = loss + transition_weight * transition_loss
            else:
                loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        val_y, val_pred, _, val_loss = evaluate(model, val_loader, device, criterion, tcst)
        val_acc = accuracy_score(val_y, val_pred)
        val_f1 = f1_score(val_y, val_pred, average="macro", zero_division=0)
        scheduler.step(val_f1)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"{name} epoch {epoch:02d}: val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_score:
            best_score, best_epoch = val_f1, epoch
            torch.save({"model_state": model.state_dict(), "model": name, "metadata": metadata, "args": vars(args)}, output / "best_model.pt")

    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    y_true, y_pred, y_prob, test_loss = evaluate(model, test_loader, device, criterion, tcst)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    metrics = {
        "model": name,
        "best_epoch": best_epoch,
        "test_sequences": int(len(y_true)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_roc_auc": float(roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")),
        "micro_average_precision": float(average_precision_score(y_bin, y_prob, average="micro")),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "test_loss": test_loss,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "classification_report.txt").write_text(classification_report(y_true, y_pred, target_names=metadata["activity_names"], zero_division=0), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob, class_names=np.asarray(metadata["activity_names"]))
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="tcst_har,cnn1d,deepconvlstm,lstm,patchtst,itransformer,resnet1d,inceptiontime,tcn,grufcn,timesnet,harmamba,moderntcn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tcst-variant", choices=["full", "no_transition_loss", "no_causal_mask", "no_transition", "transformer_only"], default="full")
    args = parser.parse_args()
    raw = np.load(args.data, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    data = {key: raw[key] for key in raw.files if key.startswith("x_") or key.startswith("y_")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [train_one(name.strip(), args, data, metadata, device) for name in args.models.split(",") if name.strip()]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
