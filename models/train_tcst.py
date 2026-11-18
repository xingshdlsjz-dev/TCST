import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TCSTHAR(nn.Module):
    """Temporal Causal State Transformer for wearable HAR.

    Each sensor window is encoded into a latent human state z_t. A causal
    Transformer models state evolution across consecutive windows, and a
    learnable transition operator predicts z_{t+1} from z_t and the current
    motion/action embedding.
    """

    def __init__(
        self,
        input_size,
        num_classes,
        latent_size=128,
        heads=4,
        layers=2,
        dropout=0.2,
        instance_norm=False,
        causal_mask=True,
        disable_transition=False,
        disable_state_encoder=False,
        action_mode="ground_truth",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.instance_norm = instance_norm
        self.causal_mask = causal_mask
        self.disable_transition = disable_transition
        self.disable_state_encoder = disable_state_encoder
        self.action_mode = action_mode
        self.window_encoder = nn.Sequential(
            nn.Conv1d(input_size, latent_size, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(latent_size),
            nn.GELU(),
            nn.Conv1d(latent_size, latent_size, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(latent_size),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=heads,
            dim_feedforward=latent_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = None if disable_state_encoder else nn.TransformerEncoder(layer, num_layers=layers)
        self.action_embedding = None if disable_transition or action_mode == "independent" else nn.Embedding(num_classes, latent_size)
        self.transition = (
            None
            if disable_transition
            else nn.Sequential(
                nn.Linear(latent_size if action_mode == "independent" else latent_size * 2, latent_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size, latent_size),
            )
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Dropout(dropout),
            nn.Linear(latent_size, num_classes),
        )

    def forward(self, x, labels=None):
        batch, seq_len, steps, channels = x.shape
        if self.instance_norm:
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True).clamp_min(1e-5)
            x = (x - mean) / std
        flat = x.reshape(batch * seq_len, steps, channels).transpose(1, 2)
        z0 = self.window_encoder(flat).squeeze(-1).reshape(batch, seq_len, -1)
        mask = None
        if self.causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        z = z0 if self.disable_state_encoder else self.state_encoder(z0, mask=mask)
        logits = self.classifier(z)

        trans_loss = None
        if seq_len > 1 and not self.disable_transition:
            if self.action_mode == "independent":
                transition_input = z[:, :-1]
            else:
                if self.action_mode == "predicted":
                    action_prob = torch.softmax(logits[:, :-1], dim=-1).detach()
                    action = action_prob @ self.action_embedding.weight
                else:
                    if labels is None:
                        raise ValueError("Ground-truth action conditioning requires labels during training")
                    action = self.action_embedding(labels[:, :-1])
                transition_input = torch.cat([z[:, :-1], action], dim=-1)
            z_next = self.transition(transition_input)
            trans_loss = nn.functional.mse_loss(z_next, z[:, 1:].detach())
        return logits, trans_loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_sequences(x, y, subjects, seq_len, seq_stride):
    xs, ys = [], []
    # Keep adjacent windows from the same subject together.
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        idx = np.sort(idx)
        sx, sy = x[idx], y[idx]
        for start in range(0, len(sy) - seq_len + 1, seq_stride):
            end = start + seq_len
            xs.append(sx[start:end])
            ys.append(sy[start:end])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.int64)


def load_sequence_data(path, seq_len, seq_stride):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    arrays = {}
    for split in ["train", "val", "test"]:
        x, y, subjects = data[f"x_{split}"], data[f"y_{split}"], data[f"subject_{split}"]
        if x.ndim == 4 and y.ndim == 2:
            arrays[f"x_{split}"], arrays[f"y_{split}"] = x.astype(np.float32), y.astype(np.int64)
        else:
            arrays[f"x_{split}"], arrays[f"y_{split}"] = build_sequences(x, y, subjects, seq_len, seq_stride)
    return arrays, metadata


def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device, criterion, transition_weight, eval_mode):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, trans_loss = model(x, y)
            if eval_mode == "last":
                ce = criterion(logits[:, -1], y[:, -1])
                pred_batch = torch.argmax(logits[:, -1], dim=-1)
                target_batch = y[:, -1]
            else:
                ce = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                pred_batch = torch.argmax(logits, dim=-1).reshape(-1)
                target_batch = y.reshape(-1)
            loss = ce + transition_weight * (trans_loss if trans_loss is not None else 0.0)
            losses.append(float(loss.item()))
            preds.extend(pred_batch.cpu().numpy().reshape(-1).tolist())
            targets.extend(target_batch.cpu().numpy().reshape(-1).tolist())
    return (
        float(np.mean(losses)),
        accuracy_score(targets, preds),
        f1_score(targets, preds, average="macro", zero_division=0),
        np.asarray(targets),
        np.asarray(preds),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("processed/pamap2_paper10_50hz_128_random.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/tcst_pamap2_paper10_random"))
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--seq-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--transition-weight", type=float, default=0.2)
    p.add_argument("--class-balanced", action="store_true")
    p.add_argument("--instance-norm", action="store_true")
    p.add_argument("--no-causal-mask", action="store_true")
    p.add_argument("--disable-transition", action="store_true")
    p.add_argument("--disable-state-encoder", action="store_true")
    p.add_argument("--action-mode", choices=["ground_truth", "independent", "predicted"], default="ground_truth")
    p.add_argument("--eval-mode", choices=["all", "last"], default="all")
    p.add_argument("--selection-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    arrays, metadata = load_sequence_data(args.data, args.seq_len, args.seq_stride)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(metadata["activity_ids"])
    model = TCSTHAR(
        input_size=arrays["x_train"].shape[-1],
        num_classes=num_classes,
        latent_size=args.latent_size,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        instance_norm=args.instance_norm,
        causal_mask=not args.no_causal_mask,
        disable_transition=args.disable_transition,
        disable_state_encoder=args.disable_state_encoder,
        action_mode=args.action_mode,
    ).to(device)

    train_loader = make_loader(arrays["x_train"], arrays["y_train"], args.batch_size, True)
    val_loader = make_loader(arrays["x_val"], arrays["y_val"], args.batch_size, False)
    test_loader = make_loader(arrays["x_test"], arrays["y_test"], args.batch_size, False)
    weight = None
    if args.class_balanced:
        labels = arrays["y_train"].reshape(-1)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weight = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    best_score, best_f1, best_acc, best_epoch, bad = -1.0, -1.0, -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, trans_loss = model(x, y)
            if args.eval_mode == "last":
                ce = criterion(logits[:, -1], y[:, -1])
            else:
                ce = criterion(logits.reshape(-1, num_classes), y.reshape(-1))
            aux = trans_loss if trans_loss is not None else torch.zeros((), device=device)
            loss = ce + args.transition_weight * aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, criterion, args.transition_weight, args.eval_mode
        )
        sched.step(val_f1)
        print(
            f"epoch {epoch:03d} train_loss={np.mean(losses):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )
        score = val_acc if args.selection_metric == "accuracy" else val_f1
        if score > best_score:
            best_score, best_f1, best_acc, best_epoch, bad = score, val_f1, val_acc, epoch, 0
            torch.save({"model_state": model.state_dict(), "metadata": metadata, "args": vars(args)}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_f1, targets, preds = evaluate(
        model, test_loader, device, criterion, args.transition_weight, args.eval_mode
    )
    report = classification_report(
        targets,
        preds,
        labels=list(range(num_classes)),
        target_names=metadata["activity_names"],
        zero_division=0,
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "best_val_macro_f1": best_f1,
        "selection_metric": args.selection_metric,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "seq_len": args.seq_len,
        "transition_weight": args.transition_weight,
        "class_balanced": args.class_balanced,
        "instance_norm": args.instance_norm,
        "causal_mask": not args.no_causal_mask,
        "disable_transition": args.disable_transition,
        "disable_state_encoder": args.disable_state_encoder,
        "eval_mode": args.eval_mode,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
