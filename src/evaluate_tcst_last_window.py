"""Evaluate a saved TCST-HAR checkpoint on the sixth-window label only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.train_tcst import TCSTHAR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    data_path = args.data or Path(config["data"])
    raw = np.load(data_path, allow_pickle=False)
    metadata = json.loads(str(raw["metadata"]))
    x = torch.from_numpy(raw["x_test"].astype(np.float32))
    y = torch.from_numpy(raw["y_test"].astype(np.int64))
    classes = len(metadata["activity_ids"])

    model = TCSTHAR(
        input_size=x.shape[-1], num_classes=classes,
        latent_size=config["latent_size"], heads=config["heads"], layers=config["layers"],
        dropout=config["dropout"], instance_norm=config.get("instance_norm", False),
        causal_mask=not config.get("no_causal_mask", False),
        disable_transition=config.get("disable_transition", False),
        disable_state_encoder=config.get("disable_state_encoder", False),
        action_mode=config.get("action_mode", "ground_truth"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logits = []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(x, y), batch_size=64):
            logits.append(model(xb, yb)[0][:, -1])
    logits = torch.cat(logits)
    y_true = y[:, -1].numpy()
    y_prob = torch.softmax(logits, dim=-1).numpy()
    y_pred = y_prob.argmax(axis=1)
    y_bin = label_binarize(y_true, classes=np.arange(classes))
    metrics = {
        "test_sequences": int(len(y_true)),
        "errors": int((y_true != y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_roc_auc": float(roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")),
        "micro_average_precision": float(average_precision_score(y_bin, y_prob, average="micro")),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint": str(args.checkpoint),
        "data": str(data_path),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=metadata["activity_names"], zero_division=0), encoding="utf-8"
    )
    np.savez_compressed(args.output / "predictions.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob, class_names=np.asarray(metadata["activity_names"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
