"""Reproduce probability-based test metrics for random-window baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from train_random_baselines import HARMamba, ModernTCN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../data/random_windows.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["harmamba", "moderntcn"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    x_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = data["y_test"].astype(np.int64)
    num_classes = len(metadata["activity_names"])

    model = HARMamba(27, num_classes) if args.model == "harmamba" else ModernTCN(27, num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    probabilities = []
    loader = DataLoader(TensorDataset(x_test), batch_size=128, shuffle=False)
    with torch.no_grad():
        for (x_batch,) in loader:
            probabilities.append(torch.softmax(model(x_batch), dim=-1).cpu().numpy())
    y_prob = np.concatenate(probabilities, axis=0)
    y_bin = label_binarize(y_test, classes=np.arange(num_classes))

    result = {
        "model": args.model,
        "test_count": int(len(y_test)),
        "micro_roc_auc": float(roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")),
        "micro_average_precision": float(average_precision_score(y_bin, y_prob, average="micro")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
