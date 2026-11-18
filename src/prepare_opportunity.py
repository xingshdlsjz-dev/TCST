import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
LABEL_NAMES = ["Null", "Stand", "Walk", "Sit", "Lie"]

def selected_columns():
    delete = np.concatenate([
        np.arange(46, 50), np.arange(59, 63), np.arange(72, 76),
        np.arange(85, 89), np.arange(98, 102), np.arange(134, 243),
        np.arange(244, 249),
    ])
    keep = np.delete(np.arange(250), delete)
    assert len(keep) == 116 and keep[114] == 243
    return keep[1:114], keep[114]

def load_file(path, feature_cols, label_col):
    raw = np.loadtxt(path, dtype=np.float32, usecols=np.r_[feature_cols, label_col])
    x = pd.DataFrame(raw[:, :-1]).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)
    y_raw = raw[:, -1].astype(np.int64)
    y = np.asarray([LABEL_MAP.get(int(v), 0) for v in y_raw], dtype=np.int64)
    return x, y

def make_windows(x, y, size, stride, min_ratio):
    xs, ys = [], []
    for start in range(0, len(y) - size + 1, stride):
        labels = y[start:start + size]
        counts = np.bincount(labels, minlength=len(LABEL_NAMES))
        label = int(counts.argmax())
        if counts[label] / size >= min_ratio:
            xs.append(x[start:start + size])
            ys.append(label)
    return np.asarray(xs, dtype=np.float16), np.asarray(ys, dtype=np.int64)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--min-ratio", type=float, default=0.8)
    args = p.parse_args()
    feature_cols, label_col = selected_columns()
    split_subjects = {"train": [4, 2], "val": [3], "test": [1]}
    files = {split: sorted([p for s in subjects for p in args.dataset.glob(f"S{s}-*.dat")]) for split, subjects in split_subjects.items()}

    total = np.zeros(len(feature_cols), np.float64); total2 = np.zeros_like(total); count = 0
    for path in files["train"]:
        x, _ = load_file(path, feature_cols, label_col)
        total += x.sum(0); total2 += np.square(x, dtype=np.float64).sum(0); count += len(x)
    mean = total / count; std = np.sqrt(np.maximum(total2 / count - mean ** 2, 1e-8))

    output = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    summary = {}
    for split, paths in files.items():
        all_x, all_y, all_group = [], [], []
        for group, path in enumerate(paths):
            x, y = load_file(path, feature_cols, label_col)
            x = (x - mean) / std
            wx, wy = make_windows(x, y, args.window, args.stride, args.min_ratio)
            all_x.append(wx); all_y.append(wy); all_group.append(np.full(len(wy), group, np.int16))
        output[f"x_{split}"] = np.concatenate(all_x)
        output[f"y_{split}"] = np.concatenate(all_y)
        output[f"group_{split}"] = np.concatenate(all_group)
        summary[split] = {"windows": int(len(output[f"y_{split}"])), "class_counts": np.bincount(output[f"y_{split}"], minlength=5).tolist(), "files": [p.name for p in paths]}
    metadata = {"dataset": "Opportunity", "task": "locomotion", "classes": LABEL_NAMES, "channels": 113, "window": args.window, "stride": args.stride, "sequence_windows": 6, "splits": split_subjects, "summary": summary}
    output["metadata"] = json.dumps(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__": main()
