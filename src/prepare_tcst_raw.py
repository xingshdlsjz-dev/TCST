import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    24: "rope_jumping",
}
ACTIVITY_IDS = list(ACTIVITY_NAMES)


def columns():
    names = ["timestamp", "activity_id", "heart_rate"]
    imu = [
        "temperature",
        "acc16_x",
        "acc16_y",
        "acc16_z",
        "acc6_x",
        "acc6_y",
        "acc6_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
        "orientation_1",
        "orientation_2",
        "orientation_3",
        "orientation_4",
    ]
    for place in ["hand", "chest", "ankle"]:
        names += [f"{place}_{field}" for field in imu]
    return names


def paper27_features():
    feats = []
    for place in ["hand", "chest", "ankle"]:
        for family in ["acc16", "gyro", "mag"]:
            feats += [f"{place}_{family}_x", f"{place}_{family}_y", f"{place}_{family}_z"]
    return feats


def full40_features():
    feats = ["heart_rate"]
    for place in ["hand", "chest", "ankle"]:
        feats += [
            f"{place}_temperature",
            f"{place}_acc16_x",
            f"{place}_acc16_y",
            f"{place}_acc16_z",
            f"{place}_acc6_x",
            f"{place}_acc6_y",
            f"{place}_acc6_z",
            f"{place}_gyro_x",
            f"{place}_gyro_y",
            f"{place}_gyro_z",
            f"{place}_mag_x",
            f"{place}_mag_y",
            f"{place}_mag_z",
        ]
    return feats


def add_magnitudes(df):
    for place in ["hand", "chest", "ankle"]:
        for family in ["acc16", "acc6", "gyro", "mag"]:
            cols = [f"{place}_{family}_x", f"{place}_{family}_y", f"{place}_{family}_z"]
            df[f"{place}_{family}_mag"] = np.sqrt(np.square(df[cols]).sum(axis=1))
    return df


def read_subject(path, features, add_mag):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns(), na_values="NaN")
    df = df[df["activity_id"].isin(ACTIVITY_IDS)].copy()
    base_features = full40_features()
    df[base_features] = df[base_features].ffill().bfill()
    if add_mag:
        df = add_magnitudes(df)
    return df[["timestamp", "activity_id"] + features]


def pool(df, features, pool_size):
    usable = len(df) - len(df) % pool_size
    df = df.iloc[:usable].copy()
    gid = np.arange(usable) // pool_size
    pooled = df[features].groupby(gid).mean()
    pooled.insert(0, "activity_id", df["activity_id"].groupby(gid).agg(lambda s: s.mode().iloc[0]).to_numpy())
    pooled.insert(0, "timestamp", df["timestamp"].groupby(gid).first().to_numpy())
    return pooled[["timestamp", "activity_id"] + features]


def windows(df, features, window_size, stride, min_ratio):
    x, y = [], []
    values = df[features].to_numpy(np.float32)
    labels = df["activity_id"].to_numpy(np.int64)
    label_map = {activity_id: i for i, activity_id in enumerate(ACTIVITY_IDS)}
    for start in range(0, len(labels) - window_size + 1, stride):
        end = start + window_size
        common, count = Counter(labels[start:end].tolist()).most_common(1)[0]
        if count / window_size >= min_ratio:
            x.append(values[start:end])
            y.append(label_map[int(common)])
    return np.stack(x).astype(np.float32), np.asarray(y, np.int64)


def sequences(x, y, subject, seq_len, seq_stride):
    xs, ys, ss = [], [], []
    for start in range(0, len(y) - seq_len + 1, seq_stride):
        end = start + seq_len
        xs.append(x[start:end])
        ys.append(y[start:end])
        ss.append(subject)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.int64), np.asarray(ss, np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("PAMAP2_Dataset"))
    p.add_argument("--output", type=Path, default=Path("processed/pamap2_tcst_paper10_seq6_random.npz"))
    p.add_argument("--pool-size", type=int, default=2)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--seq-stride", type=int, default=1)
    p.add_argument("--min-label-ratio", type=float, default=0.8)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-set", choices=["paper27", "full40", "full52"], default="paper27")
    p.add_argument("--split-mode", choices=["random", "subject"], default="random")
    p.add_argument("--train-subjects", type=str, default="101,102,103,104,105,106,109")
    p.add_argument("--val-subjects", type=str, default="107")
    p.add_argument("--test-subjects", type=str, default="108")
    args = p.parse_args()

    if args.feature_set == "paper27":
        features = paper27_features()
    elif args.feature_set == "full40":
        features = full40_features()
    else:
        features = full40_features()
        for place in ["hand", "chest", "ankle"]:
            for family in ["acc16", "acc6", "gyro", "mag"]:
                features.append(f"{place}_{family}_mag")
    subject_data = {}
    for path in sorted((args.data_root / "Protocol").glob("subject*.dat")):
        subject = int(path.stem.replace("subject", ""))
        df = pool(read_subject(path, features, add_mag=args.feature_set == "full52"), features, args.pool_size)
        x, y = windows(df, features, args.window_size, args.stride, args.min_label_ratio)
        if len(y) >= args.seq_len:
            sx, sy, ss = sequences(x, y, subject, args.seq_len, args.seq_stride)
            subject_data[subject] = (sx, sy, ss)

    arrays = {}
    if args.split_mode == "random":
        x = np.concatenate([item[0] for item in subject_data.values()])
        y = np.concatenate([item[1] for item in subject_data.values()])
        s = np.concatenate([item[2] for item in subject_data.values()])
        idx = np.arange(len(y))
        strat = y[:, -1]
        train_val, test = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=strat)
        train, val = train_test_split(train_val, test_size=args.val_size, random_state=args.seed, stratify=strat[train_val])
        for name, split_idx in {"train": train, "val": val, "test": test}.items():
            arrays[f"x_{name}"] = x[split_idx]
            arrays[f"y_{name}"] = y[split_idx]
            arrays[f"subject_{name}"] = s[split_idx]
        split_desc = "random_sequence_split_after_chronological_sequence_building"
    else:
        def parse_subjects(text):
            return [int(v.strip()) for v in text.split(",") if v.strip()]

        split_subjects = {
            "train": parse_subjects(args.train_subjects),
            "val": parse_subjects(args.val_subjects),
            "test": parse_subjects(args.test_subjects),
        }
        for name, subjects in split_subjects.items():
            chosen = [subject_data[s] for s in subjects if s in subject_data]
            arrays[f"x_{name}"] = np.concatenate([item[0] for item in chosen])
            arrays[f"y_{name}"] = np.concatenate([item[1] for item in chosen])
            arrays[f"subject_{name}"] = np.concatenate([item[2] for item in chosen])
        split_desc = split_subjects

    mean = np.nanmean(arrays["x_train"], axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = np.nanstd(arrays["x_train"], axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    for name in ["train", "val", "test"]:
        xx = np.where(np.isnan(arrays[f"x_{name}"]), mean, arrays[f"x_{name}"])
        arrays[f"x_{name}"] = ((xx - mean) / std).astype(np.float32)

    meta = {
        "activity_ids": ACTIVITY_IDS,
        "activity_names": [ACTIVITY_NAMES[i] for i in ACTIVITY_IDS],
        "features": features,
        "feature_set": args.feature_set,
        "sample_rate_hz": 50.0,
        "window_size": args.window_size,
        "window_stride": args.stride,
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "split_mode": args.split_mode,
        "splits": split_desc,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, mean=mean, std=std, metadata=json.dumps(meta, indent=2))
    print(f"Saved {args.output}")
    for name in ["train", "val", "test"]:
        print(name, arrays[f"x_{name}"].shape, arrays[f"y_{name}"].shape)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
