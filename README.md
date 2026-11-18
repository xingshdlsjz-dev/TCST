# TCST-HAR Paper Reproducibility Package

This folder contains the code corresponding to the latest manuscript `main_revised_ablation (2).tex`.
It covers the four reported experiment groups:

- PAMAP2 matched six-window comparison, including CNN1D, LSTM, DeepConvLSTM, ResNet1D, TCN, InceptionTime, GRU-FCN, TimesNet, PatchTST, iTransformer, HARMamba-style, ModernTCN, and TCST-HAR.
- PAMAP2 all-six-window-position ablation study.
- PAMAP2 strict subject-independent evaluation and the three transition-conditioning variants.
- Opportunity subject-disjoint evaluation with S4+S2 training, S3 validation, and S1 testing.

The package uses the same paper settings: 50 Hz PAMAP2 preprocessing, 128-sample windows, 50% overlap, six-window temporal context, 27 PAMAP2 channels, 10 retained activities, AdamW, ReduceLROnPlateau, and seed 42 unless overridden.

## Environment

Use the project's existing conda environment with Python, PyTorch, NumPy, scikit-learn, and Matplotlib. From this directory:

```powershell
conda activate <your-environment>
python -m pip install -r requirements.txt
```

## Data

Processed data are intentionally not duplicated here. See `data/README.md` for the exact files produced by the existing preprocessing pipeline and the commands that copy or regenerate them.

## Main PAMAP2 comparison

```powershell
python src/train_pamap2_six_window.py `
  --data data/random_windows.npz `
  --output runs/pamap2_main `
  --epochs 60 `
  --batch-size 64 `
  --seed 42
```

Every model receives the same six consecutive `128 x 27` windows and is scored on the sixth-window label. TCST-HAR keeps the six-window hierarchy; baselines receive the same observations concatenated as `768 x 27`.

## PAMAP2 ablation

The ablation protocol evaluates all six window positions:

```powershell
python src/train_tcst_base.py `
  --data data/random_windows.npz `
  --output-dir runs/pamap2_ablation/full `
  --seq-len 6 `
  --eval-mode all `
  --epochs 60 `
  --batch-size 64 `
  --seed 42
```

Use `--no-causal-mask`, `--disable-transition`, `--disable-state-encoder`, or `--action-mode independent` for controlled variants. The transition-conditioning variants for the strict subject-independent experiment are handled by `src/train_revision4.py`.

## Strict subject-independent PAMAP2

```powershell
python src/train_revision4.py --data data/subject108_split.npz --out runs/subject_independent/full --model tcst --action-mode ground_truth --epochs 15 --seed 42
python src/train_revision4.py --data data/subject108_split.npz --out runs/subject_independent/label_independent --model tcst --action-mode independent --epochs 15 --seed 42
python src/train_revision4.py --data data/subject108_split.npz --out runs/subject_independent/predicted_label --model tcst --action-mode predicted --epochs 15 --seed 42
```

## Opportunity

The Opportunity preprocessing and subject-disjoint training entry points are:

```powershell
python src/prepare_opportunity.py --help
python src/train_opportunity.py `
  --data data/opportunity_locomotion_subject_split.npz `
  --output runs/opportunity `
  --epochs 30 `
  --batch-size 32 `
  --patience 5 `
  --seed 42
```

The split is S4+S2 for training, S3 for validation, and S1 for testing, with 113 channels, five locomotion labels, six consecutive windows, and stride 64.

## Evaluation and figures

`evaluate_tcst_last_window.py` evaluates a saved checkpoint on the main sixth-window unit. `evaluate_random_probability_metrics.py` computes micro-ROC-AUC and micro-AP. `plot_pamap2_six_window_figures.py` regenerates the manuscript performance figures from the output directories.
