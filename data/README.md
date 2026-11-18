# Data placement

This package does not duplicate the large processed arrays or public raw datasets. Place the processed files below, or pass their paths explicitly with `--data`.

Required PAMAP2 files:

- `random_windows.npz`: the 10-class, 50 Hz, 128-sample PAMAP2 windows used by the matched six-window comparison and all-position ablation.
- `subject108_split.npz`: the strict subject-independent PAMAP2 split used for Section III-E and the conditioning ablation.

Required Opportunity file:

- `opportunity_locomotion_subject_split.npz`: five-class locomotion data with 113 channels, using S4+S2 for training, S3 for validation, and S1 for testing.

The existing processed files are stored under `revision4_experiments/data` and `revision4_experiments/opportunity/processed` in the parent project. Copy them into this directory or use absolute paths in the commands in the package README.
