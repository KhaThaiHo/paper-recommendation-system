# Paper Submission Recommendation

This project trains and benchmarks journal classification with:

- Standard BERT (`ToMe OFF`)
- BERT + Token Merging (`ToMe ON`)

The main entrypoint is `main.py`.

## 1) Requirements

- Windows 10/11
- Python 3.10+
- `pip`

## 2) Environment Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Required Input Files

You must provide 2 CSV files:

1. full dataset (`--df_path`)
2. journal metadata (`--journal_path`)

Optional for faster testing:

- `--sample_size` to run on a smaller subset with label-aware sampling.

### 3.1 Full Dataset Columns

Required:

- `Label` (or your custom `--label_col`)

Optional, depending on `--text_combination`:

- `Title` (`T`)
- `Abstract` (`A`)
- `Keywords` (`K`)

### 3.2 Journal columns

Required:

- Join label column (default `Categories`, configurable via `--journal_label_col`)
- Category text column (default `Categories`, configurable via `--journal_category_col`)

Scope/Aims column:

- By default, `Aims` is used when `S` is selected in `--text_combination`.
- You can override with `--journal_scope_col`.

## 4) Preprocessing Behavior (Current)

Preprocessing always does these steps:

1. Load full dataset CSV.
2. (Optional) sample a smaller subset while preserving label distribution.
3. Split internally into train/val/test.
4. Load journal CSV.
5. Join each split with journal data by label.
6. Build `text` field from selected feature codes in `--text_combination`.

Feature code mapping:

- `T` = `Title`
- `A` = `Abstract`
- `K` = `Keywords`
- `C` = journal categories field (`journal_categories`)
- `S` = journal scope/aims field (`journal_scope_aims`)

Examples:

- `TAK` -> Title + Abstract + Keywords
- `CS` -> Journal Categories + Scope/Aims
- `TAKCS` -> All fields combined

## 5) Run Command (Exact)

Use this command format (PowerShell):

```powershell
python main.py `
  --df_path "D:\File\Preprocessed_data\train_set.csv" `
  --journal_path "D:\File\Preprocessed_data\journal_category.csv" `
  --sample_size 5000 `
  --text_combination "TAKCS" `
  --label_col "Label" `
  --journal_label_col "Label" `
  --journal_category_col "Categories" `
  --journal_scope_col "Aims" `
  --num_epochs 20 `
  --batch_size 8 `
  --max_length 512 `
  --tome_r 8 `
  --learning_rate 2e-5 `
  --early_stopping_patience 3 `
  --checkpoint_dir "./checkpoints"
```

Or on one line:

```powershell
python main.py --train_path "D:\File\data\train.csv" --val_path "D:\File\data\val.csv" --test_path "D:\File\data\test.csv" --journal_path "D:\File\data\journal.csv" --text_combination "TAKCS" --label_col "Label" --journal_label_col "Categories" --journal_category_col "Categories" --journal_scope_col "Aims" --num_epochs 20 --batch_size 8 --max_length 128 --tome_r 8 --learning_rate 2e-5 --early_stopping_patience 3 --checkpoint_dir "./checkpoints"
```

Notes:

- `--journal_path` is required.
- `--sample_size` is optional. If omitted, all rows from `--df_path` are used.
- The run executes both `ToMe OFF` and `ToMe ON` sequentially.

## 6) What You Will See

Console output includes:

- Dataset/preprocessing summary
- Epoch-wise training + validation logs
- Test metrics (`Top-1/3/5/10`)
- Average inference time
- Peak GPU memory (if CUDA is available)
- Final comparison table (`ToMe OFF` vs `ToMe ON`)

## 7) Checkpoints

Saved in `--checkpoint_dir`:

- `ToMe_OFF_last.pt`, `ToMe_OFF_best.pt`
- `ToMe_ON_last.pt`, `ToMe_ON_best.pt`

## 8) Quick Troubleshooting

Error: missing column in split CSV

- Ensure selected `--text_combination` columns exist in the full dataset.

Error: missing journal columns

- Verify `--journal_label_col`, `--journal_category_col`, and `--journal_scope_col` names.

No GPU.

- Training still runs on CPU, but slower.