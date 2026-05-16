# Paper Submission Recommendation

This project trains and benchmarks journal classification with:

- Standard BERT (ToMe OFF)
- BERT + Token Merging (ToMe ON)

The main entrypoint is `main.py`.

## 1) Requirements

- Windows 10/11
- Python 3.10+
- pip

## 2) Environment Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Required Input Files

You must provide 4 CSV files:

1. Train split (`--train_path`)
2. Validation split (`--val_path`)
3. Test split (`--test_path`)
4. Journal metadata (`--journal_path`)

### 3.1 Paper split columns

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

## 4) Text Combination Codes

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

## 5) Run Command (PowerShell)

Minimal example:

```powershell
python main.py `
  --train_path "D:\File\data\train.csv" `
  --val_path "D:\File\data\val.csv" `
  --test_path "D:\File\data\test.csv" `
  --journal_path "D:\File\data\journal.csv" `
  --text_combination "TAK"
```

Full example:

```powershell
python main.py `
  --train_path "D:\File\Preprocessed_data\train_set.csv" `
  --val_path "D:\File\Preprocessed_data\val_set.csv" `
  --test_path "D:\File\Preprocessed_data\test_set.csv" `
  --journal_path "D:\File\Preprocessed_data\journal_category.csv" `
  --text_combination "TAK" `
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
  --checkpoint_dir "./checkpoints" `
  --cache_dir "./tokenized_cache" `
  --model_name "dmis-lab/biobert-v1.1" `
  --run_mode "tome" `
  --accum_steps 1 `
  --log_dir "./logs"
```

Notes:

- `--journal_path` is required.
- `--run_mode` choices: `baseline`, `tome`, or `both` (default `both`).
- `--cache_dir` enables disk token cache. Clear it if `--max_length` or `--text_combination` changes.
- Use `--cache_dir ""` to disable disk cache and tokenize on the fly.
- `--model_name` must be a BERT-compatible model.

## 6) What You Will See

Console output includes:

- Dataset/preprocessing summary
- Epoch-wise training + validation logs
- Test metrics (Top-1/3/5/10)
- Average inference time
- Peak GPU memory (if CUDA is available)
- Final comparison table (ToMe OFF vs ToMe ON)

If `--log_dir` is set, two files are saved per run:

- session_<timestamp>.txt (full console log)
- session_<timestamp>.json (config + results)

## 7) Checkpoints

Saved in `--checkpoint_dir`:

- `ToMe_OFF_last.pt`, `ToMe_OFF_best.pt`
- `ToMe_ON_last.pt`, `ToMe_ON_best.pt`

## 8) Quick Troubleshooting

Error: missing column in split CSV

- Ensure selected `--text_combination` columns exist in the paper CSVs.

Error: missing journal columns

- Verify `--journal_label_col`, `--journal_category_col`, and `--journal_scope_col` names.

No GPU

- Training still runs on CPU, but slower.