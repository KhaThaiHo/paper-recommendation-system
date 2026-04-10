# Paper Submission Recommendation

This project benchmarks academic paper classification with BERT/BioBERT, comparing:
- Standard BERT (without Token Merging)
- BERT + ToMe (Token Merging)

## 1) Requirements

- Windows 10/11 (the current project setup is on Windows)
- Python 3.10+ (3.10 or 3.11 recommended)
- Latest pip

## 2) Environment Setup

Create Virtual Environment:

```powershell
python -m venv .venv
```

Activate the venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Download the dependencies:

```powershell
pip install -r requirements.txt
```

## 3) Prepare the Dataset

In the current scripts, the dataset is loaded from:

`D:\File\train_dataset\train_set.csv`

The CSV file must contain at least these columns:
- `Title`
- `Abstract`
- `Keywords`
- `Label`

If your dataset is in a different path, update the `pd.read_csv(...)` line in the corresponding script.

## 4) Run the Project

### Run BERT + ToMe benchmark

```powershell
python ToMe_Bert_Classify.py
```

### Run BioBERT + ToMe benchmark

```powershell
python test_claude_biobert.py
```

After execution, the console will print:
- Accuracy
- Average inference time
- Peak GPU memory (neu co CUDA)
- Comparison between ToMe OFF and ToMe ON

## 5) Quick Notes


- The notebooks (`test.ipynb`, `test_claude.ipynb`) can be used for interactive experiments.
- If no GPU is available, the scripts will still run on CPU but more slowly.