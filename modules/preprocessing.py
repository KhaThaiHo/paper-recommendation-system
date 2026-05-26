from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import LabelEncoder

@dataclass
class PreprocessConfig:
    paper_cols: list = None
    journal_cols: list = None
    label_col: str = "Label"
    journal_label_col: str = "Categories"

    def __post_init__(self):
        if self.paper_cols is None:
            self.paper_cols = ["Title", "Abstract", "Keywords"]
        if self.journal_cols is None:
            self.journal_cols = ["Aims", "Categories"]

@dataclass
class PreparedDataBundle:
    p_train: list; j_train: list; y_train: list
    p_val: list; j_val: list; y_val: list
    p_test: list; j_test: list; y_test: list
    num_labels: int
    label_encoder: LabelEncoder

def _drop_missing_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return df.dropna(subset=[label_col]).reset_index(drop=True)

def merge_journal_info(paper_df: pd.DataFrame, journal_df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    available_journal_cols = [c for c in config.journal_cols if c in journal_df.columns]
    if not available_journal_cols: return paper_df
    join_cols = [config.journal_label_col] + available_journal_cols
    journal_subset = journal_df[join_cols].rename(columns={config.journal_label_col: config.label_col}).drop_duplicates(subset=[config.label_col]).reset_index(drop=True)
    return paper_df.merge(journal_subset, on=config.label_col, how="left")

def load_and_preprocess_dual(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    df = df.copy()
    # Paper branch
    p_parts = [df[f].fillna("").astype(str) for f in config.paper_cols if f in df.columns]
    df["paper_text"] = p_parts[0] if p_parts else ""
    for part in p_parts[1:]:
        df["paper_text"] += " [SEP] " + part

    # Journal branch
    j_parts = [df[f].fillna("").astype(str) for f in config.journal_cols if f in df.columns]
    df["journal_text"] = j_parts[0] if j_parts else ""
    for part in j_parts[1:]:
        df["journal_text"] += " [SEP] " + part
    return df

def _encode_split_labels(split_df: pd.DataFrame, label_to_id: dict, label_col: str):
    raw_labels = split_df[label_col].astype(str)
    known_mask = raw_labels.isin(label_to_id)
    filtered_df = split_df[known_mask].reset_index(drop=True)
    
    encoded_labels = raw_labels[known_mask].map(label_to_id).astype(int).tolist()
    return filtered_df["paper_text"].tolist(), filtered_df["journal_text"].tolist(), encoded_labels

def load_and_prepare_splits(train_path: str, val_path: str, test_path: str, config: PreprocessConfig, journal_path: str) -> PreparedDataBundle:
    train_df, val_df, test_df, journal_df = map(pd.read_csv, [train_path, val_path, test_path, journal_path])

    train_df = load_and_preprocess_dual(merge_journal_info(train_df, journal_df, config), config)
    val_df   = load_and_preprocess_dual(merge_journal_info(val_df, journal_df, config), config)
    test_df  = load_and_preprocess_dual(merge_journal_info(test_df, journal_df, config), config)

    train_df, val_df, test_df = map(lambda df: _drop_missing_labels(df, config.label_col), [train_df, val_df, test_df])

    encoder = LabelEncoder()
    encoder.fit(train_df[config.label_col].astype(str))
    label_to_id = {lbl: idx for idx, lbl in enumerate(encoder.classes_)}

    p_tr, j_tr, y_tr = _encode_split_labels(train_df, label_to_id, config.label_col)
    p_va, j_va, y_va = _encode_split_labels(val_df, label_to_id, config.label_col)
    p_te, j_te, y_te = _encode_split_labels(test_df, label_to_id, config.label_col)

    return PreparedDataBundle(p_tr, j_tr, y_tr, p_va, j_va, y_va, p_te, j_te, y_te, len(encoder.classes_), encoder)