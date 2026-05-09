from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


ALLOWED_TEXT_CODES = {"T", "A", "K", "C", "S"}


@dataclass
class PreprocessConfig:
    text_combination: str = "TAK"
    label_col: str = "Label"
    journal_label_col: str = "Categories"
    journal_category_col: str = "Categories"
    journal_scope_col: str = "Aims"
    separator: str = " [SEP] "


@dataclass
class PreparedDataBundle:
    x_train: list[str]
    y_train: list[int]
    x_val: list[str]
    y_val: list[int]
    x_test: list[str]
    y_test: list[int]
    num_labels: int
    label_encoder: LabelEncoder


def _require_columns(df: pd.DataFrame, columns: list[str], df_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")


def _join_text_columns(df: pd.DataFrame, columns: list[str], separator: str) -> pd.Series:
    parts = [df[column].fillna("").astype(str).str.strip() for column in columns]
    text = parts[0]
    for part in parts[1:]:
        text = text + separator + part
    return text


def _normalize_text_combination(text_combination: str) -> list[str]:
    letters = [character for character in text_combination.upper() if character.isalpha()]
    if not letters:
        raise ValueError("text_combination must contain at least one of: T, A, K, C, S")

    invalid = sorted({character for character in letters if character not in ALLOWED_TEXT_CODES})
    if invalid:
        raise ValueError(
            f"Invalid text_combination codes: {invalid}. Allowed codes are T, A, K, C, S"
        )

    unique_letters: list[str] = []
    seen: set[str] = set()
    for character in letters:
        if character not in seen:
            unique_letters.append(character)
            seen.add(character)
    return unique_letters





def _attach_journal_features(
    split_df: pd.DataFrame,
    journal_df: pd.DataFrame,
    config: PreprocessConfig,
) -> pd.DataFrame:
    _require_columns(split_df, [config.label_col], "split dataframe")
    _require_columns(
        journal_df,
        [config.journal_label_col, config.journal_category_col],
        "journal dataframe",
    )
    scope_column = config.journal_scope_col if config.journal_scope_col else "Aims"

    journal_lookup = pd.DataFrame(
        {
            "_join_label": journal_df[config.journal_label_col].astype(str),
            "journal_categories": journal_df[config.journal_category_col],
            "journal_scope_aims": journal_df[scope_column],
        }
    ).drop_duplicates(subset=["_join_label"])

    merged = split_df.copy()
    merged["_join_label"] = merged[config.label_col].astype(str)

    merged = merged.merge(journal_lookup, on="_join_label", how="left")
    merged = merged.drop(columns=["_join_label"])
    merged["journal_scope_aims"] = merged["journal_scope_aims"].fillna("")
    merged["journal_categories"] = merged["journal_categories"].fillna("")
    return merged


def build_text_column(
    split_df: pd.DataFrame,
    config: PreprocessConfig,
    journal_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = _attach_journal_features(split_df.copy(), journal_df, config)

    selected_codes = _normalize_text_combination(config.text_combination)
    code_to_column = {
        "T": "Title",
        "A": "Abstract",
        "K": "Keywords",
        "C": "journal_categories",
        "S": "journal_scope_aims",
    }
    text_columns = [code_to_column[code] for code in selected_codes]

    _require_columns(merged_df, text_columns, "merged dataframe")
    merged_df["text"] = _join_text_columns(merged_df, text_columns, config.separator)
    return merged_df


def _drop_missing_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    _require_columns(df, [label_col], "split dataframe")
    return df.dropna(subset=[label_col]).reset_index(drop=True)


def _encode_split_labels(split_df: pd.DataFrame, label_to_id: dict[str, int], label_col: str, split_name: str) -> tuple[list[str], list[int]]:
    raw_labels = split_df[label_col].astype(str)
    known_mask = raw_labels.isin(label_to_id)

    dropped_count = (~known_mask).sum()
    if dropped_count > 0:
        print(f"[Warning] Dropping {dropped_count} samples in {split_name} split with unseen labels.")

    filtered_df = split_df.loc[known_mask].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"{split_name} split has no samples with labels seen in training split.")

    texts = filtered_df["text"].astype(str).tolist()
    encoded_labels = raw_labels[known_mask].map(label_to_id).astype(int).tolist()
    return texts, encoded_labels


def load_and_prepare_splits(
    df_path: str,
    config: PreprocessConfig,
    journal_path: str,
) -> PreparedDataBundle:
    print("[DEBUG] *** ENTERED load_and_prepare_splits() ***")
    if not journal_path:
        raise ValueError("journal_path is required because train/journal joining is always enabled")

    print(f"[DEBUG] Reading CSV from {df_path}")
    df = pd.read_csv(df_path)
    print(f"[DEBUG] CSV read complete, DataFrame has {len(df)} rows and {len(df.columns)} columns")
    print(f"[DEBUG] Starting drop_duplicates()...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[DEBUG] After drop_duplicates: {len(df)} samples")
    print(f"Loaded dataframe with {len(df)} samples from {df_path}")

    label_mask = df["Label"].between(1, 15) # = 13688 samples
    df_filtered = df[label_mask].sample(10000, random_state=42).reset_index(drop=True)
    print(f"Filtered to {len(df_filtered)} samples with Label between 1 and 15, then took a random sample of 10,000 for processing.")

    # Assuming the dataframe has a column indicating the split (e.g., 'split')
    train_df, temp_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Split into train ({len(train_df)} samples), validation ({len(val_df)} samples), and test ({len(test_df)} samples)")

    journal_df = pd.read_csv(journal_path)

    train_df = build_text_column(train_df, config, journal_df=journal_df)
    val_df = build_text_column(val_df, config, journal_df=journal_df)
    test_df = build_text_column(test_df, config, journal_df=journal_df)
    print(f"Built text column for all splits using text combination '{config.text_combination}'")

    train_df = _drop_missing_labels(train_df, config.label_col)
    val_df = _drop_missing_labels(val_df, config.label_col)
    test_df = _drop_missing_labels(test_df, config.label_col)
    print(f"Dropped samples with missing labels. Remaining samples - train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}")

    encoder = LabelEncoder()
    encoder.fit(train_df[config.label_col].astype(str))
    label_to_id = {label: idx for idx, label in enumerate(encoder.classes_)}
    print(f"Encoded labels in training split. Found {len(encoder.classes_)} unique labels.")

    x_train, y_train = _encode_split_labels(train_df, label_to_id, config.label_col, "train")
    x_val, y_val = _encode_split_labels(val_df, label_to_id, config.label_col, "validation")
    x_test, y_test = _encode_split_labels(test_df, label_to_id, config.label_col, "test")
    print(f"Encoded labels in validation and test splits. After filtering unseen labels - validation: {len(y_val)} samples, test: {len(y_test)} samples.")

    return PreparedDataBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        num_labels=len(encoder.classes_),
        label_encoder=encoder,
    )
