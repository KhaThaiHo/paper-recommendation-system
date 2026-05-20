import os
import json
import hashlib
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class PaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _hash_sequence(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_metadata(texts, labels, tokenizer, max_length, split_name) -> dict:
    return {
        "split_name": split_name,
        "num_samples": len(texts),
        "max_length": max_length,
        "tokenizer_name": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "texts_sha256": _hash_sequence(texts),
        "labels_sha256": _hash_sequence(labels),
    }


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def pretokenize_to_disk(texts, labels, tokenizer, max_length, cache_dir, split_name):
    """
    Tokenize all texts once and save as memory-mapped numpy arrays.
    This lets the DataLoader read directly from disk with minimal RAM usage.
    """
    os.makedirs(cache_dir, exist_ok=True)
    n = len(texts)

    input_ids_path = os.path.join(cache_dir, f"{split_name}_input_ids.npy")
    attention_mask_path = os.path.join(cache_dir, f"{split_name}_attention_mask.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.npy")
    metadata_path = os.path.join(cache_dir, f"{split_name}_metadata.json")
    expected_metadata = _cache_metadata(texts, labels, tokenizer, max_length, split_name)

    if (
        os.path.exists(input_ids_path)
        and os.path.exists(attention_mask_path)
        and os.path.exists(labels_path)
        and _load_json(metadata_path) == expected_metadata
    ):
        print(f"  [Cache hit] Loading pre-tokenized {split_name} from disk ...")
        input_ids = np.load(input_ids_path, mmap_mode="r")
        attention_mask = np.load(attention_mask_path, mmap_mode="r")
        labels_arr = np.load(labels_path)
        return input_ids, attention_mask, labels_arr

    if (
        os.path.exists(input_ids_path)
        or os.path.exists(attention_mask_path)
        or os.path.exists(labels_path)
    ):
        print(f"  [Cache stale] Re-tokenizing {split_name} because metadata changed ...")

    print(f"  [Pre-tokenizing {split_name}: {n} samples] ...")
    input_ids = np.lib.format.open_memmap(
        input_ids_path, mode="w+", dtype=np.int32, shape=(n, max_length)
    )
    attention_mask = np.lib.format.open_memmap(
        attention_mask_path, mode="w+", dtype=np.int8, shape=(n, max_length)
    )

    chunk_size = 2000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = tokenizer(
            texts[start:end],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids[start:end] = chunk["input_ids"].astype(np.int32)
        attention_mask[start:end] = chunk["attention_mask"].astype(np.int8)
        if start % 50000 == 0:
            print(f"    ... {end}/{n}")

    np.save(labels_path, np.array(labels, dtype=np.int32))
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(expected_metadata, file, indent=2)
    print(f"  [Pre-tokenized {split_name} saved to {cache_dir}]")

    input_ids = np.load(input_ids_path, mmap_mode="r")
    attention_mask = np.load(attention_mask_path, mmap_mode="r")
    labels_arr = np.load(labels_path)
    return input_ids, attention_mask, labels_arr


class DiskDataset(Dataset):
    """
    Reads tokenized data from memory-mapped numpy arrays.
    Only the requested batch rows are loaded from disk.
    """

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
