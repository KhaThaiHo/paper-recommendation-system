import os
import numpy as np
import torch
from torch.utils.data import Dataset

def pretokenize_dual_to_disk(paper_texts, journal_texts, labels, tokenizer, max_length, save_dir, split_name):
    os.makedirs(save_dir, exist_ok=True)
    n = len(labels)
    # Define paths
    paths = {
        key: os.path.join(save_dir, f"{split_name}_{key}.npy")
        for key in ["p_ids", "p_mask", "j_ids", "j_mask", "labels"]
    }

    if all(os.path.exists(p) for p in paths.values()):
        print(f"  [Cache hit] Loading DUAL {split_name} from disk ...")
        return (np.load(paths["p_ids"], mmap_mode="r"), np.load(paths["p_mask"], mmap_mode="r"),
                np.load(paths["j_ids"], mmap_mode="r"), np.load(paths["j_mask"], mmap_mode="r"), np.load(paths["labels"]))

    print(f"  [Pre-tokenizing DUAL {split_name}: {n} samples] ...")
    p_ids  = np.lib.format.open_memmap(paths["p_ids"], mode="w+", dtype=np.int32, shape=(n, max_length))
    p_mask = np.lib.format.open_memmap(paths["p_mask"], mode="w+", dtype=np.int8, shape=(n, max_length))
    j_ids  = np.lib.format.open_memmap(paths["j_ids"], mode="w+", dtype=np.int32, shape=(n, max_length))
    j_mask = np.lib.format.open_memmap(paths["j_mask"], mode="w+", dtype=np.int8, shape=(n, max_length))

    CHUNK = 2000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        cp = tokenizer(paper_texts[start:end], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        p_ids[start:end] = cp["input_ids"].astype(np.int32)
        p_mask[start:end] = cp["attention_mask"].astype(np.int8)

        cj = tokenizer(journal_texts[start:end], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        j_ids[start:end] = cj["input_ids"].astype(np.int32)
        j_mask[start:end] = cj["attention_mask"].astype(np.int8)

    np.save(paths["labels"], np.array(labels, dtype=np.int32))
    return (np.load(paths["p_ids"], mmap_mode="r"), np.load(paths["p_mask"], mmap_mode="r"),
            np.load(paths["j_ids"], mmap_mode="r"), np.load(paths["j_mask"], mmap_mode="r"), np.load(paths["labels"]))

class DualDiskDataset(Dataset):
    def __init__(self, p_ids, p_mask, j_ids, j_mask, labels):
        self.p_ids, self.p_mask, self.j_ids, self.j_mask, self.labels = p_ids, p_mask, j_ids, j_mask, labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "p_ids": torch.tensor(self.p_ids[idx], dtype=torch.long),
            "p_mask": torch.tensor(self.p_mask[idx], dtype=torch.long),
            "j_ids": torch.tensor(self.j_ids[idx], dtype=torch.long),
            "j_mask": torch.tensor(self.j_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }