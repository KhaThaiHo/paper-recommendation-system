import os
import time

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from modules.BertClassifier import BertClassifier
from modules.PaperDataset import DualDiskDataset


def compute_topk_accuracy(all_logits: torch.Tensor, all_labels: list[int], k: int) -> float:
    _, topk_preds = torch.topk(all_logits, min(k, all_logits.size(1)), dim=-1)
    topk_preds = topk_preds.cpu().numpy()

    correct = 0
    for true_label, pred_k in zip(all_labels, topk_preds):
        if true_label in pred_k:
            correct += 1
    return correct / len(all_labels)


def evaluate(model, loader, device, log_every: int = 50) -> tuple[dict, float]:
    model.eval()
    all_logits = []
    all_labels = []
    latencies = []
    total_batches = len(loader)
    eval_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            p_ids = batch["p_ids"].to(device)
            p_mask = batch["p_mask"].to(device)
            j_ids = batch["j_ids"].to(device)
            j_mask = batch["j_mask"].to(device)
            labels = batch["labels"]

            t0 = time.perf_counter()
            if device.type == "cuda":
                with autocast(device_type=device.type):
                    logits = model(p_ids, p_mask, j_ids, j_mask)
            else:
                logits = model(p_ids, p_mask, j_ids, j_mask)
            latencies.append(time.perf_counter() - t0)

            all_logits.append(logits.float().cpu())
            all_labels.extend(labels.tolist())

            if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.perf_counter() - eval_start
                avg_latency = sum(latencies) / max(len(latencies), 1)
                print(
                    f"  [Eval] batch {batch_idx}/{total_batches} "
                    f"| avg_latency={avg_latency:.4f}s "
                    f"| elapsed={elapsed:.1f}s",
                    flush=True,
                )

    all_logits = torch.cat(all_logits, dim=0)

    metrics = {
        "top1": compute_topk_accuracy(all_logits, all_labels, 1),
        "top3": compute_topk_accuracy(all_logits, all_labels, 3),
        "top5": compute_topk_accuracy(all_logits, all_labels, 5),
        "top10": compute_topk_accuracy(all_logits, all_labels, 10),
    }
    return metrics, float(np.mean(latencies))


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _resolve_use_tome(use_tome_arg: str, model_path: str) -> bool:
    if use_tome_arg == "true":
        return True
    if use_tome_arg == "false":
        return False

    name = os.path.basename(model_path).lower()
    if "tome_off" in name or "baseline" in name:
        return False
    if "tome_on" in name or "tome" in name:
        return True
    raise ValueError(
        "Unable to infer ToMe mode from model_path. "
        "Set 'use_tome' to 'true' or 'false' in config."
    )


def load_checkpoint(model_path: str, device: torch.device) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    print(f"[Load] Checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        print(f"[Load] Checkpoint keys: {list(checkpoint.keys())}")
    else:
        print(f"[Load] Checkpoint type: {type(checkpoint)}")
    return checkpoint


def build_model(
    checkpoint: dict,
    device: torch.device,
    use_tome: bool,
    model_name: str | None,
    num_labels: int | None,
    tome_r: int | None,
) -> torch.nn.Module:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    resolved_num_labels = num_labels or config.get("num_labels")
    if resolved_num_labels is None:
        raise ValueError("num_labels is required (not found in checkpoint or args).")

    resolved_model_name = model_name or config.get("model_name", "bert-base-uncased")
    resolved_tome_r = tome_r if tome_r is not None else config.get("tome_r", 8)

    print(
        "[Model] Building BertClassifier "
        f"| use_tome={use_tome} "
        f"| tome_r={resolved_tome_r} "
        f"| num_labels={resolved_num_labels} "
        f"| model_name={resolved_model_name}"
    )

    model = BertClassifier(
        num_labels=resolved_num_labels,
        use_tome=use_tome,
        tome_r=resolved_tome_r,
        model_name=resolved_model_name,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    print("[Model] Checkpoint weights loaded")
    return model


def load_tokenized_split(cache_dir: str, split_name: str) -> DualDiskDataset:
    p_ids_path = os.path.join(cache_dir, f"{split_name}_p_ids.npy")
    p_mask_path = os.path.join(cache_dir, f"{split_name}_p_mask.npy")
    j_ids_path = os.path.join(cache_dir, f"{split_name}_j_ids.npy")
    j_mask_path = os.path.join(cache_dir, f"{split_name}_j_mask.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.npy")

    missing = [
        path
        for path in [p_ids_path, p_mask_path, j_ids_path, j_mask_path, labels_path]
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing tokenized files:\n" + "\n".join(missing)
        )

    print(f"[Data] Loading tokenized split '{split_name}' from {cache_dir}")
    p_ids = np.load(p_ids_path, mmap_mode="r")
    p_mask = np.load(p_mask_path, mmap_mode="r")
    j_ids = np.load(j_ids_path, mmap_mode="r")
    j_mask = np.load(j_mask_path, mmap_mode="r")
    labels = np.load(labels_path)
    print(
        f"[Data] p_ids shape={p_ids.shape}, "
        f"p_mask shape={p_mask.shape}, "
        f"labels shape={labels.shape}"
    )
    return DualDiskDataset(p_ids, p_mask, j_ids, j_mask, labels)


def run_inference(
    model_path: str,
    cache_dir: str,
    split_name: str,
    batch_size: int,
    use_tome_arg: str,
    model_name: str | None,
    num_labels: int | None,
    tome_r: int | None,
    log_every: int = 50,
) -> tuple[dict, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run] Device: {device}")
    print(
        "[Run] Config "
        f"| model_path={model_path} "
        f"| cache_dir={cache_dir} "
        f"| split={split_name} "
        f"| batch_size={batch_size}"
    )
    checkpoint = load_checkpoint(model_path, device)

    use_tome = _resolve_use_tome(use_tome_arg, model_path)
    print(f"[Run] use_tome={use_tome} (from '{use_tome_arg}')")
    model = build_model(
        checkpoint=checkpoint,
        device=device,
        use_tome=use_tome,
        model_name=model_name,
        num_labels=num_labels,
        tome_r=tome_r,
    )

    dataset = load_tokenized_split(cache_dir, split_name)
    print(f"[Data] Dataset size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size)
    print(f"[Data] Dataloader batches: {len(loader)}")
    return evaluate(model, loader, device, log_every=log_every)


DEFAULT_CONFIG: dict = {
    "model_path": "",
    "cache_dir": "./tokenized_cache",
    "split_name": "test",
    "batch_size": 16,
    "use_tome": "auto",
    "model_name": None,
    "num_labels": None,
    "tome_r": 8,
    "log_every": 1000,
}


def main(config: dict | None = None) -> None:
    run_config = DEFAULT_CONFIG.copy()
    if config:
        run_config.update(config)

    model_path = run_config["model_path"]
    if not model_path or "path/to" in model_path:
        raise ValueError("Set config['model_path'] to a valid checkpoint path.")

    metrics, latency = run_inference(
        model_path=model_path,
        cache_dir=run_config["cache_dir"],
        split_name=run_config["split_name"],
        batch_size=run_config["batch_size"],
        use_tome_arg=run_config["use_tome"],
        model_name=run_config["model_name"],
        num_labels=run_config["num_labels"],
        tome_r=run_config["tome_r"],
        log_every=run_config["log_every"],
    )

    print(f"Metrics: {metrics}")
    print(f"Average Latency: {latency:.4f} seconds")


if __name__ == "__main__":
    main()
