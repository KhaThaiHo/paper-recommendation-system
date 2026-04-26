import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from helpers import BenchmarkResult, count_params, peak_memory_mb
from .BertClassifier import BertClassifier
from .PaperDataset import PaperDataset
from .ToMeBertAttention import get_tome_timer_stats, reset_tome_timer
from .preprocessing import PreprocessConfig, load_and_prepare_splits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    epoch_start = time.perf_counter()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time.perf_counter() - epoch_start
    return total_loss / max(len(loader), 1), epoch_time


def compute_topk_accuracy(all_logits: torch.Tensor, all_labels: list[int], k: int) -> float:
    _, topk_preds = torch.topk(all_logits, min(k, all_logits.size(1)), dim=-1)
    topk_preds = topk_preds.cpu().numpy()

    correct = 0
    for true_label, pred_k in zip(all_labels, topk_preds):
        if true_label in pred_k:
            correct += 1
    return correct / len(all_labels)


def evaluate(model, loader, device) -> tuple[dict, float]:
    model.eval()
    all_logits = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            t0 = time.perf_counter()
            logits = model(ids, mask)
            latencies.append(time.perf_counter() - t0)

            all_logits.append(logits.cpu())
            all_labels.extend(labels.tolist())

    all_logits = torch.cat(all_logits, dim=0)

    metrics = {
        "top1": compute_topk_accuracy(all_logits, all_labels, 1),
        "top3": compute_topk_accuracy(all_logits, all_labels, 3),
        "top5": compute_topk_accuracy(all_logits, all_labels, 5),
        "top10": compute_topk_accuracy(all_logits, all_labels, 10),
    }
    return metrics, float(np.mean(latencies))


def _build_checkpoint_paths(checkpoint_dir: str, mode_name: str) -> tuple[str, str]:
    safe_name = mode_name.replace(" ", "_")
    last_path = os.path.join(checkpoint_dir, f"{safe_name}_last.pt")
    best_path = os.path.join(checkpoint_dir, f"{safe_name}_best.pt")
    return last_path, best_path


def run_benchmark(
    train_path: str,
    val_path: str,
    test_path: str,
    num_epochs: int = 10,
    batch_size: int = 16,
    max_length: int = 256,
    tome_r: int = 8,
    learning_rate: float = 2e-5,
    early_stopping_patience: int = 3,
    checkpoint_dir: str = "./checkpoints",
    text_combination: str = "TAK",
    journal_path: Optional[str] = None,
    label_col: str = "Label",
    journal_label_col: str = "Categories",
    journal_category_col: str = "Categories",
    journal_scope_col: Optional[str] = None,
) -> tuple[BenchmarkResult, Optional[BenchmarkResult]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n{'=' * 55}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if not journal_path:
        raise ValueError("journal_path is required because preprocessing always joins train with journal data")

    preprocess_config = PreprocessConfig(
        text_combination=text_combination,
        label_col=label_col,
        journal_label_col=journal_label_col,
        journal_category_col=journal_category_col,
        journal_scope_col=journal_scope_col,
    )

    t_preprocess = time.perf_counter()
    prepared_data = load_and_prepare_splits(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        config=preprocess_config,
        journal_path=journal_path,
    )

    print(f"Text combination: {text_combination.upper()}")
    print(f"Classes: {prepared_data.num_labels}")
    print(
        f"Train: {len(prepared_data.x_train)} | "
        f"Val: {len(prepared_data.x_val)} | "
        f"Test: {len(prepared_data.x_test)}"
    )

    t_tokenize = time.perf_counter()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"  [Loaded tokenizer]: {time.perf_counter() - t_tokenize:.2f}s")

    t_dataset = time.perf_counter()
    train_ds = PaperDataset(prepared_data.x_train, prepared_data.y_train, tokenizer, max_length)
    val_ds = PaperDataset(prepared_data.x_val, prepared_data.y_val, tokenizer, max_length)
    test_ds = PaperDataset(prepared_data.x_test, prepared_data.y_test, tokenizer, max_length)
    print(f"  [Created datasets]: {time.perf_counter() - t_dataset:.2f}s")

    t_loader = time.perf_counter()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    print(f"  [Created dataloaders]: {time.perf_counter() - t_loader:.2f}s")
    print(f"  Total preprocessing & data prep: {time.perf_counter() - t_preprocess:.2f}s\n")

    results: list[BenchmarkResult] = []

    for use_tome in [False, True]:
        mode_name = "ToMe ON" if use_tome else "ToMe OFF"
        print(f"\n-- {mode_name} {'-' * 41}")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t_init = time.perf_counter()
        model = BertClassifier(prepared_data.num_labels, use_tome=use_tome, tome_r=tome_r).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        last_checkpoint_path, best_checkpoint_path = _build_checkpoint_paths(checkpoint_dir, mode_name)

        best_val_acc = 0.0
        patience_counter = 0
        start_epoch = 0

        if os.path.exists(last_checkpoint_path):
            print(f"Resuming from {last_checkpoint_path}")
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint["best_val_acc"]
            patience_counter = checkpoint["patience_counter"]

        t_init_s = time.perf_counter() - t_init
        print(f"  [Model initialized]: {t_init_s:.2f}s")

        total_train_time = 0.0
        epochs_trained = 0
        best_epoch = 0

        t_training_start = time.perf_counter()
        for epoch in range(start_epoch, num_epochs):
            reset_tome_timer()

            loss, epoch_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
            total_train_time += epoch_time
            epochs_trained += 1

            t_eval = time.perf_counter()
            val_metrics, _ = evaluate(model, val_loader, device)
            eval_time_s = time.perf_counter() - t_eval
            val_acc = val_metrics["top1"]

            tome_stats = get_tome_timer_stats()
            tome_info = ""
            if tome_stats["call_count"] > 0:
                tome_info = (
                    f"  ToMe merge: {tome_stats['total_s']:.2f}s "
                    f"({tome_stats['call_count']} calls)"
                )

            print(
                f"  Epoch {epoch + 1}/{num_epochs}  "
                f"loss={loss:.4f}  "
                f"train={epoch_time:.2f}s  "
                f"eval={eval_time_s:.2f}s  "
                f"val_acc={val_acc:.4f}{tome_info}",
                end="",
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "patience_counter": patience_counter,
                },
                last_checkpoint_path,
            )

            if val_acc > best_val_acc:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "epoch": epoch,
                    },
                    best_checkpoint_path,
                )
                best_val_acc = val_acc
                patience_counter = 0
                best_epoch = epoch + 1
                print(" *")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{early_stopping_patience})")
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch + 1} (best epoch: {best_epoch})")
                    break

        training_duration = time.perf_counter() - t_training_start
        print(f"  [Training completed]: {training_duration:.2f}s ({epochs_trained} epochs)")

        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        t_test = time.perf_counter()
        test_metrics, avg_s = evaluate(model, test_loader, device)
        test_duration_s = time.perf_counter() - t_test
        print(f"  [Testing completed]: {test_duration_s:.2f}s")

        result = BenchmarkResult(
            mode=mode_name,
            accuracy_top1=test_metrics["top1"],
            accuracy_top3=test_metrics["top3"],
            accuracy_top5=test_metrics["top5"],
            accuracy_top10=test_metrics["top10"],
            avg_inference_s=avg_s,
            peak_memory_mb=peak_memory_mb(device),
            total_params=count_params(model),
            epochs_trained=epochs_trained,
        )
        results.append(result)

        print("\n  Test Results:")
        print(f"    Top-1  Accuracy: {test_metrics['top1']:.4f}")
        print(f"    Top-3  Accuracy: {test_metrics['top3']:.4f}")
        print(f"    Top-5  Accuracy: {test_metrics['top5']:.4f}")
        print(f"    Top-10 Accuracy: {test_metrics['top10']:.4f}")
        print(f"    Avg batch inference: {avg_s:.2f} s")
        print(f"    Peak GPU memory: {peak_memory_mb(device):.1f} MB")
        print(f"    Total training time: {total_train_time:.2f}s")
        print(f"    Epochs trained: {epochs_trained}/{num_epochs}")

    baseline = next((result for result in results if result.mode == "ToMe OFF"), None)
    tome = next((result for result in results if result.mode == "ToMe ON"), None)

    if baseline is None and results:
        baseline = results[0]

    return baseline, tome
