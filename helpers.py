import argparse
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BenchmarkResult:
	mode: str
	accuracy_top1: float
	accuracy_top3: float
	accuracy_top5: float
	accuracy_top10: float
	avg_inference_s: float
	peak_memory_mb: float
	total_params: int
	epochs_trained: int


def peak_memory_mb(device) -> float:
	if device.type == "cuda":
		return torch.cuda.max_memory_allocated(device) / 1e6
	return 0.0


def count_params(model) -> int:
	return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def print_comparison(baseline: BenchmarkResult, tome: Optional[BenchmarkResult]) -> None:
	if tome is None:
		print("\n" + "=" * 75)
		print("  SINGLE RUN SUMMARY")
		print("=" * 75)
		print(f"Mode: {baseline.mode}")
		print(f"Top-1 Accuracy:  {baseline.accuracy_top1:.4f}")
		print(f"Top-3 Accuracy:  {baseline.accuracy_top3:.4f}")
		print(f"Top-5 Accuracy:  {baseline.accuracy_top5:.4f}")
		print(f"Top-10 Accuracy: {baseline.accuracy_top10:.4f}")
		print(f"Avg Inference (s): {baseline.avg_inference_s:.2f}")
		print(f"Peak GPU Memory (MB): {baseline.peak_memory_mb:.1f}")
		print(f"Epochs Trained: {baseline.epochs_trained}")
		print("=" * 75)
		return

	speedup = baseline.avg_inference_s / (tome.avg_inference_s + 1e-9)

	print("\n" + "=" * 75)
	print("  COMPARISON SUMMARY - TOP-K ACCURACY TABLE")
	print("=" * 75)
	print(f"{'Metric':<25} {'Baseline':>15} {'ToMe':>15} {'Delta':>15}")
	print("-" * 75)

	delta_top1 = (tome.accuracy_top1 - baseline.accuracy_top1) * 100
	delta_top3 = (tome.accuracy_top3 - baseline.accuracy_top3) * 100
	delta_top5 = (tome.accuracy_top5 - baseline.accuracy_top5) * 100
	delta_top10 = (tome.accuracy_top10 - baseline.accuracy_top10) * 100

	print(f"{'Top-1 Accuracy':<25} {baseline.accuracy_top1:>15.4f} {tome.accuracy_top1:>15.4f} {delta_top1:>14.2f}%")
	print(f"{'Top-3 Accuracy':<25} {baseline.accuracy_top3:>15.4f} {tome.accuracy_top3:>15.4f} {delta_top3:>14.2f}%")
	print(f"{'Top-5 Accuracy':<25} {baseline.accuracy_top5:>15.4f} {tome.accuracy_top5:>15.4f} {delta_top5:>14.2f}%")
	print(f"{'Top-10 Accuracy':<25} {baseline.accuracy_top10:>15.4f} {tome.accuracy_top10:>15.4f} {delta_top10:>14.2f}%")
	print("-" * 75)
	print(
		f"{'Avg Inference (s)':<25} "
		f"{baseline.avg_inference_s:>15.2f} "
		f"{tome.avg_inference_s:>15.2f} "
		f"{(tome.avg_inference_s - baseline.avg_inference_s):>14.2f}"
	)
	print(
		f"{'Peak GPU Memory (MB)':<25} "
		f"{baseline.peak_memory_mb:>15.1f} "
		f"{tome.peak_memory_mb:>15.1f} "
		f"{(tome.peak_memory_mb - baseline.peak_memory_mb):>14.1f}"
	)
	print(f"{'Epochs Trained':<25} {baseline.epochs_trained:>15} {tome.epochs_trained:>15}")
	print("-" * 75)
	print(f"  Speed-up factor: {speedup:.2f}x")
	print("=" * 75)


def parse_args():
	parser = argparse.ArgumentParser(description="Run benchmark with configurable parameters")

	parser.add_argument("--train_path", type=str, required=True)
	parser.add_argument("--val_path", type=str, required=True)
	parser.add_argument("--test_path", type=str, required=True)
	parser.add_argument("--journal_path", type=str, required=True)

	parser.add_argument("--text_combination", type=str, default="TAK")
	parser.add_argument("--label_col", type=str, default="Label")
	parser.add_argument("--journal_label_col", type=str, default="Categories")
	parser.add_argument("--journal_category_col", type=str, default="Categories")
	parser.add_argument("--journal_scope_col", type=str, default=None)

	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--max_length", type=int, default=128)
	parser.add_argument("--tome_r", type=int, default=8)
	parser.add_argument("--learning_rate", type=float, default=2e-5)
	parser.add_argument("--early_stopping_patience", type=int, default=3)
	parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

	return parser.parse_args()
