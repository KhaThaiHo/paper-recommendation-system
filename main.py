"""Entrypoint for benchmark training with modularized pipeline."""

import time

from helpers import parse_args, print_comparison
from modules.trainer import run_benchmark


def main() -> None:
    args = parse_args()

    script_start = time.perf_counter()

    baseline, tome = run_benchmark(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tome_r=args.tome_r,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        text_combination=args.text_combination,
        journal_path=args.journal_path,
        label_col=args.label_col,
        journal_label_col=args.journal_label_col,
        journal_category_col=args.journal_category_col,
        journal_scope_col=args.journal_scope_col,
    )

    print_comparison(baseline, tome)

    total_time = time.perf_counter() - script_start
    print(f"\n{'=' * 55}")
    print(f"  TOTAL SCRIPT EXECUTION TIME: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
