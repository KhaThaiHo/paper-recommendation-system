"""Entrypoint for benchmark training with modularized pipeline."""

import datetime
import os
import sys
import time

from helpers import parse_args, print_comparison
from modules.logging_utils import TeeLogger, save_session_log
from modules.trainer import run_benchmark


def main() -> None:
    args = parse_args()

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = None

    if args.log_dir:
        log_txt_path = os.path.join(args.log_dir, f"session_{session_id}.txt")
        logger = TeeLogger(log_txt_path)
        sys.stdout = logger
        print(f"[Session {session_id}] Log file: {log_txt_path}")

    try:
        script_start = time.perf_counter()

        cache_dir = args.cache_dir or None
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
            cache_dir=cache_dir,
            model_name=args.model_name,
            run_mode=args.run_mode,
            accum_steps=args.accum_steps,
            log_every=args.log_every,
        )

        if baseline is not None:
            print_comparison(baseline, tome)
        elif tome is not None:
            print_comparison(tome, None)

        total_time = time.perf_counter() - script_start
        print(f"\n{'=' * 55}")
        print(f"  TOTAL SCRIPT EXECUTION TIME: {total_time:.2f}s ({total_time / 60:.2f} min)")
        print(f"{'=' * 55}")

        if args.log_dir:
            config = {
                "run_mode": args.run_mode,
                "text_combination": args.text_combination,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "tome_r": args.tome_r,
                "learning_rate": args.learning_rate,
                "early_stopping_patience": args.early_stopping_patience,
                "train_path": args.train_path,
                "val_path": args.val_path,
                "test_path": args.test_path,
                "journal_path": args.journal_path,
                "label_col": args.label_col,
                "journal_label_col": args.journal_label_col,
                "journal_category_col": args.journal_category_col,
                "journal_scope_col": args.journal_scope_col,
                "checkpoint_dir": args.checkpoint_dir,
                "cache_dir": cache_dir,
                "model_name": args.model_name,
                "accum_steps": args.accum_steps,
                "log_every": args.log_every,
            }
            save_session_log(
                config=config,
                baseline=baseline,
                tome=tome,
                total_time_s=total_time,
                log_dir=args.log_dir,
                session_id=session_id,
            )
            print(f"[Log saved -> TXT]  {log_txt_path}")
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
