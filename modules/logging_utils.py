import json
import os
import sys
from typing import Optional


class TeeLogger:
    """
    Mirrors every print() call to both the terminal and a log file by
    replacing sys.stdout.
    """

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        sys.stdout = self.terminal
        self.log.close()


def save_session_log(
    config: dict,
    baseline: Optional[object],
    tome: Optional[object],
    total_time_s: float,
    log_dir: str,
    session_id: str,
) -> str:
    def _result_to_dict(result: Optional[object]) -> Optional[dict]:
        if result is None:
            return None
        return {
            "mode": result.mode.strip(),
            "accuracy_top1": result.accuracy_top1,
            "accuracy_top3": result.accuracy_top3,
            "accuracy_top5": result.accuracy_top5,
            "accuracy_top10": result.accuracy_top10,
            "avg_inference_s": result.avg_inference_s,
            "peak_memory_mb": result.peak_memory_mb,
            "total_params": result.total_params,
            "epochs_trained": result.epochs_trained,
        }

    log_data = {
        "session_id": session_id,
        "timestamp": session_id,
        "config": config,
        "results": {
            "baseline": _result_to_dict(baseline),
            "tome": _result_to_dict(tome),
        },
        "total_time_s": round(total_time_s, 3),
    }

    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, f"session_{session_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n[Log saved -> JSON] {json_path}")
    return json_path
