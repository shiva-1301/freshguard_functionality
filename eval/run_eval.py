"""Run the FreshGuard evaluation module and write metrics to eval/metrics.json (FR21)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from freshguard.pipeline import FreshGuard          # noqa: E402
from freshguard.storage import Storage              # noqa: E402
from freshguard import evaluation                   # noqa: E402


def main() -> None:
    fg = FreshGuard(storage=Storage(ROOT / "artifacts" / "eval.db"))
    results = evaluation.run_all(fg)
    out = ROOT / "eval" / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
