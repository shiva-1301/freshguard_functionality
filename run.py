"""Convenience launcher: train the estimator if needed, then serve the API + UI.

Usage:
  python run.py            # serve on http://127.0.0.1:8000
  python run.py --train    # (re)train the freshness estimator and exit
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    if "--train" in sys.argv:
        from freshguard.data.make_freshness import train_and_save
        print(train_and_save())
        return
    import uvicorn
    uvicorn.run("freshguard.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
