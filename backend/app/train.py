"""Pre-train the performance model into the bundled artifacts folder.

    python -m backend.app.train                 # performance model (seconds)
    python -m backend.app.injury.train          # injury models (minutes; see that module)
"""
import argparse
import json

from .ml_engine import BUNDLED_DIR, AthleteLensModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(BUNDLED_DIR))
    args = parser.parse_args()
    meta = AthleteLensModel(model_dir=args.out).train(out_dir=args.out)
    print(json.dumps({k: meta[k] for k in ("cohort_size", "train_rows", "test_rows", "sklearn_version", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
