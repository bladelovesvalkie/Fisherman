import argparse
import json
from pathlib import Path

from fisherman_ai.data.loader import load_csv
from fisherman_ai.evaluation.metrics import chronological_split, evaluate
from fisherman_ai.models.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Fisherman danger model.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", choices=("logistic_regression", "random_forest"), default="random_forest")
    parser.add_argument("--output", type=Path, default=Path("models/fisherman.joblib"))
    args = parser.parse_args()

    frame = load_csv(args.csv)
    training_frame, holdout_frame = chronological_split(frame)
    model = train(training_frame, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(json.dumps({"model": model.model_name, "output": str(args.output), "holdout_metrics": evaluate(model, holdout_frame)}))


if __name__ == "__main__":
    main()