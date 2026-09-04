import argparse
import json
from pathlib import Path

from fisherman_ai.data.loader import load_csv
from fisherman_ai.models.prediction import predict
from fisherman_ai.models.training import TrainedModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict environmental danger from recent observations.")
    parser.add_argument("model", type=Path)
    parser.add_argument("observations", type=Path)
    args = parser.parse_args()
    result = predict(TrainedModel.load(args.model), load_csv(args.observations))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()