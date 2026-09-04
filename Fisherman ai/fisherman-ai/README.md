# Fisherman AI V1

A computer-only prototype for environmental danger estimation. It is an experimental model, not a certified marine warning system.

## Architecture

`provider-neutral observations -> causal time-series features -> classifier -> calibrated/decision-layer risk score -> structured prediction`

- `src/fisherman_ai/schema.py`: standardized observation contract.
- `src/fisherman_ai/api/`: weather-provider boundary; providers map API responses into `WeatherObservation`.
- `src/fisherman_ai/features.py`: configurable lag, rolling mean/std, rate-of-change, and circular wind-direction features.
- `src/fisherman_ai/models/`: Logistic Regression baseline, Random Forest, persistence, inference, and model-derived feature ranking.
- `src/fisherman_ai/evaluation/`: metrics and confusion matrix.

## Data and labeling

The CSV must contain `timestamp`, `latitude`, `longitude`, environmental/motion columns, and integer `danger_label` values 0-3. The repository does not include real labels. Do not train or report real-world performance until labels are derived from authoritative marine warnings, observed conditions, or a documented safety standard. Use chronological holdout or blocked time-series validation; never random-split records from the same event across train and test.

The four classes are conceptual and configurable. Thresholds must be agreed from the selected authoritative source before training. Sensor motion should be treated as a vessel-state/context signal until evidence shows it improves environmental danger prediction.

## Install and test

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

An empty `data/raw/training.csv` is intentionally not a trainable dataset. The training command uses a chronological holdout, rather than a random split, to reduce event leakage. Train both model options and select based on holdout macro-F1, per-class recall, calibration, and especially dangerous-class false negatives.

## CLI

After installing dependencies and providing documented labels:

```powershell
python train.py data/raw/training.csv --model random_forest --output models/fisherman.joblib
python predict.py models/fisherman.joblib data/raw/recent_observations.csv
```

The prediction command emits JSON. The input observation file should contain the current row plus enough recent history for the configured lag and rolling windows.

## Risk score

The current decision layer maps class probabilities to expected severity points `(0, 33.33, 66.67, 100)`. This is a transparent prototype policy, not a claim that probability equals physical risk. Replace it with a policy approved for the actual operational context, and calibrate probabilities on held-out data before safety use.
