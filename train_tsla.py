from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tsla_quant.data import fetch_tsla_data
from tsla_quant.features import build_features
from tsla_quant.labeling import make_labels
from tsla_quant.model import save_model, walk_forward_train
from tsla_quant.evaluate import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TSLA quant model with walk-forward validation")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model_out", type=str, default="models/tsla_lgbm.joblib")
    args = parser.parse_args()

    price_df = fetch_tsla_data(start=args.start, end=args.end, interval="1d")
    features = build_features(price_df)
    y, y_dir = make_labels(price_df, horizon=args.horizon, method="direction")

    # Align features and labels
    data = features.join(y).dropna()
    X = data.drop(columns=["y"])  # type: ignore[assignment]
    y_final = data["y"].astype(int)

    model, wf_result = walk_forward_train(X, y_final, n_splits=args.splits)

    # Tune probability threshold on OOF to maximize accuracy
    mask = wf_result.oof_probabilities.notna()
    probs_oof = wf_result.oof_probabilities[mask]
    y_true_oof = y_final.loc[probs_oof.index]
    best_thresh = args.threshold
    best_acc = -1.0
    for t in np.linspace(0.3, 0.7, 81):
        preds_t = (probs_oof >= t).astype(int)
        acc_t = (preds_t.values == y_true_oof.values).mean()
        if acc_t > best_acc:
            best_acc = acc_t
            best_thresh = float(t)

    metrics = evaluate_predictions(
        price_df.loc[X.index],
        (wf_result.oof_probabilities >= best_thresh).astype(int),
        wf_result.oof_probabilities,
        trading_threshold=best_thresh,
        horizon=args.horizon,
    )

    print("Walk-forward metrics (tuned threshold):")
    print(f"best_threshold: {best_thresh:.3f} | oof_acc: {best_acc:.4f}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save model
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(model_out))
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    main()

