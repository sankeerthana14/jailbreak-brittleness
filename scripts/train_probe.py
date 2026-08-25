#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train layer-wise linear harmfulness probes on "
            "XSTest activations using stratified 5-fold CV."
        )
    )

    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to XSTest activations.pt file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save probe results and final probe.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splitting and probe training.",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )

    return parser.parse_args()


# ============================================================
# Probe constructor
# ============================================================

def make_probe(seed, C):
    return LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load activation file
    # --------------------------------------------------------

    data = torch.load(
        args.activations,
        map_location="cpu",
        weights_only=False,
    )

    activations = (
        data["activations"]
        .float()
        .numpy()
    )

    labels = (
        data["labels"]
        .cpu()
        .numpy()
        .astype(int)
    )

    sample_ids = np.array(
        data["sample_ids"]
    )

    model_name = data["model"]

    # Expected shape:
    #
    # [num_samples, num_layers, hidden_size]

    if activations.ndim != 3:
        raise ValueError(
            f"Expected activation tensor with 3 dimensions, "
            f"got {activations.shape}"
        )

    n_samples, n_layers, hidden_size = (
        activations.shape
    )

    print("=" * 70)
    print("PROBE TRAINING")
    print("=" * 70)

    print(f"Model:       {model_name}")
    print(f"Samples:     {n_samples}")
    print(f"Layers:      {n_layers}")
    print(f"Hidden size: {hidden_size}")

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    print(
        "Labels:",
        dict(zip(unique, counts)),
    )

    if n_samples != len(labels):
        raise ValueError(
            "Number of activations and labels does not match."
        )

    # --------------------------------------------------------
    # CV splitter
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_results = []

    # ========================================================
    # Layer-wise 5-fold CV
    # ========================================================

    for layer_idx in range(n_layers):

        print(
            f"\nLayer {layer_idx + 1}/{n_layers}"
        )

        X_layer = activations[
            :,
            layer_idx,
            :
        ]

        for fold_idx, (
            train_idx,
            test_idx,
        ) in enumerate(
            skf.split(
                X_layer,
                labels,
            ),
            start=1,
        ):

            X_train = X_layer[train_idx]
            X_test = X_layer[test_idx]

            y_train = labels[train_idx]
            y_test = labels[test_idx]

            # -----------------------------------------------
            # IMPORTANT:
            # Fit scaler using TRAIN fold only.
            #
            # Never fit preprocessing on the test fold.
            # -----------------------------------------------

            scaler = StandardScaler()

            X_train_scaled = (
                scaler.fit_transform(
                    X_train
                )
            )

            X_test_scaled = (
                scaler.transform(
                    X_test
                )
            )

            probe = make_probe(
                seed=args.seed,
                C=args.C,
            )

            probe.fit(
                X_train_scaled,
                y_train,
            )

            # Continuous harmfulness probability
            y_score = probe.predict_proba(
                X_test_scaled
            )[:, 1]

            y_pred = (
                y_score >= 0.5
            ).astype(int)

            auroc = roc_auc_score(
                y_test,
                y_score,
            )

            auprc = average_precision_score(
                y_test,
                y_score,
            )

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y_test,
                    y_pred,
                )
            )

            f1 = f1_score(
                y_test,
                y_pred,
            )

            fold_results.append(
                {
                    "model": model_name,
                    "layer": layer_idx,
                    "relative_depth": (
                        (layer_idx + 1)
                        / n_layers
                    ),
                    "fold": fold_idx,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "auroc": auroc,
                    "auprc": auprc,
                    "accuracy": accuracy,
                    "balanced_accuracy": (
                        balanced_accuracy
                    ),
                    "f1": f1,
                }
            )

    # ========================================================
    # Save individual fold results
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_path = (
        args.output_dir
        / "cv_results.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False,
    )

    # ========================================================
    # Aggregate across folds
    # ========================================================

    layer_summary = (
        fold_df
        .groupby(
            [
                "model",
                "layer",
                "relative_depth",
            ],
            as_index=False,
        )
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),

            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),

            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),

            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),

            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
        )
    )

    summary_path = (
        args.output_dir
        / "layer_summary.csv"
    )

    layer_summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Select best layer
    #
    # Primary criterion:
    # mean cross-validated AUROC
    # ========================================================

    best_row = (
        layer_summary
        .sort_values(
            [
                "mean_auroc",
                "mean_auprc",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    best_layer = int(
        best_row["layer"]
    )

    print()
    print("=" * 70)
    print("BEST LAYER")
    print("=" * 70)

    print(
        f"Layer index: {best_layer}"
    )

    print(
        f"Relative depth: "
        f"{best_row['relative_depth']:.3f}"
    )

    print(
        f"CV AUROC: "
        f"{best_row['mean_auroc']:.4f} "
        f"+/- {best_row['std_auroc']:.4f}"
    )

    print(
        f"CV AUPRC: "
        f"{best_row['mean_auprc']:.4f} "
        f"+/- {best_row['std_auprc']:.4f}"
    )

    # ========================================================
    # Train FINAL probe on ALL XSTest examples
    #
    # This frozen probe is what we later apply to JBB.
    # ========================================================

    X_final = activations[
        :,
        best_layer,
        :
    ]

    final_scaler = StandardScaler()

    X_final_scaled = (
        final_scaler.fit_transform(
            X_final
        )
    )

    final_probe = make_probe(
        seed=args.seed,
        C=args.C,
    )

    final_probe.fit(
        X_final_scaled,
        labels,
    )

    # ========================================================
    # Save final scaler + probe
    # ========================================================

    scaler_path = (
        args.output_dir
        / "scaler.joblib"
    )

    probe_path = (
        args.output_dir
        / "probe.joblib"
    )

    joblib.dump(
        final_scaler,
        scaler_path,
    )

    joblib.dump(
        final_probe,
        probe_path,
    )

    # ========================================================
    # Save metadata
    # ========================================================

    metadata = {
        "model": model_name,
        "training_dataset": (
            data["dataset"]
        ),
        "num_samples": n_samples,
        "num_layers": n_layers,
        "hidden_size": hidden_size,

        "cv": {
            "n_splits": args.n_splits,
            "shuffle": True,
            "seed": args.seed,
        },

        "probe": {
            "type": (
                "logistic_regression"
            ),
            "penalty": "l2",
            "solver": "liblinear",
            "C": args.C,
            "max_iter": 5000,
        },

        "best_layer": best_layer,

        "best_layer_relative_depth": (
            float(
                best_row[
                    "relative_depth"
                ]
            )
        ),

        "best_layer_cv": {
            "mean_auroc": float(
                best_row[
                    "mean_auroc"
                ]
            ),
            "std_auroc": float(
                best_row[
                    "std_auroc"
                ]
            ),
            "mean_auprc": float(
                best_row[
                    "mean_auprc"
                ]
            ),
            "std_auprc": float(
                best_row[
                    "std_auprc"
                ]
            ),
        },

        "final_probe_training": (
            "all_XSTest_examples"
        ),

        "activation_type": (
            data.get(
                "activation_type",
                "unknown",
            )
        ),
    }

    metadata_path = (
        args.output_dir
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"CV results:    {fold_path}"
    )

    print(
        f"Layer summary: {summary_path}"
    )

    print(
        f"Scaler:        {scaler_path}"
    )

    print(
        f"Final probe:   {probe_path}"
    )

    print(
        f"Metadata:      {metadata_path}"
    )


if __name__ == "__main__":
    main()