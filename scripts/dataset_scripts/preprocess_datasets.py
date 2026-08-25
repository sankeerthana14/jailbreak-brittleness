#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


STANDARD_COLUMNS = [
    "sample_id",
    "dataset",
    "prompt",
    "prompt_harmful",
    "category",
    "source",
    "adversarial",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess XSTest, JailbreakBench, and WildGuardTest "
            "into a common prompt-level schema."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Directory containing the raw downloaded Parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed"),
        help="Directory in which processed datasets will be saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used only for deterministic JBB shuffling.",
    )

    return parser.parse_args()


def preprocess_xstest(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "xstest.parquet"
    df = pd.read_parquet(path)

    required_columns = {
        "id",
        "prompt",
        "type",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"XSTest is missing required columns: {sorted(missing)}"
        )

    label_map = {
        "safe": 0,
        "unsafe": 1,
    }

    unknown_labels = (
        set(df["label"].dropna().unique())
        - set(label_map)
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected XSTest labels: {unknown_labels}"
        )

    processed = pd.DataFrame({
        "sample_id": "xstest_" + df["id"].astype(str),
        "dataset": "xstest",
        "prompt": df["prompt"],
        "prompt_harmful": df["label"].map(label_map),
        "category": df["type"],
        "source": "xstest",
        "adversarial": pd.NA,
    })

    processed["prompt_harmful"] = (
        processed["prompt_harmful"].astype(int)
    )

    # Sanity checks
    assert len(processed) == 450
    assert processed["sample_id"].is_unique
    assert processed["prompt"].notna().all()
    assert processed["prompt_harmful"].notna().all()

    counts = processed["prompt_harmful"].value_counts()

    assert counts[0] == 250
    assert counts[1] == 200

    return processed[STANDARD_COLUMNS]


def preprocess_jbb(raw_dir: Path, seed: int) -> pd.DataFrame:
    harmful_path = raw_dir / "jbb_harmful.parquet"
    benign_path = raw_dir / "jbb_benign.parquet"

    harmful = pd.read_parquet(harmful_path)
    benign = pd.read_parquet(benign_path)

    required_columns = {
        "Index",
        "Goal",
        "Category",
        "Source",
    }

    for name, df in {
        "JBB harmful": harmful,
        "JBB benign": benign,
    }.items():

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing)}"
            )

    harmful_processed = pd.DataFrame({
        "sample_id": (
            "jbb_harmful_" + harmful["Index"].astype(str)
        ),
        "dataset": "jbb",
        "prompt": harmful["Goal"],
        "prompt_harmful": 1,
        "category": harmful["Category"],
        "source": harmful["Source"],
        "adversarial": pd.NA,
    })

    benign_processed = pd.DataFrame({
        "sample_id": (
            "jbb_benign_" + benign["Index"].astype(str)
        ),
        "dataset": "jbb",
        "prompt": benign["Goal"],
        "prompt_harmful": 0,
        "category": benign["Category"],
        "source": benign["Source"],
        "adversarial": pd.NA,
    })

    # Combine harmful and benign examples.
    processed = pd.concat(
        [harmful_processed, benign_processed],
        ignore_index=True,
    )

    # JBB is already perfectly balanced:
    # 100 harmful + 100 benign.
    #
    # Shuffle only so examples are not ordered by class.
    # random_state makes this deterministic/reproducible.
    processed = processed.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    # Sanity checks
    assert len(processed) == 200
    assert processed["sample_id"].is_unique
    assert processed["prompt"].notna().all()
    assert processed["prompt_harmful"].notna().all()

    counts = processed["prompt_harmful"].value_counts()

    assert counts[0] == 100
    assert counts[1] == 100

    return processed[STANDARD_COLUMNS]


def preprocess_wildguard(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "wildguard_test.parquet"
    df = pd.read_parquet(path)

    required_columns = {
        "prompt",
        "prompt_harm_label",
        "subcategory",
        "adversarial",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"WildGuardTest is missing required columns: "
            f"{sorted(missing)}"
        )

    # Preserve original raw row index for stable sample IDs.
    df = df.reset_index(names="raw_index")

    original_n = len(df)

    # Remove samples without ground-truth prompt-harm labels.
    df = df.dropna(
        subset=["prompt_harm_label"]
    ).copy()

    dropped_n = original_n - len(df)

    print(
        f"WildGuardTest: dropped {dropped_n} rows "
        "without prompt harm labels."
    )

    label_map = {
        "unharmful": 0,
        "harmful": 1,
    }

    unknown_labels = (
        set(df["prompt_harm_label"].unique())
        - set(label_map)
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected WildGuard labels: {unknown_labels}"
        )

    processed = pd.DataFrame({
        "sample_id": (
            "wildguard_" + df["raw_index"].astype(str)
        ),
        "dataset": "wildguard",
        "prompt": df["prompt"],
        "prompt_harmful": (
            df["prompt_harm_label"].map(label_map)
        ),
        "category": df["subcategory"],
        "source": "wildguardtest",
        "adversarial": df["adversarial"],
    })

    processed["prompt_harmful"] = (
        processed["prompt_harmful"].astype(int)
    )

    # Sanity checks
    assert original_n == 1725
    assert dropped_n == 26
    assert len(processed) == 1699

    assert processed["sample_id"].is_unique
    assert processed["prompt"].notna().all()
    assert processed["prompt_harmful"].notna().all()

    counts = processed["prompt_harmful"].value_counts()

    assert counts[0] == 945
    assert counts[1] == 754

    return processed[STANDARD_COLUMNS]


def print_summary(name: str, df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {len(df)}")

    print("\nLabel counts:")
    counts = (
        df["prompt_harmful"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():
        label_name = (
            "safe / benign / unharmful"
            if label == 0
            else "unsafe / harmful"
        )

        print(
            f"{label} ({label_name}): {count}"
        )

    print("\nColumns:")
    print(df.columns.tolist())


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    xstest = preprocess_xstest(
        args.raw_dir,
    )

    jbb = preprocess_jbb(
        args.raw_dir,
        args.seed,
    )

    wildguard = preprocess_wildguard(
        args.raw_dir,
    )

    # Save each processed benchmark separately.
    xstest.to_parquet(
        args.output_dir / "xstest.parquet",
        index=False,
    )

    jbb.to_parquet(
        args.output_dir / "jbb.parquet",
        index=False,
    )

    wildguard.to_parquet(
        args.output_dir / "wildguard.parquet",
        index=False,
    )

    print_summary(
        "XSTest",
        xstest,
    )

    print_summary(
        "JailbreakBench",
        jbb,
    )

    print_summary(
        "WildGuardTest",
        wildguard,
    )

    print("\nProcessed files saved to:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()