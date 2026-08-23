#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAMES = [
    "xstest",
    "jailbreakbench",
    "wildguard",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets used in the jailbreak brittleness study."
    )

    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_NAMES],
        default="all",
        help="Dataset to download.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory in which datasets will be stored.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token. Required for gated datasets "
            "such as WildGuardTest unless already authenticated."
        ),
    )

    return parser.parse_args()


def save_dataset(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    df = dataset.to_pandas()
    df.to_parquet(path, index=False)

    print(f"Saved {len(df):,} rows -> {path}")


def download_xstest(output_dir, token=None):
    print("\nDownloading XSTest...")

    dataset = load_dataset(
        "Paul/XSTest",
        split="train",
        token=token,
    )

    save_dataset(
        dataset,
        output_dir / "xstest.parquet",
    )


def download_jailbreakbench(output_dir, token=None):
    print("\nDownloading JailbreakBench...")

    dataset = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        "behaviors",
        token=token,
    )

    print(dataset)

    # The HF dataset contains separate splits for harmful and benign
    # behaviors. Save each independently so provenance is explicit.
    for split_name, split in dataset.items():

        save_dataset(
            split,
            output_dir / f"jbb_{split_name}.parquet",
        )


def download_wildguard(output_dir, token=None):
    print("\nDownloading WildGuardTest...")

    dataset = load_dataset(
        "allenai/wildguardmix",
        "wildguardtest",
        split="test",
        token=token,
    )

    save_dataset(
        dataset,
        output_dir / "wildguard_test.parquet",
    )


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested = (
        DATASET_NAMES
        if args.dataset == "all"
        else [args.dataset]
    )

    if "xstest" in requested:
        download_xstest(
            args.output_dir,
            args.token,
        )

    if "jailbreakbench" in requested:
        download_jailbreakbench(
            args.output_dir,
            args.token,
        )

    if "wildguard" in requested:
        download_wildguard(
            args.output_dir,
            args.token,
        )

    print("\nAll requested dataset downloads completed.")


if __name__ == "__main__":
    main()