#!/usr/bin/env python3

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "qwen3": {
        "small": "Qwen/Qwen3-4B",
        "medium": "Qwen/Qwen3-8B",
        "large": "Qwen/Qwen3-14B",
    },
    "ministral3": {
        "small": "mistralai/Ministral-3-3B-Instruct-2512",
        "medium": "mistralai/Ministral-3-8B-Instruct-2512",
        "large": "mistralai/Ministral-3-14B-Instruct-2512",
    },
    "olmo2": {
        "small": "allenai/OLMo-2-0425-1B-Instruct",
        "medium": "allenai/OLMo-2-1124-7B-Instruct",
        "large": "allenai/OLMo-2-1124-13B-Instruct",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all model checkpoints used in the jailbreak brittleness study."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory in which models will be stored.",
    )

    parser.add_argument(
        "--family",
        choices=["all", *MODELS.keys()],
        default="all",
        help="Download one model family or all families.",
    )

    parser.add_argument(
        "--scale",
        choices=["all", "small", "medium", "large"],
        default="all",
        help="Download one scale or all scales.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token. "
            "Prefer setting HF_TOKEN instead of passing it on the command line."
        ),
    )

    return parser.parse_args()


def selected_models(family, scale):
    for family_name, family_models in MODELS.items():

        if family != "all" and family_name != family:
            continue

        for scale_name, repo_id in family_models.items():

            if scale != "all" and scale_name != scale:
                continue

            yield family_name, scale_name, repo_id


def download_model(repo_id, destination, token=None):
    destination.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Downloading: {repo_id}")
    print(f"Destination: {destination}")
    print("=" * 80)

    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        token=token,
    )

    print(f"Finished downloading {repo_id}\n")


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = list(selected_models(args.family, args.scale))

    if not models:
        raise RuntimeError("No models matched the requested selection.")

    print(f"Downloading {len(models)} model(s).\n")

    for family, scale, repo_id in models:

        destination = (
            args.output_dir
            / family
            / scale
        )

        download_model(
            repo_id=repo_id,
            destination=destination,
            token=args.token,
        )

    print("All requested model downloads completed.")


if __name__ == "__main__":
    main()