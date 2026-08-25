#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)


# ============================================================
# Model configuration
# ============================================================

MODEL_CONFIG = {
    # --------------------------------------------------------
    # Qwen 3
    # --------------------------------------------------------
    "qwen3-4b": {
        "family": "qwen3",
        "scale": "small",
        "path": "qwen3/small",
        "loader": "causal_lm",
    },
    "qwen3-8b": {
        "family": "qwen3",
        "scale": "medium",
        "path": "qwen3/medium",
        "loader": "causal_lm",
    },
    "qwen3-14b": {
        "family": "qwen3",
        "scale": "large",
        "path": "qwen3/large",
        "loader": "causal_lm",
    },

    # --------------------------------------------------------
    # Ministral 3
    # --------------------------------------------------------
    "ministral3-3b": {
        "family": "ministral3",
        "scale": "small",
        "path": "ministral3/small",
        "loader": "mistral3",
    },
    "ministral3-8b": {
        "family": "ministral3",
        "scale": "medium",
        "path": "ministral3/medium",
        "loader": "mistral3",
    },
    "ministral3-14b": {
        "family": "ministral3",
        "scale": "large",
        "path": "ministral3/large",
        "loader": "mistral3",
    },

    # --------------------------------------------------------
    # OLMo 2
    # --------------------------------------------------------
    "olmo2-1b": {
        "family": "olmo2",
        "scale": "small",
        "path": "olmo2/small",
        "loader": "causal_lm",
    },
    "olmo2-7b": {
        "family": "olmo2",
        "scale": "medium",
        "path": "olmo2/medium",
        "loader": "causal_lm",
    },
    "olmo2-13b": {
        "family": "olmo2",
        "scale": "large",
        "path": "olmo2/large",
        "loader": "causal_lm",
    },
}


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the last-prompt-token hidden state from every "
            "transformer layer and generate one response per prompt."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CONFIG.keys(),
        help="Model checkpoint to evaluate.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["xstest", "jbb", "wildguard"],
        help="Processed dataset to evaluate.",
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Root directory containing downloaded model folders.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/processed"),
        help="Directory containing processed Parquet datasets.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Directory in which activations/responses are saved.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of response tokens to generate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples for debugging.",
    )

    return parser.parse_args()


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(model_path: Path, cfg: dict):
    """
    Load the appropriate model/tokenizer implementation.

    Qwen3 / OLMo2:
        AutoTokenizer + AutoModelForCausalLM

    Ministral3:
        MistralCommonBackend +
        Mistral3ForConditionalGeneration
    """

    if cfg["loader"] == "mistral3":

        tokenizer = MistralCommonBackend.from_pretrained(
            model_path
        )

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
        )

        return model, tokenizer

    if cfg["loader"] == "causal_lm":

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
        )

        return model, tokenizer

    raise ValueError(
        f"Unknown loader type: {cfg['loader']}"
    )


# ============================================================
# Chat formatting
# ============================================================

def build_model_inputs(tokenizer, prompt: str, family: str):
    """
    Apply the official chat template for each model family.
    """

    # --------------------------------------------------------
    # Ministral 3
    # --------------------------------------------------------

    if family == "ministral3":

        # Ministral3 uses structured multimodal-style messages,
        # but our experiment supplies text only.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )

        return encoded

    # --------------------------------------------------------
    # Qwen3 / OLMo2
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }

    # Qwen3 thinking is enabled by default.
    # Disable it so Qwen is evaluated in non-thinking mode.
    if family == "qwen3":
        kwargs["enable_thinking"] = False

    encoded = tokenizer.apply_chat_template(
        messages,
        **kwargs,
    )

    return encoded


# ============================================================
# Helper: determine input device
# ============================================================

def get_input_device(model):
    """
    Get the device hosting the language model's input embeddings.

    More reliable than next(model.parameters()).device when using
    device_map='auto'.
    """

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


# ============================================================
# Helper: generation token IDs
# ============================================================

def get_generation_token_id(model, name):
    """
    Get pad/eos token IDs from the model's generation config,
    falling back to the model config.
    """

    value = getattr(
        model.generation_config,
        name,
        None,
    )

    if value is None:
        value = getattr(
            model.config,
            name,
            None,
        )

    return value


# ============================================================
# Helper: hidden states
# ============================================================

def get_transformer_hidden_states(outputs):
    """
    Hugging Face hidden_states convention:

        hidden_states[0] = embedding output
        hidden_states[1] = transformer layer 1 output
        ...
        hidden_states[L] = transformer layer L output

    We intentionally exclude the embedding output.
    """

    hidden_states = outputs.hidden_states

    if hidden_states is None:
        raise RuntimeError(
            "Model returned hidden_states=None."
        )

    if len(hidden_states) < 2:
        raise RuntimeError(
            "Unexpected hidden-state structure."
        )

    return hidden_states[1:]


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    cfg = MODEL_CONFIG[args.model]

    model_path = (
        args.model_root / cfg["path"]
    ).resolve()

    dataset_path = (
        args.data_root / f"{args.dataset}.parquet"
    ).resolve()

    output_dir = (
        args.output_root
        / args.model
        / args.dataset
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    print(f"Model:        {args.model}")
    print(f"Family:       {cfg['family']}")
    print(f"Scale:        {cfg['scale']}")
    print(f"Loader:       {cfg['loader']}")
    print(f"Model path:   {model_path}")
    print(f"Dataset:      {args.dataset}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path:  {output_dir}")

    # ========================================================
    # Validate paths
    # ========================================================

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_path}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_path}"
        )

    # ========================================================
    # Load dataset
    # ========================================================

    df = pd.read_parquet(
        dataset_path
    )

    if args.limit is not None:
        df = df.head(
            args.limit
        ).copy()

    required_columns = {
        "sample_id",
        "dataset",
        "prompt",
        "prompt_harmful",
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Dataset missing required columns: "
            f"{sorted(missing)}"
        )

    print(f"Number of prompts: {len(df)}")

    # ========================================================
    # Load model + tokenizer
    # ========================================================

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        cfg=cfg,
    )

    model.eval()

    input_device = get_input_device(
        model
    )

    print(f"Input device: {input_device}")

    pad_token_id = get_generation_token_id(
        model,
        "pad_token_id",
    )

    eos_token_id = get_generation_token_id(
        model,
        "eos_token_id",
    )

    print(f"pad_token_id: {pad_token_id}")
    print(f"eos_token_id: {eos_token_id}")

    # ========================================================
    # Output containers
    # ========================================================

    all_sample_ids = []
    all_labels = []
    all_activations = []

    response_records = []

    # ========================================================
    # Process prompts
    # ========================================================

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{args.model} / {args.dataset}",
    ):

        sample_id = str(
            row["sample_id"]
        )

        prompt = str(
            row["prompt"]
        )

        label = int(
            row["prompt_harmful"]
        )

        # ----------------------------------------------------
        # Apply model-specific chat template
        # ----------------------------------------------------

        encoded = build_model_inputs(
            tokenizer=tokenizer,
            prompt=prompt,
            family=cfg["family"],
        )

        # Preserve non-tensor fields if a model returns any.
        encoded = {
            key: (
                value.to(input_device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in encoded.items()
        }

        if "input_ids" not in encoded:
            raise RuntimeError(
                "Tokenizer output does not contain input_ids."
            )

        prompt_length = (
            encoded["input_ids"].shape[-1]
        )

        # ====================================================
        # 1. PRE-GENERATION ACTIVATION EXTRACTION
        # ====================================================

        with torch.inference_mode():

            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        layer_hidden_states = (
            get_transformer_hidden_states(
                outputs
            )
        )

        # ----------------------------------------------------
        # Extract final prompt token from every layer
        #
        # Each hidden tensor:
        #
        # [batch_size, sequence_length, hidden_size]
        #
        # We retain:
        #
        # hidden[0, prompt_length - 1, :]
        # ----------------------------------------------------

        last_token_per_layer = torch.stack(
            [
                hidden[
                    0,
                    prompt_length - 1,
                    :
                ]
                .detach()
                .to(
                    dtype=torch.float16
                )
                .cpu()
                for hidden in layer_hidden_states
            ],
            dim=0,
        )

        # Shape:
        #
        # [num_layers, hidden_size]

        all_activations.append(
            last_token_per_layer
        )

        all_sample_ids.append(
            sample_id
        )

        all_labels.append(
            label
        )

        # Free large forward-pass tensors before generation.
        del outputs
        del layer_hidden_states

        # ====================================================
        # 2. RESPONSE GENERATION
        # ====================================================

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }

        if pad_token_id is not None:
            generation_kwargs[
                "pad_token_id"
            ] = pad_token_id

        if eos_token_id is not None:
            generation_kwargs[
                "eos_token_id"
            ] = eos_token_id

        with torch.inference_mode():

            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )

        # Keep only newly generated response tokens.
        generated_tokens = generated[
            0,
            prompt_length:
        ]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        # ====================================================
        # Save response metadata
        # ====================================================

        response_records.append(
            {
                "sample_id": sample_id,
                "dataset": row["dataset"],
                "model": args.model,
                "family": cfg["family"],
                "scale": cfg["scale"],
                "prompt": prompt,
                "prompt_harmful": label,
                "response": response,
            }
        )

        # ----------------------------------------------------
        # Cleanup sample tensors
        # ----------------------------------------------------

        del generated
        del generated_tokens
        del encoded

    # ========================================================
    # Stack activation tensors
    # ========================================================

    activations = torch.stack(
        all_activations,
        dim=0,
    )

    # Shape:
    #
    # [num_samples, num_layers, hidden_size]

    # ========================================================
    # Save activation file
    # ========================================================

    activation_output = {
        "sample_ids": (
            all_sample_ids
        ),

        "labels": torch.tensor(
            all_labels,
            dtype=torch.long,
        ),

        "activations": (
            activations
        ),

        "model": args.model,
        "family": cfg["family"],
        "scale": cfg["scale"],
        "dataset": args.dataset,

        "num_samples": (
            len(all_sample_ids)
        ),

        "num_layers": (
            activations.shape[1]
        ),

        "hidden_size": (
            activations.shape[2]
        ),

        "activation_type": (
            "last_prompt_token"
        ),

        "embedding_output_included": False,
    }

    activation_path = (
        output_dir
        / "activations.pt"
    )

    torch.save(
        activation_output,
        activation_path,
    )

    # ========================================================
    # Save generated responses
    # ========================================================

    responses_df = pd.DataFrame(
        response_records
    )

    response_path = (
        output_dir
        / "responses.parquet"
    )

    responses_df.to_parquet(
        response_path,
        index=False,
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "model": args.model,
        "family": cfg["family"],
        "scale": cfg["scale"],
        "loader": cfg["loader"],
        "dataset": args.dataset,

        "num_samples": len(df),

        "num_layers": int(
            activations.shape[1]
        ),

        "hidden_size": int(
            activations.shape[2]
        ),

        "activation_type": (
            "last_prompt_token"
        ),

        "embedding_output_included": False,

        "chat_template": {
            "qwen_thinking_enabled": (
                False
                if cfg["family"] == "qwen3"
                else None
            ),
        },

        "generation": {
            "max_new_tokens": (
                args.max_new_tokens
            ),
            "do_sample": False,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        },
    }

    metadata_path = (
        output_dir
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
        f"Activations: {activation_path}"
    )

    print(
        f"Responses:   {response_path}"
    )

    print(
        f"Metadata:    {metadata_path}"
    )

    print(
        "Activation tensor shape:",
        tuple(
            activations.shape
        ),
    )

    print(
        f"Responses saved: {len(responses_df)}"
    )

    # ========================================================
    # Cleanup
    # ========================================================

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()