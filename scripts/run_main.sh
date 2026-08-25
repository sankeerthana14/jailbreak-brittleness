#!/bin/bash

#SBATCH --job-name=extract_qwen4b_xstest
#SBATCH --gres=gpu:1
#SBATCH --partition=PA10080q
#SBATCH --nodelist=node04
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mech_interp


# ============================================================
# GPU Debug
# ============================================================

echo "=== GPU Debug ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )
PY

echo "================="


# ============================================================
# Paths
# ============================================================

REPO_ROOT="$HOME/jailbreak-brittleness"
MODEL_ROOT="$REPO_ROOT/models"
DATA_ROOT="$REPO_ROOT/data/processed"
OUTPUT_ROOT="$REPO_ROOT/outputs"


# ============================================================
# Setup
# ============================================================

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"

echo "Repository root: $REPO_ROOT"
echo "Model root:      $MODEL_ROOT"
echo "Data root:       $DATA_ROOT"
echo "Output root:     $OUTPUT_ROOT"

echo "Starting full Qwen3-4B / XSTest extraction..."


# ============================================================
# Experiment
# ============================================================

python scripts/extract_activations.py \
    --model qwen3-4b \
    --dataset xstest \
    --model-root "$MODEL_ROOT" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --max-new-tokens 256


# ============================================================
# Completion
# ============================================================

EXIT_CODE=$?

echo "================="
echo "Experiment finished."
echo "Exit code: $EXIT_CODE"
echo "================="

exit $EXIT_CODE