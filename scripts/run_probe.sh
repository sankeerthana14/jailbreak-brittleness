#!/bin/bash

#SBATCH --job-name=probe_qwen4b_xstest
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node11
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mech_interp


# ============================================================
# Environment information
# ============================================================

echo "=== Environment ==="

python - <<'PY'
import sklearn
import torch

print("INFO: scikit-learn:", sklearn.__version__)
print("INFO: PyTorch:", torch.__version__)
print("INFO: CUDA available:", torch.cuda.is_available())
PY

echo "==================="


# ============================================================
# Paths
# ============================================================

REPO_ROOT="$HOME/jailbreak-brittleness"

ACTIVATIONS="$REPO_ROOT/outputs/qwen3-4b/xstest/activations.pt"
PROBE_OUTPUT="$REPO_ROOT/probes/qwen3-4b"


# ============================================================
# Setup
# ============================================================

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$PROBE_OUTPUT"

cd "$REPO_ROOT"

echo "Activations: $ACTIVATIONS"
echo "Probe output: $PROBE_OUTPUT"


# ============================================================
# Train probe
# ============================================================

python scripts/train_probe.py \
    --activations "$ACTIVATIONS" \
    --output-dir "$PROBE_OUTPUT" \
    --n-splits 5 \
    --seed 42 \
    --C 1.0


# ============================================================
# Completion
# ============================================================

EXIT_CODE=$?

echo "================="
echo "Probe training finished."
echo "Exit code: $EXIT_CODE"
echo "================="

exit $EXIT_CODE