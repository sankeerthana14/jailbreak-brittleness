import os
from pathlib import Path

CWD = Path.cwd().resolve().parent

# MAIN DIRECTORIES
DATA_DIR = os.path.join(CWD, "data")
DATASETS_DIR = os.path.join(CWD, "datasets")
MODELS_DIR = os.path.join(CWD, "models")
NOTEBOOKS_DIR = os.path.join(CWD, "notebooks")
SCRIPTS_DIR = os.path.join(CWD, "scripts")

