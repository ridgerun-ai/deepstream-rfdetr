#!/usr/bin/env python3
# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "inference",
# ]
# ///

import sys
import shutil

from inference import get_model
from inference.models.aliases import RFDETR_ALIASES
from pathlib import Path


def usage():
    print("Download RF-DETR ONNX models", file=sys.stderr)
    print("Usage: uv run ./download_weights.py <MODEL_ID>\nMODEL_ID:", file=sys.stderr)
    [print(f"- {key}", file=sys.stderr) for key in RFDETR_ALIASES.keys()]

if len(sys.argv) != 2:
    usage()
    sys.exit(1)

model_id = sys.argv[1]

if model_id not in RFDETR_ALIASES.keys():
    print(f'"{model_id}" is not a valid model', file=sys.stderr)
    usage()
    sys.exit(1)

print(f"Downloading {model_id}...")
model = get_model(model_id)

src = Path(model.cache_dir) / model.weights_file
dst = Path.cwd() / f"{model_id}.onnx"

if dst.exists():
    print(f"{dst} already exists. Can't overwrite", file=sys.stderr)
    sys.exit(1)

shutil.copy2(src, dst)
print(f"Successfully downloaded {dst}")
