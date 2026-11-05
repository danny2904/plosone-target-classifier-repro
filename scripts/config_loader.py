"""Paths used by eval_mre.py. Package-local; override at call time with --model_path."""
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(root_dir, "data")

DATA_PATH = os.path.join(data_dir, "vihos_prepared.jsonl")
MODEL_DIR = os.path.join(root_dir, "output", "mre")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")


def load_env():
    return None
