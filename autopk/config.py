"""
config.py
Global configuration for AutoPK.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Optional: restrict CUDA
import torch

# Target parameters we care about
TARGET_PARAMETERS = ['half-life', 'AUC', 'CL', "MRT", "CMAX", "TMAX"]

# Model and pipeline settings # TODO: make configurable via CLI args
PIPELINE2_MODEL_NAME = "llama3"
THRESHOLD = 0.69
WEIGHTS = (0.6, 0.2, 0.2)  # (cosine, edit distance, token overlap)

# Logging and saving
SAVE_LOG_FILE_NAME = f"../experiments/autopk_{PIPELINE2_MODEL_NAME}.txt"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache containers
EMBEDDING_CACHE = {}
SCORES_CACHE = {}

# Initialize scores log
SCORES_LOG_ALL = {
    str(THRESHOLD): {parm: {} for parm in TARGET_PARAMETERS}
}

# ==============================
# API Keys & Base URLs
# ==============================
# It could be Direct OpenAI or a proxy like LiteLLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "IAMAPIKEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

HF_TOKEN = os.getenv("HF_TOKEN", "IAMHFTOKEN")