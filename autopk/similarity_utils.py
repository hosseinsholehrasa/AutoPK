"""
similarity_utils.py
Functions for embeddings, similarity scoring, and fuzzy matching.
"""

import numpy as np
import torch
import re
from Levenshtein import distance as levenshtein_distance
from unidecode import unidecode
import unicodedata

from transformers import AutoTokenizer, AutoModel


# TODO: Fix this part to not load the model and tokenizer for every usage
# Load BioBERT (or ClinicalBERT)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BIOBERT_TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
_BIOBERT_MODEL = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
_BIOBERT_MODEL.eval()


# ----- Embeddings -----
def term_embedding(term: str):
    """Return CLS embedding for a given term using BioBERT."""
    with torch.no_grad():
        inputs = _BIOBERT_TOKENIZER(term, return_tensors="pt", truncation=True,
                                   padding=True, max_length=32).to(DEVICE)
        outputs = _BIOBERT_MODEL(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding


# def normalize_vector(vec):
#     norm = np.linalg.norm(vec)
#     return vec if norm == 0 else vec / norm


# ----- Text preprocessing -----
def auto_preprocess(text):
    """Normalize text (remove accents, symbols, and lowercase)."""
    return text # TODO for now, disable preprocessing
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKD", text)
    text = unidecode(text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[\[\],@_]', ' ', text)
    text = text.replace("-", " ")
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# ----- Similarity metrics -----
def token_overlap_score(a: str, b: str):
    def tokenize(s):
        return set(re.split(r'[\s_\-()]+', s.lower().strip()))
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    return round((2 * len(tokens_a & tokens_b)) / (len(tokens_a) + len(tokens_b)), 4)

def edit_distance_score(a: str, b: str):
    return 1 - (levenshtein_distance(a, b) / max(len(a), len(b)))

def weighted_score(cosine, edit, token_overlap, w_cos=0.6, w_edit=0.2, w_token=0.2):
    """
    Calculate a weighted similarity score based on cosine similarity, edit distance, and token overlap.
    """
    
    # Calculate the weighted score
    score = (w_cos * cosine) + (w_edit * edit) + (w_token * token_overlap)

    return round(score, 4)
