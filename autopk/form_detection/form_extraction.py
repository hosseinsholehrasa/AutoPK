"""
form_extraction.py
Pipeline 1: Detecting and validating PK parameter forms/variants.
"""
import numpy as np
import torch

from autopk.llm_utils import llm_chat_with_history, extract_value_dollar
from autopk.similarity_utils import (
    auto_preprocess,
    term_embedding,
    weighted_score,
    token_overlap_score,
    edit_distance_score,
)
from autopk.form_detection.pk_form_prompts import build_pk_prompt_examples, pk_extraction_config


# ===== Variant extraction =====
def extract_pk_parameter_llm(table: str, pk_name: str, model_name: str='gemma3') -> str:
    """Extract pharmacokinetic (PK) parameters from a table using a language model.
    Args:
        table (str): The table in CSV format as a string.
        pk_name (str): The name of the PK parameter to extract.
        model_name (str): The name of the language model to use.
    Returns:
        str: The extracted PK parameters in the specified format.
    """
    # 1) Build prompt and examples messages
    pk_prompt, examples = build_pk_prompt_examples(pk_name, pk_extraction_config)


    chat_history = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer the user's questions based on the provided table in CSV format concisely."
        },

    ] + examples + [
                {
            "role": "user",
            "content": f"This is the table:\n{table}\n question: {pk_prompt}"
        }
    ]

    # Generate the assistant's response
    reply, updated_history = llm_chat_with_history(chat_history, model_name=model_name, max_tokens=512)

    return reply


# ===== Check if a term is a variant using LLM =====
def is_term_variant_of_parameter(
    term,
    reference_set,
    cache,
    concept_name="half-life",
    model_name="llama3",
    max_tokens=32,
):
    """
    Check whether a given term is a variant/alias of a specific PK parameter (e.g., "half-life").
    Uses LLM to avoid repeatedly querying for known comparisons (cached).
    
    Args:
        term (str): The term to evaluate.
        reference_set (set): Set of known aliases/variants for the concept.
        cache (dict): A dictionary cache for storing prior results to avoid redundant LLM calls.
        concept_name (str): The main parameter being evaluated (e.g., "half-life", "AUC", etc.)
        client: The LLM API client.
        model_name (str): The name of the LLM model to use.
        max_tokens (int): Max tokens for LLM output.

    Returns:
        bool or str: True/False if decision is binary, or full response string if unclear.
    """

    # If already cached
    if term in cache:
        return cache[term]

    sorted_refs = sorted(reference_set)
    formatted_refs = ",".join(
        [f'${ref.replace("^", "").strip()}$' for ref in sorted_refs]
    )


    chat_history = [
        {
            "role": "system",
            "content": "You are a pharmacokinetics terminology expert."
        },
        {
            "role": "user",
            "content": f"""Your task is to determine whether the following term is a variant or notation of the pharmacokinetic concept "{concept_name}".
Known forms include:
{formatted_refs}

These terms often appear in different formats, sometimes with symbols (e.g., Greek letters, λz), units (e.g., hours, days), or slight variations. Do NOT accept unrelated pharmacokinetic concepts like '12+31' (only number) or 'hr' (only unit) or 'Ka' (Absorption rate constant which is not related to {concept_name}). Only answer "yes" if it is clearly a variant of "{concept_name}".

Term: {term}
Is this a variant of "{concept_name}"? Answer with "yes" or "no"."""
        }
    ]


    reply, _ = llm_chat_with_history(
        history=chat_history, model_name=model_name, max_tokens=max_tokens
    )
    reply_clean = reply.strip().lower()
    cache[term] = reply_clean

    if "yes" in reply_clean:
        return True
    elif "no" in reply_clean:
        return False
    else:
        return reply  # fallback


# ===== Find PK parameter locations =====
def find_pk_parameter_locations(
    input_table,
    initial_parameter_set,
    candidate_parameters,
    similarity_cache,
    scores_caches,
    embedding_cache,
    weights=(0.6, 0.2, 0.2),
    concept_name="half-life",
    threshold=0.69,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Find locations in a table that match or are similar to known PK parameter terms (e.g., half-life forms).

    Args:
        input_table (pd.DataFrame): The table to search.
        initial_parameter_set (set): Original list of valid terms for the target PK parameter.
        candidate_parameters (list): Current known variants of the PK parameter.
        similarity_cache (dict): Cache LLM-based is_variant responses to avoid repeated calls.
        scores_caches (dict): Cache for combined similarity scores.
        embedding_cache (dict): Cache for embeddings.
        threshold (float): Similarity score threshold for matching.
        device: Torch device.

    Returns:
        pk_locations (list of tuple): Matching cell locations.
        candidate_parameters (list): Updated list of candidate terms.
        similarity_cache (dict): Updated similarity is_variant decisions.
        scores_caches (dict): Updated scores cache.
        embedding_cache (dict): Updated embedding cache.
    """

    def get_cached_embedding(term):
        norm_term = auto_preprocess(term)
        if norm_term in embedding_cache:
            return embedding_cache[norm_term]
        vec = term_embedding(norm_term).astype("float32")
        embedding_cache[norm_term] = vec
        return vec

    def normalize_tensor(tensor, dim):
        return torch.nn.functional.normalize(tensor, dim=dim)

    def is_valid_candidate(term, scores):
        if max(scores) < threshold:
            return False

        # best_candidate = candidate_parameters[np.argmax(scores)]

        decision = similarity_cache.get(term, None)
        if decision and decision.lower() == "no":
            return False
        if decision is None:
            return is_term_variant_of_parameter(
                term, initial_parameter_set, similarity_cache, concept_name=concept_name
            )
        return True  # "yes"

    def evaluate_and_register_term(term, loc):
        """
        Check a single preprocessed term at location loc:
          - If already known → record and return True
          - Otherwise → compute similarity, maybe accept & expand candidates
        """
        nonlocal candidate_tensor

        if term in known_terms:
            pk_locations.append(loc)
            return True

        if len(term) < 2:
            return False

        vec = get_cached_embedding(term)
        if np.linalg.norm(vec) == 0:
            return False

        query_vec = normalize_tensor(torch.tensor(vec, device=device), dim=0)
        cosine = torch.matmul(candidate_tensor, query_vec).cpu().numpy()

        scores = []
        for i, cand in enumerate(candidate_parameters):
            pair_key = tuple(sorted([term, cand]))
            if pair_key in scores_caches:
                cos_sim, edit_sim, token_sim = scores_caches[pair_key]
            else:
                edit_sim = edit_distance_score(term, cand)
                token_sim = token_overlap_score(term, cand)
                cos_sim = cosine[i]
                scores_caches[pair_key] = (cos_sim, edit_sim, token_sim)

            scores.append(
                weighted_score(cos_sim, edit_sim, token_sim,
                               w_cos=weights[0], w_edit=weights[1], w_token=weights[2])
            )

        if is_valid_candidate(term, scores):
            pk_locations.append(loc)
            candidate_parameters.append(term)
            known_terms.add(term)

            new_vec = get_cached_embedding(term)
            new_t = normalize_tensor(torch.tensor(new_vec, device=device).unsqueeze(0), dim=1)
            candidate_tensor = torch.cat([candidate_tensor, new_t], dim=0)
            return True

        return False

    # --- Prepare embeddings for initial candidates ---
    candidate_vecs = [get_cached_embedding(candidate) for candidate in candidate_parameters]
    candidate_tensor = normalize_tensor(torch.tensor(np.stack(candidate_vecs), device=device), dim=1)

    pk_locations = []
    known_terms = set(candidate_parameters)

    # 🔍 scan headers
    for col_idx, header in enumerate(input_table.columns):
        loc = (-1, col_idx - 1)
        for raw in header.split("^"):
            term = auto_preprocess(raw)
            if evaluate_and_register_term(term, loc):
                break

    if pk_locations:
        return pk_locations, candidate_parameters, similarity_cache, scores_caches, embedding_cache

    # 🔍 scan data cells
    for row_idx, row in input_table.iterrows():
        for col_idx, cell in row.items():
            if not isinstance(cell, str):
                continue
            term = auto_preprocess(cell)
            if evaluate_and_register_term(term, (row_idx, col_idx)):
                break

    return pk_locations, candidate_parameters, similarity_cache, scores_caches, embedding_cache


# ===== Extract rows based on parameter locations =====
def extract_rows(df, parameters_locations):
    rows = []
    if parameters_locations is None or len(parameters_locations) == 0:
        print("No parameters_locations provided, using default first row as parameters_locations.")
        print("No parameters found. This is the df:\n", df.to_csv(index=False,header=True))
        # set the first row as the parameters_locations
        parameters_locations = []
        for i in range(0, len(df)):
            parameters_locations.append((i, 0))
    
    for loc in parameters_locations:
        target_row = df.iloc[loc[0]]
        # combine each cell of the target row with the second row separated by @
        res = ""
        for col in target_row.index:
            res += f"<{target_row[col]}@{col}>"
        rows.append(res)
    return set(rows)