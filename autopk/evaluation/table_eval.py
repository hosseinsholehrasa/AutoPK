from Levenshtein import ratio as levenshtein_ratio
from scipy.optimize import linear_sum_assignment

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model ONCE when the module is imported
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def term_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768, dtype=np.float32)

    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**tokens)
        # Average-pool the token embeddings to get sentence-level vector
        embedding = output.last_hidden_state.mean(dim=1).squeeze()

    return embedding.cpu().numpy()


def clean_text_headers(text):
    return str(text).strip().lower().replace(" ", "_")


def match_column_name(ref_col, gen_col):
    """
    Compute the Levenshtein similarity between cleaned column names.
    Returns a score between 0 and 1.
    """
    return levenshtein_ratio(clean_text_headers(ref_col), clean_text_headers(gen_col))


def reorder_generated_table_by_header(ref_headers, gen_df, threshold=0.75):
    matched_gen_cols = set()
    reordered_data = []
    column_mapping = {}

    for ref in ref_headers:
        best_score = -1
        best_match = None

        for gen_col in gen_df.columns:
            if gen_col in matched_gen_cols:
                continue

            score = match_column_name(ref, gen_col)

            if score > best_score:
                best_score = score
                best_match = gen_col

        if best_score >= threshold:
            reordered_data.append(gen_df[best_match])
            column_mapping[ref] = best_match
            matched_gen_cols.add(best_match)
        else:
            reordered_data.append(pd.Series([np.nan] * len(gen_df)))
            column_mapping[ref] = None

    reordered_df = pd.concat(reordered_data, axis=1)
    reordered_df.columns = ref_headers
    extra_columns = [col for col in gen_df.columns if col not in matched_gen_cols]

    return reordered_df, column_mapping, extra_columns


def match_rows_by_overlap(ref_df, gen_df, threshold=0.5):
    """Aligns each row in ref_df with the best matching row in gen_df using token overlap."""
    matched_indices = {}
    used_gen_indices = set()

    for i, ref_row in ref_df.iterrows():
        best_match_idx = None
        best_score = -1

        for j, gen_row in gen_df.iterrows():
            if j in used_gen_indices:
                continue

            score = 0
            count = 0
            for col in ref_df.columns:
                ref_val = str(ref_row[col]).strip().lower()
                gen_val = str(gen_row[col]).strip().lower()

                if ref_val == 'nan' or gen_val == 'nan':
                    continue

                if ref_val == gen_val:
                    score += 1
                else:
                    ref_tokens = set(ref_val.split())
                    gen_tokens = set(gen_val.split())
                    token_overlap = len(ref_tokens & gen_tokens) / len(ref_tokens | gen_tokens) if (ref_tokens | gen_tokens) else 0
                    score += token_overlap
                count += 1

            avg_score = score / count if count else 0
            if avg_score > best_score:
                best_score = avg_score
                best_match_idx = j

        if best_score >= threshold:
            matched_indices[i] = best_match_idx
            used_gen_indices.add(best_match_idx)
        else:
            matched_indices[i] = None

    # 🆕 Get unmatched (extra) rows in gen_df
    extra_rows = [j for j in gen_df.index if j not in used_gen_indices]

    return matched_indices, extra_rows

# def match_rows_by_overlap(ref_df, gen_df, threshold=0.5):
#     """Align rows using weighted Levenshtein similarity for each column."""
#     matched_indices = {}
#     used_gen_indices = set()

#     # Define column weights (higher = more important)
#     column_weights = {
#         "pk_parameter_value": 3.0,
#         "drug_dosage": 2.0,
#         "route_of_administration": 2.0,
#         "animal": 1.0,
#         "drug": 1.0,
#         "pk_parameter": 1.0,
#         "pk_parameter_unit": 1.0,
#         "animal_matrix/commodity": 0.5
#     }

#     for i, ref_row in ref_df.iterrows():
#         best_match_idx = None
#         best_score = -1

#         for j, gen_row in gen_df.iterrows():
#             if j in used_gen_indices:
#                 continue

#             score = 0.0
#             total_weight = 0.0
#             for col in ref_df.columns:
#                 weight = column_weights.get(col, 1.0)
#                 ref_val = str(ref_row[col]).strip().lower()
#                 gen_val = str(gen_row[col]).strip().lower()

#                 if ref_val in ['nan', '', 'none'] and gen_val in ['nan', '', 'none']:
#                     sim = 1.0
#                 elif ref_val in ['nan', '', 'none'] or gen_val in ['nan', '', 'none']:
#                     sim = 0.0
#                 else:
#                     sim = levenshtein_ratio(ref_val, gen_val)

#                 score += sim * weight
#                 total_weight += weight

#             avg_score = score / total_weight if total_weight else 0.0
#             if avg_score > best_score:
#                 best_score = avg_score
#                 best_match_idx = j

#         if best_score >= threshold:
#             matched_indices[i] = best_match_idx
#             used_gen_indices.add(best_match_idx)
#         else:
#             matched_indices[i] = None

#     extra_rows = [j for j in gen_df.index if j not in used_gen_indices]
#     return matched_indices, extra_rows



def compute_similarity_matrix(ref_df, gen_df):
    n_ref = len(ref_df)
    n_gen = len(gen_df)
    sim_matrix = np.zeros((n_ref, n_gen))

    for i, ref_row in ref_df.iterrows():
        for j, gen_row in gen_df.iterrows():
            score = 0.0
            total_weight = 0.0
            for col in ref_df.columns:
                ref_val = str(ref_row[col]).strip().lower()
                gen_val = str(gen_row[col]).strip().lower()
                weight = 1.0
                if col == "pk_parameter_value":
                    weight = 3.0
                elif col in ["drug_dosage", "route_of_administration"]:
                    weight = 2.0

                if ref_val in ['nan', '', 'none'] and gen_val in ['nan', '', 'none']:
                    sim = 1.0
                elif ref_val in ['nan', '', 'none'] or gen_val in ['nan', '', 'none']:
                    sim = 0.0
                else:
                    sim = levenshtein_ratio(ref_val, gen_val)

                score += sim * weight
                total_weight += weight

            sim_matrix[i, j] = score / total_weight if total_weight else 0.0

    return sim_matrix


def optimal_row_assignment(ref_df, gen_df, threshold=0.5):
    sim_matrix = compute_similarity_matrix(ref_df, gen_df)
    cost_matrix = 1.0 - sim_matrix  # convert similarity to cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_indices = {}
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i, j] >= threshold:
            matched_indices[i] = j
        else:
            matched_indices[i] = None

    extra_rows = [j for j in gen_df.index if j not in col_ind]
    return matched_indices, extra_rows


def align_generated_table_rows(ref_df, gen_df_reordered, row_map):
    aligned_rows = []
    for i in ref_df.index:
        j = row_map.get(i)
        if j is not None:
            aligned_rows.append(gen_df_reordered.loc[j])
        else:
            aligned_rows.append(pd.Series([np.nan] * len(ref_df.columns), index=ref_df.columns))
    return pd.DataFrame(aligned_rows, columns=ref_df.columns)


def clean_cell_value(value):
    """Cleans a cell value by stripping whitespace and converting to lowercase."""
    value = str(value).strip().lower()
    # replace nan, None, or empty strings with None
    if value in ['nan', '', 'none']:
        return "None"
    value = value.replace("+-", "±")  # standardize the symbol
    # value = value.replace(" ", "_")  # remove spaces
    value = value.replace("-", "_")  # standardize term
    # print("Cleaned value:", value)
    return value

def compare_cells(ref_df, gen_df_aligned, extra_columns=list(), extra_rows=list(), threshold=0.8):
    tp = fp = fn = hc = 0
    # Columns that use Levenshtein only because of the numerical values or the cosine will not work
    use_levenshtein_cols = {"pk_parameter_value", "drug_dosage", "route_of_administration", 'pk_parameter', 'animal_matrix/commodity'}

    # 1. Compare aligned cells
    for i in range(len(ref_df)):
        for col in ref_df.columns:
            ref_val = ref_df.at[i, col]
            gen_val = gen_df_aligned.at[i, col]

            if pd.isna(ref_val):
                continue  # Skip unlabelled cell
            elif pd.isna(gen_val):
                fn += 1
                continue

            ref_str = clean_cell_value(ref_val)
            gen_str = clean_cell_value(gen_val)

            if col in use_levenshtein_cols:
                sim = levenshtein_ratio(ref_str, gen_str)
            else:
                # Embedding-based similarity
                vec_ref = term_embedding(ref_str).astype('float32')
                vec_gen = term_embedding(gen_str).astype('float32')

                # Skip if embeddings are empty
                if np.linalg.norm(vec_ref) == 0 or np.linalg.norm(vec_gen) == 0:
                    sim = 0.0
                else:
                    t_ref = torch.tensor(vec_ref, device=device)
                    t_gen = torch.tensor(vec_gen, device=device)
                    t_ref = torch.nn.functional.normalize(t_ref, dim=0)
                    t_gen = torch.nn.functional.normalize(t_gen, dim=0)

                    sim = float(torch.matmul(t_ref, t_gen).detach().cpu())
                    # print(f"Comparing {ref_str} with {gen_str} using cosine similarity: {sim:.4f}")
            if sim >= threshold:
                tp += 1
            else:
                # print(f"False Positive: {ref_str} vs {gen_str} (Similarity: {sim:.4f})")
                fp += 1

                    
    # 2. Add hallucinated false positives
    num_rows = len(ref_df)
    num_cols = len(ref_df.columns)

    if extra_columns:
        fp += len(extra_columns) * num_rows  # FP per extra column across all rows
        hc += len(extra_columns) * num_rows  # Hallucinated cells

    if extra_rows:
        fp += len(extra_rows) * num_cols  # FP per extra row across all columns
        hc += len(extra_rows) * num_cols  # Hallucinated cells

    return tp, fp, fn, hc



if __name__ == "__main__":

    # Example reference table
    # ref_df = pd.DataFrame([
    #     ["Terminal half-life", "hours", "121±28.3", None, None, "10 mg/kg", "Intraportal", None],
    #     ["Terminal half-life", "min", "184±25.3", None, None, "10 mg/kg", "Intragastric", None],

    # ], columns=[
    #     "pk_parameter", "pk_parameter_unit", "pk_parameter_value",
    #     "animal", "drug", "drug_dosage", "route_of_administration", "animal_matrix/commodity"
    # ])

    # # Generated table with missing/extra columns and possible row disorder
    # gen_df = pd.DataFrame([
    #     ["184±25.3", "min", "Terminal half-life", None, None, "10 mg/kg", "Intragastric", "extra2"],
    #     ["11±21.3", "min", "Terminal half-life", None, None, "10 mg/kg", "Intraportal", "extra3"],
    #     ["121 +- 28.3", "min", "Terminal halflife", None, None, "10 mg/kg", "Intraportal", "extra1"],

    # ], columns=[
    #     "pk_parameter_valux", "pk_parameter_unit", "pk_parameter",
    #     "drug", "animal", "drug_dosage", "route_of_administration", "unexpected_column"
    # ])




    # Unified Generated DataFrame
    gen_df = pd.DataFrame([
        ["t1/2", "h", "9.87±0.41", "rat", "9", "5mg/kg", "iv", "none"],
        ["t1/2", "h", "11.2±1.95", "rat", "9", "10mg/kg", "po", "none"],
        ["t1/2", "h", "5.50±0.16", "rat", "11l", "5mg/kg", "iv", "none"],
        ["t1/2", "h", "6.77±0.87", "rat", "11l", "10mg/kg", "po", "none"],
    ], columns=[
        "pk_parameter", "pk_parameter_unit", "pk_parameter_value",
        "animal", "drug", "drug_dosage", "route_of_administration", "animal_matrix/commodity"
    ])

    # Unified Reference DataFrame
    ref_df = pd.DataFrame([
        ["t1/2", "h", "9.87±0.41", "rat", "9", "10mg/kg", "po", "none"],
        ["t1/2", "h", "11.2±1.95", "rat", "9", "10mg/kg", "po", "none"],
        ["t1/2", "h", "5.50±0.16", "rat", "11l", "5mg/kg", "iv", "none"],
        ["t1/2", "h", "6.77±0.87", "rat", "11l", "10mg/kg", "po", "none"],

    ], columns=[
        "pk_parameter", "pk_parameter_unit", "pk_parameter_value",
        "animal", "drug", "drug_dosage", "route_of_administration", "animal_matrix/commodity"
    ])
    print(ref_df.T)
    x = ref_df.T
    # use the first row as the header
    x.columns = x.iloc[0]
    x = x[1:]
    print()
    print()
    print(x)

    # Column alignment
    reordered_df, mapping, extra_cols = reorder_generated_table_by_header(ref_df.columns, gen_df)
    print("🧭 Column Mapping:", mapping)
    print("🧨 Extra Columns in Generated Table:", extra_cols)

    # Row alignment
    row_match_map, extra_rows = optimal_row_assignment(ref_df, reordered_df)
    print("➕ Extra Rows in Generated Table (indices):", extra_rows)
    print("🔗 Row Match Map:", row_match_map)
    aligned_gen_df = align_generated_table_rows(ref_df, reordered_df, row_match_map)
    print("\n🔗 Reference Table:")
    print(ref_df)
    print("\n📋 Final Row-Aligned Table:")
    print(aligned_gen_df)

    # reset index
    aligned_gen_df.reset_index(drop=True, inplace=True)
    ref_df.reset_index(drop=True, inplace=True)

    # Evaluate
    tp, fp, fn, hc = compare_cells(ref_df, aligned_gen_df, extra_columns=extra_cols, extra_rows=extra_rows)

    # Print metrics
    print("\n📊 Comparison Results:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Hallucinated Cells (HC): {hc}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
