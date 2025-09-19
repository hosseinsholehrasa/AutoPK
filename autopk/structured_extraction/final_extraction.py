"""
final_extraction.py
Pipeline 2: Structured PK parameter extraction from tables.
"""

import pandas as pd
from io import StringIO

from autopk.llm_utils import llm_chat_with_history
from autopk.structured_extraction.pk_final_prompts import build_final_extraction_examples, final_extraction_config


# ==============================
# Generate structured CSV via LLM
# ==============================
def generate_extraction_csv(
    extracted_rows: set,
    pk_name: str,
    pk_config: dict = final_extraction_config,
    model_name: str = "llama3",
    footnote: str = "None",
    caption: str = "None",
    title: str = "None",
    abstract: str = "None",
) -> str:
    """
    Build chat history using the prompt + examples for a given pk_name,
    send extracted_rows and document context to the LLM, and return CSV string.
    """

    # 1) Build prompt and examples
    prompt, example_messages = build_final_extraction_examples(pk_name, pk_config)

    # 2) Build chat history
    chat_history = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You converting the user text format to table csv format."
        },
    ] + example_messages + [
        {
            "role": "user",
            "content": f"""{prompt}

This is my custom format table:
{extracted_rows}

This is footnote of my table in document:
{footnote}

This is caption of my table in document:
{caption}

This is title of my document:
{title}

This is abstract of my document:
{abstract}

Output Produce nothing except the final CSV lines in the order specified.
""",
        }
    ]

    # 3) Query LLM
    reply, _ = llm_chat_with_history(history=chat_history, model_name=model_name, max_tokens=2500)

    return reply


# ==============================
# CSV Parsing Helpers
# ==============================
def fix_and_parse_llm_csv(raw_csv, expected_columns: int = 8) -> pd.DataFrame:
    """
    Fix and parse a raw CSV string from the LLM into a DataFrame.
    Handles cases where drug names contain commas.
    """
    fixed_lines = []
    for line in raw_csv.strip().splitlines():
        parts = line.strip().split(",")

        if len(parts) > expected_columns:
            # Assume the extra comma is in the "drug" field (index 4)
            pre = parts[:4]
            drug_parts = []
            post = []

            i = 4
            while i < len(parts) and not parts[i].strip().endswith(("mg", "g")):
                drug_parts.append(parts[i])
                i += 1

            if i < len(parts):
                drug_dosage = parts[i]
                post = parts[i + 1 :]
            else:
                drug_dosage = "None"
                post = ["None"] * (expected_columns - len(pre) - 2)

            drug = ",".join(drug_parts).strip()
            drug = f'"{drug}"' if "," in drug else drug
            fixed_line = ",".join(pre + [drug, drug_dosage] + post)
        else:
            fixed_line = ",".join(parts)

        fixed_lines.append(fixed_line)

    try:
        df = pd.read_csv(StringIO("\n".join(fixed_lines)))
    except Exception as e:
        print("CSV parsing failed:", e)
        df = pd.DataFrame()

    return df


def convert_llm_table2df(llm_table: str) -> pd.DataFrame:
    """
    Convert a clean LLM-generated CSV string into a Pandas DataFrame.
    Falls back to fix_and_parse_llm_csv if parsing fails.
    """
    lines = llm_table.strip().split("\n")

    header = [col.strip() for col in lines[0].strip().split(",")]
    data = [
        [cell.strip() for cell in line.strip().split(",")]
        for line in lines[1:] if line.strip()
    ]
    try:
        df = pd.DataFrame(data, columns=header)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        df = fix_and_parse_llm_csv(llm_table, expected_columns=len(header))

    return df
