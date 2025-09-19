import os
import pandas as pd
import numpy as np


def load_dataset(pk_name):

    # Path to the dataset directory
    input_dir = "dataset/input/"
    labeled_dir = "dataset/labeled/"


    # Target metadata columns
    meta_columns = [
        "article_doi", "article_title", "caption",
        "legend", "abstract", "table_id"
    ]

    # Helper: extract numeric ID from filename
    def extract_numeric_id(filename):
        filename = filename.split("_")[0]
        return int(os.path.splitext(filename)[0])


    # Get list of labeled files (assuming labeled files use the same filenames)
    labeled_files = sorted(
        [f for f in os.listdir(labeled_dir) 
        if f.endswith((".xlsx", ".xls"))],
        key=extract_numeric_id
        )


    # Filter input files to only those that are labeled
    input_files = sorted(
        [f for f in os.listdir(input_dir) 
        if f.endswith((".xlsx", ".xls")) and f in labeled_files],
        key=extract_numeric_id
    )

    dataset_list = []

    for filename in input_files:

        input_df = pd.read_excel(os.path.join(input_dir, filename), sheet_name=0)
        labeled_df = pd.read_excel(os.path.join(labeled_dir, filename), sheet_name=0)

        # if the input_df is empty, skip the file
        if input_df.empty:
            print(f"Skipping empty file: {filename}")
            continue

        # if there is no pk_parameter_value in the labeled_df, then replace the first row as the column header and then drop the first row
        if labeled_df.empty is False and 'pk_parameter_value' not in labeled_df.columns:
            labeled_df.columns = labeled_df.iloc[0]
            labeled_df = labeled_df.drop(labeled_df.index[0]).reset_index(drop=True)
        
        labeled_df = labeled_df.apply(lambda col: col.astype(str).str.replace(',', '') if col.dtype == 'object' else col)

        # Extract metadata from first row
        metadata = {col: input_df[col].iloc[0] if col in input_df.columns else None for col in meta_columns}
        
        # Drop metadata columns and the first row
        table_df = input_df.drop(columns=meta_columns, errors='ignore')
        # remove ',' with ' ' in the table_df
        table_df = table_df.apply(lambda col: col.astype(str).str.replace(',', '') if col.dtype == 'object' else col)
        
        table_df.columns = table_df.columns.str.replace("@", "^", regex=False)
        table_df.columns = table_df.columns.str.replace(" - ", "^", regex=False)

        # Assign metadata to variables
        article_doi = metadata["article_doi"]
        article_title = metadata["article_title"]
        caption = metadata["caption"]
        footnote = metadata["legend"]
        article_abstract = metadata["abstract"]
        table_id = metadata["table_id"]

        try:
            labeled_df_filtered = labeled_df[labeled_df['abbv'] == pk_name].drop(columns=['abbv'], errors='ignore')
        except KeyError:
            # for empty dataframes
            labeled_df_filtered = labeled_df

        dataset_list.append({
            "filename": filename,
            "article_doi": article_doi,
            "article_title": article_title,
            "caption": caption, 
            "footnote": footnote,
            "article_abstract": article_abstract,
            "input_table": table_df,
            "labeled_table": labeled_df_filtered,
        })

    return dataset_list