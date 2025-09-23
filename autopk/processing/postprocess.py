"""
postprocess.py
Standardize and unify PK extraction tables into consistent format.
"""

import unicodedata
from unidecode import unidecode
import pandas as pd


def unification_tables(df):

    # remove rows which the pk_parameter_value is empty or nan or None or 'nan'
    df = df.dropna(subset=['pk_parameter_value'], axis=0, how='any')

    # strip the pk_parameter_value and pk_parameter_unit
    df['pk_parameter_value'] = df['pk_parameter_value'].astype(str).str.strip()
    df['pk_parameter_unit'] = df['pk_parameter_unit'].astype(str).str.strip()
    
    df = df[~df['pk_parameter_value'].isin({
        'nan', 'NaN', 'None', 'ND', 'NA', 'N.A.', 'na', 'n.a.', '', 
        'NR', 'N/A', 'n/a', 'NE', 'N.E.', 'n.e.', 'n.e', 'n/a', 'N/A', 'NM', 'n.m.', '_', '-', '–', 'None', 'null', 'NULL', 'none'
    })]


    # df = df.sort_values(by='pk_parameter_value')
    df = df.fillna("None")
    # remove rows which the pk_parameter is 'a' or 'b'
    df = df[~df['pk_parameter'].isin({
        'a', 'b', 'c', 'd', 'α', 'β', 'γ'
        })]
    
    df['pk_parameter'] = df['pk_parameter'].map(lambda x: unidecode(unicodedata.normalize("NFKD", x)) if isinstance(x, str) else x)

    route_mapping = {
        'PO': {'po', 'oral', 'p.o.', 'per os', 'p.o', 'oral gavage'},
        'IV': {'iv', 'intravenous', 'i.v.', 'i.v', 'intra-venous'},
        'IM': {'im', 'intramuscular', 'i.m.', 'i.m'},
        'SC': {'sc', 'subcutaneous', 's.c.', 'subq', 'sub-q', 's.c'},
        'SL': {'sl', 'sublingual'},
        'BUCC': {'bucc', 'buccal'},
        'TOP': {'top', 'topical', 'topical infusion', 'topicalinfusion'},
        'TD': {'td', 'transdermal'},
        'INH': {'inh', 'inhalation', 'inhaled'},
        'PR': {'pr', 'rectal'},
        'IN': {'in', 'intranasal', 'nasal'},
        'OPHTH': {'ophth', 'ophthalmic', 'eye'},
        'OTIC': {'otic', 'ear'},
        'IT': {'it', 'intrathecal'},
        'IO': {'io', 'intraosseous'},
        'IP': {'ip', 'intraperitoneal','intraportal', 'i.p.', 'i.p'},
        'IVT': {'ivt', 'intravitreal'},
        'IPP': {'ipp', 'intrapleural'},
        'IA': {'ia', 'intraarterial'},
        'IC': {'ic', 'intracardiac'},
        'ID': {'id', 'intradermal','intraduodenal'},
        'SUB': {'sub', 'subungual'},
        'VAG': {'vag', 'vaginal'},
        'DERM': {'derm', 'dermal'},
        'TR': {'tr', 'transrectal'},
        'IMM': {'imm', 'intramammary'},
        'IG': {'ig', 'intragastric', 'i.g.'},
    }
    flat_map_route = {alias.lower(): standard for standard, aliases in route_mapping.items() for alias in aliases}

    df['route_of_administration'] = df['route_of_administration'].apply(
        lambda x: flat_map_route.get(x.strip().lower(), x) if isinstance(x, str) else x
    )

    unit_mapping = {
        'mg': {'milligram', 'milligrams'},
        'g': {'gram', 'grams'},
        'μg': {'microgram', 'micrograms'},
        'ng': {'nanogram', 'nanograms'},
        'pg': {'picogram', 'picograms'},
        'l': {'liter', 'liters'},
        'ml': {'milliliter', 'milliliters'},
        'μl': {'microliter', 'microliters'},
        'nl': {'nanoliter', 'nanoliters'},
        'pl': {'picoliter', 'picoliters'},
        'IU': {'international unit', 'international units'},
        'U': {'unit', 'units'},
        'mmol': {'millimole', 'millimoles'},
        'μmol': {'micromole', 'micromoles'},
        'nmol': {'nanomole', 'nanomoles'},
        'pmol': {'picomole', 'picomoles'},
        'mEq': {'milliequivalent', 'milliequivalents'},
        'μEq': {'microequivalent', 'microequivalents'},
        'nmol/l': {'nanomole per liter', 'nanomoles per liter'},
        'h': {'hours', 'hour'},
        'min': {'minutes', 'minute'},
        's': {'seconds', 'second'},
        'd': {'days', 'day'},
        'wk': {'weeks', 'week'},
        'mo': {'months', 'month'},
        'yr': {'years', 'year'},
    }
    flat_map_unit = {alias.lower(): standard for standard, aliases in unit_mapping.items() for alias in aliases}
    
    df['pk_parameter_unit'] = df['pk_parameter_unit'].apply(
        lambda x: flat_map_unit.get(x.strip().lower(), x) if isinstance(x, str) else x
    )

    df['animal'] = df['animal'].astype(str).apply(lambda x: {"human": "patient", "humans": "patients"}.get(x.strip().lower(), x) if isinstance(x, str) else x)

    df['animal_matrix/commodity'] = df['animal_matrix/commodity'].astype(str).apply(lambda x: {"blood": "plasma"}.get(x.strip().lower(), x) if isinstance(x, str) else x)
    # remove spaces
    df[['pk_parameter', 'pk_parameter_value', 'drug_dosage', 'route_of_administration', 'animal_matrix/commodity']] = df[['pk_parameter', 'pk_parameter_value', 'drug_dosage', 'route_of_administration', 'animal_matrix/commodity']].astype(str).apply(lambda x: x.str.replace(" ", "", regex=False))
    df = df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)

    df['pk_parameter_value'] = df['pk_parameter_value'].str.replace("+/-", "±", regex=False)

    # Use the Administrated

    df.reset_index(drop=True, inplace=True)

    return df