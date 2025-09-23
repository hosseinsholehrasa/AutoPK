"""
run_evaluation.py
Run evaluation of AutoPK pipelines across target PK parameters.
"""

import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from AutoPK.autopk.processing.postprocess import unification_tables
from autopk.form_detection.form_extraction import find_pk_parameter_locations, extract_rows
from autopk.structured_extraction.final_extraction import (
    generate_extraction_csv, convert_llm_table2df
)
from autopk.structured_extraction.pk_final_prompts import build_final_extraction_examples, final_extraction_config

from experiments.dataset_loader import load_dataset
from autopk.evaluation import table_eval  

# -------------------
# Main Evaluation Loop
# -------------------

def run_evaluation(
    target_pk_parameters,
    threshold: float,
    pipeline2_model_name: str,
    save_log_file_name: str,
    weights=(0.6, 0.2, 0.2),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Run evaluation over given PK parameters and return aggregated scores.

    Args:
        target_pk_parameters (list): PK parameters to evaluate (e.g., ["half-life"]).
        threshold (float): Similarity threshold for form detection.
        pipeline2_model_name (str): LLM model name for structured extraction.
        save_log_file_name (str): Path to save logs.
        weights (tuple): Weighting of similarity components (cosine, edit, token).
        device (str): "cpu" or "cuda".
    """
    embedding_cache = {}
    scores_caches = {}
    scores_log_all = {
        str(threshold): {parm: {} for parm in target_pk_parameters}
    }

    for target_pk_parameter in target_pk_parameters:
        print(f"🔬 Running evaluation for target PK parameter: {target_pk_parameter}")

        # -----------------
        # Dataset Loading
        # -----------------
        dataset_list = load_dataset(target_pk_parameter)
        if not dataset_list:
            print(f"No datasets found for {target_pk_parameter}")
            continue

        # ---- Hardcoded initial variants (Based on pipeline 1, we removed the false positives) ----
        if target_pk_parameter == 'half-life':
            unique_llm_parameters_dict = {
                "f1": ['Alpha t1/2 (minutes)', 'Apparent absorption half-life(h)', 'Apparent elimination half-life(h)', 'Beta t1/2 (minutes)', 'Dissociation half-life (h)', 'Elimination T1/2 (hrs)', 'Elimination half-life', 'Elimination half-life (h)', 'Elimination t1/2 (h)', 'EliminationT1/2', 'HLM', 'HLM T1/2 b (min)', 'HLM t 1/2 (min)', 'HLM/DLM t ½ (min)', 'HLMc', 'HL_Lambda_z', 'Half life Phase 1 (t1/2α)', 'Half life Phase 2 (t1/2β)', 'Half life: ln(2)/α (min)', 'Half life: ln(2)/β (min)', 'Half-Life (h)', 'Half-Life (hours)', 'Half-Life (t 1/2, h)', 'Half-Life a (h)', 'Half-life (h)', 'Half-life (hours)', 'Half-life (hr)', 'Half-life (min)', 'Half-life (t1/2 min)', 'Half-life (t1/2, days)', 'Half-life [h]', 'Half-life h', 'Half-life λZ (hour)', 'Hu Mic T½ (min)', 'K1/2el(h−1)', 'K10 HL', 'Mo Hepatocytes T½ (min)', 'Oral half life, t 1/2 (h)', 'Rat liver microsome t 1/2 (min)', 'Rat t 1/2 b (h)', 'T 1/2', 'T 1/2 (h)', 'T 1/2 (hr)', 'T 1/2 (min)', 'T 1/2 a (h)', 'T 1/2 e', 'T 1/2 iv (h)', 'T 1/2 α (h)', 'T 1/2 β (h)', 'T 1/2, eff (h)', 'T 1/2/h', 'T 1/2abs', 'T 1/2el', 'T 1/2α', 'T 1/2α (h)', 'T 1/2β', 'T 1/2β (h)', 'T 1/2β(h)', 'T 1/2λ', 'T ½', 'T ½ (min)', 'T ½ (minutes)', 'T ½ z (h)', 'T ½ α (h)', 'T ½ β (h)', 'T1/2', 'T1/2 (H)', 'T1/2 (d)', 'T1/2 (h)', 'T1/2 (h) p.o.', 'T1/2 (hr)', 'T1/2 (min)', 'T1/2 Eli (min)', 'T1/2 h', 'T1/2(h)', 'T1/2, h', 'T1/2ab', 'T1/2el', 'T1/2α', 'T1/2α (h)', 'T1/2α (min)', 'T1/2β', 'T1/2β (h)', 'T1/2β (min)', 'T1/2λz', 'Terminal half-life', 'Terminal half-life (h)', 'Terminal half-life (hours)', 'Terminal half-life (min)', 'Terminal half-life (t 1/2)', 'Terminal half-life calculated with λz (h)', 'Terminal plasma half-life (h)', 'Terminal t 1/2', 'Terminal t 1/2 (h)', 'Terminal t 1/2 (min)', 'Terminal t1/2 (min)', 'T½ (h)', 'T½ z (h)', 'T½ α', 'T½ β', 'T½ γ', 'k01 t1/2 (minutes)', 'k10 T1/2 (hr)', 'k10 t1/2 (minutes)', 'ln2/k 01 (min)', 't 1/2', 't 1/2 (h)', 't 1/2 (h) a', 't 1/2 (h) serum', 't 1/2 (h) urine', 't 1/2 (min)', 't 1/2 K 01', 't 1/2 K 10', 't 1/2 alpha', 't 1/2 b', 't 1/2 b (h)', 't 1/2 beta', 't 1/2 d', 't 1/2 pi', 't 1/2 ß (minutes)', 't 1/2 α', 't 1/2 α (t 1/2ab)', 't 1/2 β', 't 1/2 β (t 1/2el)', 't 1/2 γ', 't 1/2 λz', 't 1/2(a) (h) a', 't 1/2(d) (h) a', 't 1/2(h)', 't 1/2(min)', 't 1/2(×104)', 't 1/2- λ z', 't 1/2-ka', 't 1/2/h', 't 1/2ab', 't 1/2el', 't 1/2el (min)', 't 1/2elim', 't 1/2int', 't 1/2ka', 't 1/2z (h)', 't 1/2α', 't 1/2α (h)', 't 1/2α (hr)', 't 1/2α (min)', 't 1/2α (t 1/2ab) (HM)', 't 1/2α /t 1/2Ka', 't 1/2α b', 't 1/2β', 't 1/2β (HM)', 't 1/2β (h)', 't 1/2β (hr)', 't 1/2β (min)', 't 1/2β b', 't 1/2γ (h)', 't 1/2λ 1', 't 1/2λ z', 't 1/2λz', 't 1/2λz (min)', 't 1/2λz h', 't ½ (min)', 't ½ (po) (h)', 't ½ λ', 't ½ λ z', 't ½α (h)', 't ½β (h)', 't1/2', 't1/2 (d)', 't1/2 (h)', 't1/2 (h) a', 't1/2 (hour)', 't1/2 (hr)', 't1/2 (hrs)', 't1/2 (min)', 't1/2 Lambda z', 't1/2 [h]', 't1/2 _ka (h)', 't1/2 abs', 't1/2 days', 't1/2 el α', 't1/2 el β', 't1/2 h', 't1/2 hours', 't1/2 α', 't1/2 β', 't1/2 β†', 't1/2 λz (min)', 't1/2ss h', 't1/2α', 't1/2α (h)', 't1/2β', 't1/2β (h)', 't1/2β (min)', 't1/2λ', 't1/2λz (min)', 'terminal t 1/2 (h)', 't½ (h)', 't½ (hours)', 't½ (min)', 't½ h', 't½ harmonic mean h', 't½ mean (SD) h', 't½ min', 't½ ⁎ (h)', 't½ ⁎⁎ (h)', 't½(h)', 't½Ka (h)', 't½α', 't½β', 't½β day', 't½λ1 (h) ⁎', 't½λZ (h) ⁎', 'α Half-life (h)', 'α T1/2 (hr)', 'β Half-life (h)', 'β T1/2 (hr)'],
                # "f2": ['Apparent absorption half-life(h)', 'Apparent elimination half-life(h)', 'Distribution half-life (h)', 'Distribution half-life (min)', 'Elimination half life', 'Elimination half-life t 1/2 (h)', 'Half-Life (h)', 'Half-life (h)', 'Half-life (h) elimination', 'Half-life (min)', 'Liver Microsome Stability T 1/2 (min) c', 'Plasma t 1/2 (h)', 'T 1/2', 'T 1/2 (h)', 'T 1/2 (hr)', 'T 1/2 e', 'T 1/2β(h)', 'T ½ (min)', 'T1/2', 'T1/2 (h)', 'T1/2 (min)', 'T1/2ab', 'T1/2dis', 'T1/2el', 'T1/2α (h)', 'T1/2β', 'T1/2β (h)', 'T1/2λz', 'Terminal T 1/2 (h)', 'Terminal half life', 'Terminal half-life (min)', 'T½ (h)', 'T½ α', 'T½ β', 'T½ γ', 't 1/2', 't 1/2 (h)', 't 1/2 K 01', 't 1/2 K 10', 't 1/2 α (min)', 't 1/2 β (min)', 't 1/2(h)', 't 1/2Ka (h)', 't 1/2ab', 't 1/2el', 't 1/2ka', 't 1/2α', 't 1/2α (h)', 't 1/2α (t 1/2ab)', 't 1/2β', 't 1/2β (h)', 't 1/2β(t 1/2el)', 't 1/2λ z', 't 1/2λz', 't 1/2λz h', 't ½ (h)', 't ½ λ', 't ½ λ z', 't ½α (h)', 't ½β (h)', 't1/2', 't1/2 (day)', 't1/2 (h)', 't1/2 (min)', 't1/2 Lambda z', 't1/2 h', 't1/2 λz (min)', 't1/2(h)', 't1/2ss h', 't1/2α (h)', 't1/2β (h)', 't1/2β (min)', 't1/2λ', 't½ (h)', 't½ e (h)', 't½ mean (SD) h'],
                # "f3": ['Alpha t1/2 (minutes)', 'Apparent absorption half-life(h)', 'Apparent elimination half-life(h)', 'Beta t1/2 (minutes)', 'Distribution half-life (h)', 'Distribution half-life (min)', 'Elimination half life', 'Elimination half-life', 'Elimination half-life (h)', 'Elimination half-life from aqueous humour (day)', 'Elimination half-life t 1/2 (h)', 'Half-Life (h)', 'Half-Life (hours)', 'Half-Life (t 1/2, h)', 'Half-Life a (h)', 'Half-life (h)', 'Half-life (h) elimination', 'Half-life (min)', 'Half-life (t1/2 min)', 'Half-life [h]', 'Half-life h', 'Half‐life (days)', 'Liver Microsome Stability T 1/2 (min) c', 'Plasma t 1/2 (h)', 'T 1/2', 'T 1/2 (h)', 'T 1/2 (hr)', 'T 1/2 (min)', 'T 1/2 e', 'T 1/2/h', 'T 1/2abs', 'T 1/2el', 'T 1/2α', 'T 1/2α (h)', 'T 1/2β', 'T 1/2β (h)', 'T 1/2β(h)', 'T 1/2λ', 'T ½ (min)', 'T ½ (minutes)', 'T1/2', 'T1/2 (H)', 'T1/2 (d)', 'T1/2 (h)', 'T1/2 (hr)', 'T1/2 (min)', 'T1/2 Eli (min)', 'T1/2(h)', 'T1/2ab', 'T1/2dis', 'T1/2el', 'T1/2α', 'T1/2α (h)', 'T1/2β', 'T1/2β (h)', 'T1/2λz', 'Terminal T 1/2 (h)', 'Terminal half life', 'Terminal half-life', 'Terminal half-life (min)', 'Terminal t 1/2', 'Terminal t 1/2 (min)', 'Terminal t1/2 (min)', 'T½ (h)', 'T½ α', 'T½ β', 'T½ γ', 'Vitreal elimination half-life (day)', 'k01 t1/2 (minutes)', 'k10 t1/2 (minutes)', 't 1/2', 't 1/2 (h)', 't 1/2 (min)', 't 1/2 K 01', 't 1/2 K 10', 't 1/2 hours', 't 1/2 α', 't 1/2 α (min)', 't 1/2 β', 't 1/2 β (h)', 't 1/2 β (min)', 't 1/2 γ', 't 1/2(h)', 't 1/2(×104)', 't 1/2- λ z', 't 1/2-ka', 't 1/2/h', 't 1/2Ka (h)', 't 1/2a (h)', 't 1/2ab', 't 1/2abs (h)', 't 1/2el', 't 1/2el (min)', 't 1/2elim', 't 1/2int', 't 1/2ka', 't 1/2z (h)', 't 1/2α', 't 1/2α (h)', 't 1/2α (hr)', 't 1/2α (min)', 't 1/2α (t 1/2ab)', 't 1/2α b', 't 1/2β', 't 1/2β (h)', 't 1/2β (hr)', 't 1/2β (min)', 't 1/2β a (h)', 't 1/2β b', 't 1/2β(t 1/2el)', 't 1/2λ 1', 't 1/2λ z', 't 1/2λ1', 't 1/2λz', 't 1/2λz (min)', 't 1/2λz h', 't ½ (h)', 't ½ (min)', 't ½ λ', 't ½ λ z', 't ½α (h)', 't ½β (h)', 't1/2', 't1/2 (day)', 't1/2 (h)', 't1/2 (h) a', 't1/2 (hour)', 't1/2 (hours)', 't1/2 (min)', 't1/2 Lambda z', 't1/2 [h]', 't1/2 _ka (h)', 't1/2 h', 't1/2 λz (h)', 't1/2 λz (min)', 't1/2(h)', 't1/2ss h', 't1/2α', 't1/2α (h)', 't1/2β', 't1/2β (h)', 't1/2β (min)', 't1/2λ', 't1/2λ1', 't1/2λ2', 't1/2λz (min)', 't½ (h)', 't½ e (h)', 't½ h', 't½ harmonic mean h', 't½ mean (SD) h', 't½α', 't½α (h)', 't½β', 't½β (h)'],
            }
        elif target_pk_parameter == 'AUC':
            unique_llm_parameters_dict = {
                "f1": ['% AUC extrapolated', '(AUC0→∞)', 'A U C 0 − ∞ (μg·h/L)', 'AU C 0 – 2 4 ss (μgh/mL) a', 'AUC', 'AUC ( 0– ∞) (μgminmL−1)', 'AUC (% extrapolated)', 'AUC (%ID h/mL blood)', 'AUC (0 to infinity) (hμg/ml)', 'AUC (0,168) (μM×h)', 'AUC (0-24) (mg h/L)', 'AUC (0-t) (μg/L*h)', 'AUC (0-∞) (μg min mL−1)', 'AUC (0-∞) (μg/L*h)', 'AUC (0-∞)(ng·h/mL)', 'AUC (0–t) (ngmin/mL)', 'AUC (0–α) ngh/mL', 'AUC (0–∞) (ngmin/mL)', 'AUC (0–∞) muscle/AUC (0–∞) plasma', 'AUC (IU anti-FXa/mL×min)', 'AUC (day·μg/mL)', 'AUC (drain fluid) (mgh/L)', 'AUC (h % of dose/mL)', 'AUC (h * ng/mL) a', 'AUC (h.mg/L)', 'AUC (hng/mL)', 'AUC (h·μg/mL)', 'AUC (h×ng/mL)', 'AUC (i.v.) day·μg/ml)', 'AUC (mg h/l)', 'AUC (mg/L/h)', 'AUC (mg/Lh)', 'AUC (mg/l·h)', 'AUC (mghl−1)', 'AUC (mg⋅h/L)', 'AUC (min µg/mL)', 'AUC (min.ng/mL)', 'AUC (n = 11)', 'AUC (nM*h)', 'AUC (ng hour mL–1)', 'AUC (ng minute L–1)', 'AUC (ng minute mL–1)', 'AUC (ng · h/mL)', 'AUC (ng/mL × h)', 'AUC (ng/ml·h)', 'AUC (ngh/kg)', 'AUC (ng·h/mL)', 'AUC (pg/mL · h)', 'AUC (plasma) (mgh/L)', 'AUC (total) (hngml−1)', 'AUC (µg·h/L)', 'AUC (µg·min/mL)', 'AUC (μM × h)', 'AUC (μg h/ mL)', 'AUC (μg min per ml)', 'AUC (μg min/ml)', 'AUC (μg · min/mL)', 'AUC (μg/ml h)', 'AUC (μg/ml/hr)', 'AUC (μgh/ml)', 'AUC (μgmin/ml)', 'AUC (μg·h/mL)', 'AUC (μg·hr/mL)', 'AUC (μg·min/mL)', 'AUC (μg·min/ml)', 'AUC (μg∙min/mL)', 'AUC 0 ∞ (h·μg/mL)', 'AUC 0-60 (μmol·min/L)', 'AUC 0-60 ratio', 'AUC 0-8h (min·ng/mL)', 'AUC 0-inf (h ng/mL)', 'AUC 0-inf (h × mg L−1)', 'AUC 0-inf/D (h ng/mL/μg/kg)', 'AUC 0-infinity (h.μg/mL or g)', 'AUC 0-last time (h.μg/mL or g)', 'AUC 0-t (ng/mLh)', 'AUC 0-t,ng/mL*h', 'AUC 0-∞ (min·ng/mL)', 'AUC 0-∞ (ng/mLh)', 'AUC 0-∞ (ng·h/mL)', 'AUC 0-∞ (ug/mL.h)', 'AUC 0-∞ (μg h/mL)', 'AUC 0-∞,ng/mL*h', 'AUC 0–t observed area', 'AUC 0–∞', 'AUC 0–∞ observed area', 'AUC 0−t (hmg/L)', 'AUC 0−∞ (hmg/L)', 'AUC 0−∞,ng/mL∙h', 'AUC AMP (ng/min/ml)', 'AUC METH (ng/min/ml) (normalized for dose)', 'AUC Trapezoidal', 'AUC a (μM*hr)', 'AUC all (hour•μg mL−1)', 'AUC b (μg·min/mL)', 'AUC c', 'AUC c (μg·min/mL)', 'AUC ng·h/mL', 'AUC ratio (drain:plasma)', 'AUC ratio, PO/IV', 'AUC trapezoidal rule (t =0−∞) (hngml−1)', 'AUC trapezoidal rule (t =7) (hngml−1)', 'AUC ∞ (μMh)', 'AUC% extrap %', 'AUC(0 tlast) (ngh/ml)', 'AUC(0-INF)(h×ng/mL)', 'AUC(0-t)', 'AUC(0-t) (h*ng/mL)', 'AUC(0-t) (ng ⋅ h /ml)', 'AUC(0-t) (ng·h/mL)', 'AUC(0-t) (ug.h/L)', 'AUC(0-t) (μg/L*h)', 'AUC(0-∞)', 'AUC(0-∞) (h*ng/mL)', 'AUC(0-∞) (ug.h/L)', 'AUC(0-∞) (μg/L*h)', 'AUC(0-∞)(ng ⋅ h /ml)', 'AUC(0~48h) ((h·μg)/mL)', 'AUC(0~∞) ((h·μg)/mL)', 'AUC(0– t )', 'AUC(0–12) (μgh/ml)', 'AUC(0–24) (μg*h/mL)', 'AUC(0–8 h) b (μgh/mL)', 'AUC(0–inf) μg·day/mL', 'AUC(0–last) (μgh/mL)', 'AUC(0–t last)', 'AUC(0–t last) (ngh/mL)', 'AUC(0–t)', 'AUC(0–t) μg·day/mL', 'AUC(0–∞)', 'AUC(0–∞) (μgh/mL)', 'AUC(0→t) (μg/mLmin)', 'AUC(0→t) (μg/mL•min)', 'AUC(0→∞) (μg/mLmin)', 'AUC(0→∞) (μg/mL•min)', 'AUC(0− t) (μg/lh)', 'AUC(μM·hr)', 'AUC(μg h/mL)', 'AUC, pg ∙ min/mL', 'AUC,(min μ g)/mL', 'AUC,(μg/mL)·h', 'AUC,μgml−1 h−1', 'AUC-inf (pg h mL−1)', 'AUC-last (pg h mL−1)', 'AUC/ Dose (μM ·hr/mg/kg)', 'AUC/dose (dose normalized) (h×ng/mL)/(mg/kg)', 'AUC/dose (μg·day/mL/mg)', 'AUC0 →∞ (μg h/L)', 'AUC0 ∞ (% activity·day)', 'AUC0-12 (µg·h/mL)', 'AUC0-12 h∗ng/mL', 'AUC0-120 min (min*µg/mL)', 'AUC0-120 min (min*µg/mL) Non-ischemic Limb*', 'AUC0-12h (h·nM)', 'AUC0-21day', 'AUC0-24 (mg/L*h)', 'AUC0-24 (ng/mL∗h)', 'AUC0-24 h (µg⋅h/ml)', 'AUC0-24 h∗ng/mL', 'AUC0-24(μg/L × h)', 'AUC0-24h (h*μM)', 'AUC0-24h (h·nM)', 'AUC0-5d (μg/mL.h)', 'AUC0-96 h (μM*h)', 'AUC0-INF (h*ng/mL)', 'AUC0-INF (μg·h/mL)', 'AUC0-inf', 'AUC0-inf (hng/mL)', 'AUC0-inf (nM h)', 'AUC0-inf (ng eq·h−1·mL−1)', 'AUC0-inf (ng*hour/ml)', 'AUC0-inf (ng.h/mL)', 'AUC0-inf (ng.min/ml)', 'AUC0-inf (μgmin/mL)', 'AUC0-last (h*ng/mL)', 'AUC0-last (ng*hour/ml)', 'AUC0-last (ng.h/mL)', 'AUC0-t', 'AUC0-t (h ng/ml)', 'AUC0-t (h*ng/mL)', 'AUC0-t (h*μg/mL)', 'AUC0-t (h.ng/mL)', 'AUC0-t (h.ng/mL) i.v.', 'AUC0-t (h.ng/mL) p.o.', 'AUC0-t (h·ng/mL)', 'AUC0-t (h·μg/mL)', 'AUC0-t (mg h/l)', 'AUC0-t (min*ng/mL)', 'AUC0-t (ng.h/mL)', 'AUC0-t (ng/mL*h)', 'AUC0-t (ng/ml/h)', 'AUC0-t (μ Mh)', 'AUC0-t (μg h/mL)', 'AUC0-t (μg/L∗h)', 'AUC0-t h∗ng/mL', 'AUC0-t ng.h/ml', 'AUC0-t(μg-h/mL)', 'AUC0-• (min*ng/mL)', 'AUC0-∞', 'AUC0-∞ (h ng/ml)', 'AUC0-∞ (h µg mL−1)', 'AUC0-∞ (h*ng/mL)', 'AUC0-∞ (h*pg/mL)', 'AUC0-∞ (h*μg/mL)', 'AUC0-∞ (h·ng/mL)', 'AUC0-∞ (h·μg/mL)', 'AUC0-∞ (min*ng/mL)', 'AUC0-∞ (ng.h/mL)', 'AUC0-∞ (ng/mL*h)', 'AUC0-∞ (ng·h/mL)', 'AUC0-∞ (µg·h/mL)', 'AUC0-∞ (μM∗h)', 'AUC0-∞ (μg h/mL)', 'AUC0-∞ (μg/mL.h)', 'AUC0-∞ (μg·min/mL)', 'AUC0-∞ h∗ng/mL', 'AUC0-∞ ng • h/mL', 'AUC0-∞(μg-h/mL)', 'AUC0-∞(μg/L × h)', 'AUC0-∞(μg/L∗h)', 'AUC0-∞,μg/mL h', 'AUC0‒∞ (h·μg/L)', 'AUC0– t', 'AUC0– t (μg/mL h)', 'AUC0– ∞ (ng h/mL)', 'AUC0–12', 'AUC0–12 (h∗μg/mL)', 'AUC0–12 (μg·h/ml)', 'AUC0–12 c', 'AUC0–12 h (mg h/L)', 'AUC0–12, oral (mg · h/L)‡', 'AUC0–12/24 (mgh/L)', 'AUC0–120 (nmol×min/L)', 'AUC0–12h (mg ∗ h/L)', 'AUC0–12h μg·h/L', 'AUC0–12hss μg·h/L', 'AUC0–14d (ng∗h/ml)', 'AUC0–1d (ng∗h/ml)', 'AUC0–21d (ng∗h/ml)', 'AUC0–24', 'AUC0–24 (h*ng/mL)', 'AUC0–24 (mg L−1 h)', 'AUC0–24 (mg·h/L)', 'AUC0–24 (ng h/mL)', 'AUC0–24 (pg.h/mL)', 'AUC0–240 (mg/lmin)', 'AUC0–240 (μg/ml*min)', 'AUC0–240 min (μg·min/mL)', 'AUC0–240 pg·min/mL', 'AUC0–240/AUC0−inf', 'AUC0–24h (μg · h/mL)', 'AUC0–24h [ngh/mL]', 'AUC0–504norm (ng·h/ml/mg)', 'AUC0–72 (hng/mL)', 'AUC0–72h (μg · h/mL)', 'AUC0–7d (ng∗h/ml)', 'AUC0–8 (mg ∗ h/L)', 'AUC0–8h (μg·min/mL)', 'AUC0–90 (min ng/mL)', 'AUC0–Tlast (ngh/mL)', 'AUC0–inf (h*μg/mL)', 'AUC0–inf (h.ng/mL)', 'AUC0–inf (hμM)', 'AUC0–inf (ng·h/ml)', 'AUC0–inf (μMh)', 'AUC0–inf (μg.ml/min)', 'AUC0–inf (μmol/L·h)', 'AUC0–inf ng·h/mL', 'AUC0–last (ng h/mL)', 'AUC0–last (ngh/ml)', 'AUC0–last (μmol/L·h)', 'AUC0–last ng · h/mL', 'AUC0–t', 'AUC0–t (h·nM)', 'AUC0–t (h·ng/mL)', 'AUC0–t (ng·h/ml)', 'AUC0–t mean (SD) h·µg/mL', 'AUC0–t µg·h/mL', 'AUC0–τ (mgh/L)', 'AUC0–∝ (μg/mLh)', 'AUC0–∝/D (kg/h/L)', 'AUC0–∞', 'AUC0–∞ (h*ng/mL)', 'AUC0–∞ (hour ng mL−1)', 'AUC0–∞ (h·nM)', 'AUC0–∞ (h∗ng/mL)', 'AUC0–∞ (mg h/L)', 'AUC0–∞ (mg/lmin)', 'AUC0–∞ (ng h/mL)', 'AUC0–∞ (ng/h/ml)', 'AUC0–∞ (ng/mL h)', 'AUC0–∞ (ngh/mL)', 'AUC0–∞ (ng·h/mL)', 'AUC0–∞ (ng∗h/ml)', 'AUC0–∞ (nmol×min/L)', 'AUC0–∞ (pg.h/mL)', 'AUC0–∞ (µg⋅h/mL)', 'AUC0–∞ (μg min/ml)', 'AUC0–∞ (μg · h/mL)', 'AUC0–∞ (μg/mL h)', 'AUC0–∞ (μgh−1/mL)', 'AUC0–∞ (μgml−1 h)', 'AUC0–∞ (μg∗h/mL)', 'AUC0–∞ [ngh/mL]', 'AUC0–∞ [ng·h/mL]', 'AUC0–∞ a (μgmin/ml)', 'AUC0–∞ mean (SD) (h·µg/mL)', 'AUC0–∞ mean (SD) h·µg/mL', 'AUC0–∞ ng · h/mL', 'AUC0–∞ ng·h/mL', 'AUC0–∞ pg·min/mL', 'AUC0–∞ µg · h/L', 'AUC0–∞ µg·h/mL', 'AUC0–∞(mg ∗ h/L)', 'AUC0–∞/dose (µg⋅h/mL/D)', 'AUC0→24 (mgh/L) a', 'AUC0→24h (μgh/L)', 'AUC0→48 (h∙ng/mL)', 'AUC0→4h (μgh/L)', 'AUC0→8 hr (μg min/ml)', 'AUC0→Inf (mgh/L)', 'AUC0→inf (h∙ng/mL)', 'AUC0→t (mgh/L) b', 'AUC0→t (μg h/L)', 'AUC0→t(minng/ml)', 'AUC0→t,minng/mL', 'AUC0→∞', 'AUC0→∞ (plasma h·ng/mL; skin hng/g)', 'AUC0→∞ (µg h mL− 1)', 'AUC0→∞ [ngmin/ml]', 'AUC0→∞(minng/ml)', 'AUC0→∞(μg·min/ml)', 'AUC0→∞,minutes ng−1 mL−1', 'AUC0−72 (hng/mL)', 'AUC0−8 (μghg−1)', 'AUC0−inf (ng·h/mL)', 'AUC0−t (ng·h/mL)', 'AUC0−t (μgh/ml)', 'AUC0−t /AUC0−∞(%)', 'AUC0−t/μghmL−1', 'AUC0−∞ (hμgmL−1)', 'AUC0−∞ (μg-h/mL)', 'AUC0−∞ (μghmL−1)', 'AUC0−∞ (μg·h/mL)', 'AUC0−∞ mean (SD) h·µg/mL', 'AUC0−∞ per dose h·µg/mL/mg', 'AUC0−∞/dose', 'AUC1 (h ⁎ nmol/L)', 'AUC12 (µg·h/mL)', 'AUC24 (mgh/L)', 'AUC24 (μg*h/ml)', 'AUC24h (h ⁎ nmol/L)', 'AUCCSF/AUCub', 'AUCCSF/AUCup', 'AUCCSF:AUCSerum [%]', 'AUCExtr', 'AUCExtra (%)', 'AUCINF (predicted) a (hμg/L)', 'AUCINF_obs(h·ng·mL−1)', 'AUCINF_pred', 'AUCInf', 'AUCNpo (μMh/mpk)', 'AUC^number^(ng h/ml)', 'AUC_%Extrap(%)', 'AUC_%extrap', 'AUCa (µg · min/mL)', 'AUCb/AUCp', 'AUCc (μM*hr)', 'AUCc / Dose (μM *hr/mg/kg)', 'AUCinf ( n g ∙ h ∙ m L - 1 )', 'AUCinf (h.ng/mL)', 'AUCinf (hng/mL)', 'AUCinf (nM*h)', 'AUCinf (ng hour mL–1)', 'AUCinf (ng mL−1 hour)', 'AUCinf (ngh/mL)', 'AUCinf (ng·h/mL)', 'AUCinf (ng•h/mL)', 'AUCinf (μM*h)', 'AUCinf (μgh/ml)', 'AUCinf [ng/mL×h]', 'AUCinf d·µg/mL', 'AUCinf μg*day/ml', 'AUCinf μg/ml × h mean±SD', 'AUCinf μg·h/L', 'AUCinf^^(min*nmol/ml or g)', 'AUCiv (ng∙h/mL)', 'AUClast', 'AUClast (h.ng/mL)', 'AUClast (h·ng·mL−1)', 'AUClast (h×ng/mL)', 'AUClast (hμg/mL)', 'AUClast (ng h/mL)', 'AUClast (ng*hr/mL)', 'AUClast (ngh/mL)', 'AUClast (μgh/ml)', 'AUClast d·µg/mL', 'AUClast e (hμg/L)', 'AUClast g (ng h/mL)', 'AUClast µg · h/L', 'AUClast μg·h/L', 'AUClast/mIUhL−1', 'AUClast^^(min*nmol/ml or g)', 'AUCmilk/AUCplasma', 'AUCn (μMh/mpk)', 'AUCpo (ng∙h/mL)', 'AUCratio (sirolimus:temsirolimus)', 'AUCsum (ng · h/mL)', 'AUCt (hng/mL)', 'AUCub/AUCup', 'AUCτ ng·h/mL', 'AUCτ μg/ml × h mean±SD', 'AUC∞ (hng/mL)', 'AUC∞ (ngh/ml)', 'AUC∞ (µg·h/L)', 'AUC∞ 0 (μmol-min/l)', 'AUC∞(mg⋅h/L)', 'Area under the concentration–time curve (AUC0–∞)', 'Measured AUCR', 'Molar ratio of AMP/METH AUC d', 'N AUC po μMh/(mg/kg)', 'Oral AUC (μMh)', 'Plasma d4-13-HODE AUC', 'RAUC0–∞CD10899/volasertib', 'R_AUC(t/∞)', 'Total area: slow+fast', 'c AUC (h*μg/mL)', 'c AUC0-∞,min.μg/mL', 'fAUCdermal /fAUCplasma'],
                # "f2": ['(AUC0→∞)', 'AU C 0 – 2 4 ss (μgh/mL) a', 'AUC', 'AUC (0-t) (μg/L*h)', 'AUC (0-∞) (μg min mL−1)', 'AUC (0-∞) (μg/L*h)', 'AUC (0–t) (ngmin/mL)', 'AUC (0–∞) (ngmin/mL)', 'AUC (day·μg/mL)', 'AUC (h.mg/L)', 'AUC (hng/mL)', 'AUC (h·μg/mL)', 'AUC (h×ng/mL)', 'AUC (mg h/l)', 'AUC (mg/Lh)', 'AUC (min mg/mL)', 'AUC (min µg/mL)', 'AUC (min.ng/mL)', 'AUC (ng minute mL–1)', 'AUC (ng · h/mL)', 'AUC (ng*h/mL)', 'AUC (ngh/mL)', 'AUC (ng·h/mL)', 'AUC (ng⋅h/mL)', 'AUC (pg/mL · h)', 'AUC (pg⋅h/mL)', 'AUC (µg·h/L)', 'AUC (μg h−1 ml−1)', 'AUC (μg min per ml)', 'AUC (μg min/mL)', 'AUC (μg min/ml)', 'AUC (μg · min/mL)', 'AUC (μg · min/mL) b', 'AUC (μg/ml h)', 'AUC (μg/ml/hr)', 'AUC (μgh/mL)', 'AUC (μg·min/mL)', 'AUC 0 ∞ (h·μg/mL)', 'AUC 0 ∞/D (h·ng/mL·mg)', 'AUC 0-8h (min·ng/mL)', 'AUC 0-inf (h × mg L−1)', 'AUC 0-t (ng mL−1 h)', 'AUC 0-t (ng/g*h)', 'AUC 0-t (ng/mL* h)', 'AUC 0-t,ng/mL*h', 'AUC 0-∞ (min·ng/mL)', 'AUC 0-∞ (ng/mL* h)', 'AUC 0-∞ (ug/mL.h)', 'AUC 0-∞ (μg h/mL)', 'AUC 0-∞ ng/h/mL', 'AUC 0-∞,ng/mL*h', 'AUC 0−∞,ng/mL∙h', 'AUC AMP (ng/min/ml)', 'AUC METH (ng/min/ml) (normalized for dose)', 'AUC [pg/mL × h]', 'AUC b (μg·min/mL)', 'AUC c (μg·min/mL)', 'AUC ng·h/mL', 'AUC ∞ (μMh)', 'AUC% extrap (%)', 'AUC(0-24)', 'AUC(0-INF)(h×ng/mL)', 'AUC(0-t)', 'AUC(0-t) (h*ng/mL)', 'AUC(0-t) (ng·h/mL)', 'AUC(0-t) (ug.h/L)', 'AUC(0-t)(mg/L*h)', 'AUC(0-∞)', 'AUC(0-∞) (h*ng/mL)', 'AUC(0-∞) (ug.h/L)', 'AUC(0–24) (μg*h/mL)', 'AUC(0–last) (μgh/mL)', 'AUC(0–t)', 'AUC(0–∞)', 'AUC(0–∞) (μgh/mL)', 'AUC(0→t) (μg/mLmin)', 'AUC(0→t) (μg/mL•min)', 'AUC(0→∞) (μg/mLmin)', 'AUC(0→∞) (μg/mL•min)', 'AUC(0− t) (μg/lh)', 'AUC(0−t) (ngminmL−1)', 'AUC,(min μ g)/mL', 'AUC,(μg/mL)·h', 'AUC-inf (pg h mL−1)', 'AUC-last (pg h mL−1)', 'AUC/dose (dose normalized) (h×ng/mL)/(mg/kg)', 'AUC0 → ∞ (mg min mL−1)', 'AUC0 →∞ (μg h/L)', 'AUC0 ∞ (% activity·day)', 'AUC0 ∞ (Bq·µL−1·day)', 'AUC0-12 (ng·h/mL)', 'AUC0-12h (h·nM)', 'AUC0-168 h (h·µg/l)', 'AUC0-168h (ug/mL•h)', 'AUC0-24 (mg/L*h)', 'AUC0-24 ng • h/mL', 'AUC0-24(μg/L × h)', 'AUC0-24h (h*μM)', 'AUC0-24h (h·nM)', 'AUC0-24ss b (ng·h/mL)', 'AUC0-5d (μg/mL.h)', 'AUC0-8 (ng·h/mL)', 'AUC0-INF (μg·h/mL)', 'AUC0-inf (%ID/g *h)', 'AUC0-inf (mg•h/L)', 'AUC0-inf (ng eq·h−1·mL−1)', 'AUC0-inf (ng*hour/ml)', 'AUC0-inf (ng.h/mL)', 'AUC0-inf (ng.min/ml)', 'AUC0-inf (ng·h−1·mL−1)', 'AUC0-inf (μg*h/mL)', 'AUC0-last (h*ng/mL)', 'AUC0-last (ng*hour/ml)', 'AUC0-last (ng.h/mL)', 'AUC0-t (%ID/g *h)', 'AUC0-t (h*ng/mL)', 'AUC0-t (h*μg/mL)', 'AUC0-t (h·ng/mL)', 'AUC0-t (mg h/l)', 'AUC0-t (ng*h/mL)', 'AUC0-• (min*ng/mL)', 'AUC0-∞ (h*ng/mL)', 'AUC0-∞ (h*pg/mL)', 'AUC0-∞ (h*μg/mL)', 'AUC0-∞ (h·ng/mL)', 'AUC0-∞ (ng.h/mL)', 'AUC0-∞ (ng·h/mL)', 'AUC0-∞ (µg/ml*h)', 'AUC0-∞ (μg/mL.h)', 'AUC0-∞ (μg·min/mL)', 'AUC0-∞ ng • h/mL', 'AUC0-∞(μg/L × h)', 'AUC0‒t (mg·h/L)', 'AUC0‒∞ (mg·h/L)', 'AUC0– t', 'AUC0– t (μg/mL h)', 'AUC0–12 (h∗μg/mL)', 'AUC0–12, oral (mg · h/L)‡', 'AUC0–12h μg·h/L', 'AUC0–12hss μg·h/L', 'AUC0–14d (ng∗h/ml)', 'AUC0–168 h (h mg/l)', 'AUC0–1d (ng∗h/ml)', 'AUC0–2 h (μg min/ml)', 'AUC0–21d (ng∗h/ml)', 'AUC0–24', 'AUC0–24 (h × ng/mL)', 'AUC0–24 (mg L−1 h)', 'AUC0–24 (mg·h/L)', 'AUC0–24 (mg•h/L)', 'AUC0–24 (ng·h/ml)', 'AUC0–24 (pg.h/mL)', 'AUC0–240 (mg/lmin)', 'AUC0–240 min (μg·min/mL)', 'AUC0–24h [ngh/mL]', 'AUC0–48 h (ngh/mL)', 'AUC0–480 (mg/lmin)', 'AUC0–504norm (ng·h/ml/mg)', 'AUC0–7d (ng∗h/ml)', 'AUC0–8, oral (mg · h/L)', 'AUC0–8h (μg·min/mL)', 'AUC0–inf (h.ng/mL)', 'AUC0–inf (ng·h/ml)', 'AUC0–inf (μmol/L·h)', 'AUC0–inf ng·h/mL', 'AUC0–last (ng h/mL)', 'AUC0–last (ngh/ml)', 'AUC0–last (μmol/L·h)', 'AUC0–t', 'AUC0–t ((μgh)/ml)', 'AUC0–t (h·nM)', 'AUC0–t (h·ng/mL)', 'AUC0–t (mg/L·h)', 'AUC0–t mean (SD) h·µg/mL', 'AUC0–t µg·h/mL', 'AUC0–τ (ng·h/ml) a', 'AUC0–∝ (μg/mLh)', 'AUC0–∝/D (kg/h/L)', 'AUC0–∞', 'AUC0–∞ ((ngh)/ml)', 'AUC0–∞ ((μgh)/ml)', 'AUC0–∞ (hour ng mL−1)', 'AUC0–∞ (h·nM)', 'AUC0–∞ (h∗ng/mL)', 'AUC0–∞ (mg h/L)', 'AUC0–∞ (mg/lmin)', 'AUC0–∞ (mghL −1)', 'AUC0–∞ (ng h/mL)', 'AUC0–∞ (ng/h/ml)', 'AUC0–∞ (ng/mL h)', 'AUC0–∞ (ngh/mL)', 'AUC0–∞ (ng·h/mL)', 'AUC0–∞ (ng∗h/ml)', 'AUC0–∞ (pg.h/mL)', 'AUC0–∞ (μg min/ml)', 'AUC0–∞ (μg.h/L)', 'AUC0–∞ (μg/mL h)', 'AUC0–∞ (μgh−1/mL)', 'AUC0–∞ (μg∗h/mL)', 'AUC0–∞ [ngh/mL]', 'AUC0–∞ [ng·h/mL]', 'AUC0–∞ a (μgmin/ml)', 'AUC0–∞ mean (SD) (h·µg/mL)', 'AUC0–∞ mean (SD) h·µg/mL', 'AUC0–∞ µg · h/L', 'AUC0–∞ µg·h/mL', 'AUC0→24 (mgh/L) a', 'AUC0→48 (h∙ng/mL)', 'AUC0→Inf (mgh/L)', 'AUC0→inf (h∙ng/mL)', 'AUC0→t (mgh/L) b', 'AUC0→t (μg h/L)', 'AUC0→t(minng/ml)', 'AUC0→∞ (plasma h·ng/mL; skin hng/g)', 'AUC0→∞ (μg minutes mL–1)', 'AUC0→∞ (μg*minutes mL–1)', 'AUC0→∞ (μg/h ml)', 'AUC0→∞(minng/ml)', 'AUC0→∞(μg·min/ml)', 'AUC0− t (ng/mLmin)', 'AUC0− τ (μgh/mL)', 'AUC0−8 (μghg−1)', 'AUC0−t', 'AUC0−t/μghmL−1', 'AUC0−∞', 'AUC0−∞ (hμgmL−1)', 'AUC0−∞ (ng/mLmin)', 'AUC0−∞ (ngh/mL)', 'AUC0−∞ (μg ∗ h/mL)', 'AUC0−∞ (μghmL−1)', 'AUC0−∞ (μg·h/mL)', 'AUC0−∞ mean (SD) h·µg/mL', 'AUC0−∞ per dose h·µg/mL/mg', 'AUC12 (µg·h/mL)', 'AUC420min (μg.h/l)', 'AUCCSF:AUCSerum [%]', 'AUCExtr', 'AUCExtra (%)', 'AUCINF (predicted) a (hμg/L)', 'AUCINF_obs(h·ng·mL−1)', 'AUCInf', 'AUC^number^(ng h/ml)', 'AUCa (µg · min/mL)', 'AUCinf ( n g ∙ h ∙ m L - 1 )', 'AUCinf (h·µg/l)', 'AUCinf (ng mL−1 hour)', 'AUCinf (μg h/mL)', 'AUCinf (μg.h/l)', 'AUCinf (μgh/ml)', 'AUCinf μg*day/ml', 'AUCinf μg/ml × h mean±SD', 'AUCinf μg·h/L', 'AUClast', 'AUClast (h*ng/mL)', 'AUClast (h·ng·mL−1)', 'AUClast (h×ng/mL)', 'AUClast (ng h/mL)', 'AUClast (μg.h/l)', 'AUClast (μgh/mL)', 'AUClast (μgh/ml)', 'AUClast e (hμg/L)', 'AUClast g (ng h/mL)', 'AUClast µg · h/L', 'AUClast μg·h/L', 'AUClast/mIUhL−1', 'AUCmilk/AUCplasma', 'AUCratio (sirolimus:temsirolimus)', 'AUCsum (h·µg/l)', 'AUCsum (ng · h/mL)', 'AUCτ ng·h/mL', 'AUCτ μg/ml × h mean±SD', 'AUC∞ (µg·h/L)', 'AUC∞(mg⋅h/L)',  'Molar ratio of AMP/METH AUC d', 'Oral AUC (μMh)', 'Plasma d4-13-HODE AUC', 'RAUC0–∞CD10899/volasertib', 'R_AUC(t/∞)', 'Vitreal AUC0-inf (nM . day)', 'a AUC (ngh/mL)'],
                # "f3": ['(AUC0→∞)', 'AU C 0 – 2 4 ss (μgh/mL) a', 'AUC', 'AUC (0-t) (μg/L*h)', 'AUC (0-∞) (μg min mL−1)', 'AUC (0-∞) (μg/L*h)', 'AUC (0–t) (ngmin/mL)', 'AUC (0–∞) (ngmin/mL)', 'AUC (day·μg/mL)', 'AUC (h.mg/L)', 'AUC (hng/mL)', 'AUC (h·μg/mL)', 'AUC (h×ng/mL)', 'AUC (mg h/l)', 'AUC (mg/Lh)', 'AUC (min mg/mL)', 'AUC (min µg/mL)', 'AUC (min.ng/mL)', 'AUC (ng minute mL–1)', 'AUC (ng · h/mL)', 'AUC (ng*h/mL)', 'AUC (ngh/mL)', 'AUC (ng·h/mL)', 'AUC (ng⋅h/mL)', 'AUC (pg/mL · h)', 'AUC (pg⋅h/mL)', 'AUC (µg·h/L)', 'AUC (μg h−1 ml−1)', 'AUC (μg min per ml)', 'AUC (μg min/mL)', 'AUC (μg min/ml)', 'AUC (μg · min/mL)', 'AUC (μg · min/mL) b', 'AUC (μg/ml h)', 'AUC (μg/ml/hr)', 'AUC (μgh/mL)', 'AUC (μg·min/mL)', 'AUC 0 ∞ (h·μg/mL)', 'AUC 0 ∞/D (h·ng/mL·mg)', 'AUC 0-8h (min·ng/mL)', 'AUC 0-inf (h × mg L−1)', 'AUC 0-t (ng mL−1 h)', 'AUC 0-t (ng/g*h)', 'AUC 0-t (ng/mL* h)', 'AUC 0-t,ng/mL*h', 'AUC 0-∞ (min·ng/mL)', 'AUC 0-∞ (ng/mL* h)', 'AUC 0-∞ (ug/mL.h)', 'AUC 0-∞ (μg h/mL)', 'AUC 0-∞ ng/h/mL', 'AUC 0-∞,ng/mL*h', 'AUC 0−∞,ng/mL∙h', 'AUC AMP (ng/min/ml)', 'AUC METH (ng/min/ml) (normalized for dose)', 'AUC [pg/mL × h]', 'AUC b (μg·min/mL)', 'AUC c (μg·min/mL)', 'AUC ng·h/mL', 'AUC ∞ (μMh)', 'AUC% extrap (%)', 'AUC(0-24)', 'AUC(0-INF)(h×ng/mL)', 'AUC(0-t)', 'AUC(0-t) (h*ng/mL)', 'AUC(0-t) (ng·h/mL)', 'AUC(0-t) (ug.h/L)', 'AUC(0-t)(mg/L*h)', 'AUC(0-∞)', 'AUC(0-∞) (h*ng/mL)', 'AUC(0-∞) (ug.h/L)', 'AUC(0–24) (μg*h/mL)', 'AUC(0–last) (μgh/mL)', 'AUC(0–t)', 'AUC(0–∞)', 'AUC(0–∞) (μgh/mL)', 'AUC(0→t) (μg/mLmin)', 'AUC(0→t) (μg/mL•min)', 'AUC(0→∞) (μg/mLmin)', 'AUC(0→∞) (μg/mL•min)', 'AUC(0− t) (μg/lh)', 'AUC(0−t) (ngminmL−1)', 'AUC,(min μ g)/mL', 'AUC,(μg/mL)·h', 'AUC-inf (pg h mL−1)', 'AUC-last (pg h mL−1)', 'AUC/dose (dose normalized) (h×ng/mL)/(mg/kg)', 'AUC0 → ∞ (mg min mL−1)', 'AUC0 →∞ (μg h/L)', 'AUC0 ∞ (% activity·day)', 'AUC0 ∞ (Bq·µL−1·day)', 'AUC0-12 (ng·h/mL)', 'AUC0-12h (h·nM)', 'AUC0-168 h (h·µg/l)', 'AUC0-168h (ug/mL•h)', 'AUC0-24 (mg/L*h)', 'AUC0-24 ng • h/mL', 'AUC0-24(μg/L × h)', 'AUC0-24h (h*μM)', 'AUC0-24h (h·nM)', 'AUC0-24ss b (ng·h/mL)', 'AUC0-5d (μg/mL.h)', 'AUC0-8 (ng·h/mL)', 'AUC0-INF (μg·h/mL)', 'AUC0-inf (%ID/g *h)', 'AUC0-inf (mg•h/L)', 'AUC0-inf (ng eq·h−1·mL−1)', 'AUC0-inf (ng*hour/ml)', 'AUC0-inf (ng.h/mL)', 'AUC0-inf (ng.min/ml)', 'AUC0-inf (ng·h−1·mL−1)', 'AUC0-inf (μg*h/mL)', 'AUC0-last (h*ng/mL)', 'AUC0-last (ng*hour/ml)', 'AUC0-last (ng.h/mL)', 'AUC0-t (%ID/g *h)', 'AUC0-t (h*ng/mL)', 'AUC0-t (h*μg/mL)', 'AUC0-t (h·ng/mL)', 'AUC0-t (mg h/l)', 'AUC0-t (ng*h/mL)', 'AUC0-• (min*ng/mL)', 'AUC0-∞ (h*ng/mL)', 'AUC0-∞ (h*pg/mL)', 'AUC0-∞ (h*μg/mL)', 'AUC0-∞ (h·ng/mL)', 'AUC0-∞ (ng.h/mL)', 'AUC0-∞ (ng·h/mL)', 'AUC0-∞ (µg/ml*h)', 'AUC0-∞ (μg/mL.h)', 'AUC0-∞ (μg·min/mL)', 'AUC0-∞ ng • h/mL', 'AUC0-∞(μg/L × h)', 'AUC0‒t (mg·h/L)', 'AUC0‒∞ (mg·h/L)', 'AUC0– t', 'AUC0– t (μg/mL h)', 'AUC0–12 (h∗μg/mL)', 'AUC0–12, oral (mg · h/L)‡', 'AUC0–12h μg·h/L', 'AUC0–12hss μg·h/L', 'AUC0–14d (ng∗h/ml)', 'AUC0–168 h (h mg/l)', 'AUC0–1d (ng∗h/ml)', 'AUC0–2 h (μg min/ml)', 'AUC0–21d (ng∗h/ml)', 'AUC0–24', 'AUC0–24 (h × ng/mL)', 'AUC0–24 (mg L−1 h)', 'AUC0–24 (mg·h/L)', 'AUC0–24 (mg•h/L)', 'AUC0–24 (ng·h/ml)', 'AUC0–24 (pg.h/mL)', 'AUC0–240 (mg/lmin)', 'AUC0–240 min (μg·min/mL)', 'AUC0–24h [ngh/mL]', 'AUC0–48 h (ngh/mL)', 'AUC0–480 (mg/lmin)', 'AUC0–504norm (ng·h/ml/mg)', 'AUC0–7d (ng∗h/ml)', 'AUC0–8, oral (mg · h/L)', 'AUC0–8h (μg·min/mL)', 'AUC0–inf (h.ng/mL)', 'AUC0–inf (ng·h/ml)', 'AUC0–inf (μmol/L·h)', 'AUC0–inf ng·h/mL', 'AUC0–last (ng h/mL)', 'AUC0–last (ngh/ml)', 'AUC0–last (μmol/L·h)', 'AUC0–t', 'AUC0–t ((μgh)/ml)', 'AUC0–t (h·nM)', 'AUC0–t (h·ng/mL)', 'AUC0–t (mg/L·h)', 'AUC0–t mean (SD) h·µg/mL', 'AUC0–t µg·h/mL', 'AUC0–τ (ng·h/ml) a', 'AUC0–∝ (μg/mLh)', 'AUC0–∝/D (kg/h/L)', 'AUC0–∞', 'AUC0–∞ ((ngh)/ml)', 'AUC0–∞ ((μgh)/ml)', 'AUC0–∞ (hour ng mL−1)', 'AUC0–∞ (h·nM)', 'AUC0–∞ (h∗ng/mL)', 'AUC0–∞ (mg h/L)', 'AUC0–∞ (mg/lmin)', 'AUC0–∞ (mghL −1)', 'AUC0–∞ (ng h/mL)', 'AUC0–∞ (ng/h/ml)', 'AUC0–∞ (ng/mL h)', 'AUC0–∞ (ngh/mL)', 'AUC0–∞ (ng·h/mL)', 'AUC0–∞ (ng∗h/ml)', 'AUC0–∞ (pg.h/mL)', 'AUC0–∞ (μg min/ml)', 'AUC0–∞ (μg.h/L)', 'AUC0–∞ (μg/mL h)', 'AUC0–∞ (μgh−1/mL)', 'AUC0–∞ (μg∗h/mL)', 'AUC0–∞ [ngh/mL]', 'AUC0–∞ [ng·h/mL]', 'AUC0–∞ a (μgmin/ml)', 'AUC0–∞ mean (SD) (h·µg/mL)', 'AUC0–∞ mean (SD) h·µg/mL', 'AUC0–∞ µg · h/L', 'AUC0–∞ µg·h/mL', 'AUC0→24 (mgh/L) a', 'AUC0→48 (h∙ng/mL)', 'AUC0→Inf (mgh/L)', 'AUC0→inf (h∙ng/mL)', 'AUC0→t (mgh/L) b', 'AUC0→t (μg h/L)', 'AUC0→t(minng/ml)', 'AUC0→∞ (plasma h·ng/mL; skin hng/g)', 'AUC0→∞ (μg minutes mL–1)', 'AUC0→∞ (μg*minutes mL–1)', 'AUC0→∞ (μg/h ml)', 'AUC0→∞(minng/ml)', 'AUC0→∞(μg·min/ml)', 'AUC0− t (ng/mLmin)', 'AUC0− τ (μgh/mL)', 'AUC0−8 (μghg−1)', 'AUC0−t', 'AUC0−t/μghmL−1', 'AUC0−∞', 'AUC0−∞ (hμgmL−1)', 'AUC0−∞ (ng/mLmin)', 'AUC0−∞ (ngh/mL)', 'AUC0−∞ (μg ∗ h/mL)', 'AUC0−∞ (μghmL−1)', 'AUC0−∞ (μg·h/mL)', 'AUC0−∞ mean (SD) h·µg/mL', 'AUC0−∞ per dose h·µg/mL/mg', 'AUC12 (µg·h/mL)', 'AUC420min (μg.h/l)', 'AUCCSF:AUCSerum [%]', 'AUCExtr', 'AUCExtra (%)', 'AUCINF (predicted) a (hμg/L)', 'AUCINF_obs(h·ng·mL−1)', 'AUCInf', 'AUC^number^(ng h/ml)', 'AUCa (µg · min/mL)', 'AUCinf ( n g ∙ h ∙ m L - 1 )', 'AUCinf (h·µg/l)', 'AUCinf (ng mL−1 hour)', 'AUCinf (μg h/mL)', 'AUCinf (μg.h/l)', 'AUCinf (μgh/ml)', 'AUCinf μg*day/ml', 'AUCinf μg/ml × h mean±SD', 'AUCinf μg·h/L', 'AUClast', 'AUClast (h*ng/mL)', 'AUClast (h·ng·mL−1)', 'AUClast (h×ng/mL)', 'AUClast (ng h/mL)', 'AUClast (μg.h/l)', 'AUClast (μgh/mL)', 'AUClast (μgh/ml)', 'AUClast e (hμg/L)', 'AUClast g (ng h/mL)', 'AUClast µg · h/L', 'AUClast μg·h/L', 'AUClast/mIUhL−1', 'AUCmilk/AUCplasma', 'AUCratio (sirolimus:temsirolimus)', 'AUCsum (h·µg/l)', 'AUCsum (ng · h/mL)', 'AUCτ ng·h/mL', 'AUCτ μg/ml × h mean±SD', 'AUC∞ (µg·h/L)', 'AUC∞(mg⋅h/L)',  'Molar ratio of AMP/METH AUC d', 'Oral AUC (μMh)', 'Plasma d4-13-HODE AUC', 'RAUC0–∞CD10899/volasertib', 'R_AUC(t/∞)', 'Vitreal AUC0-inf (nM . day)', 'a AUC (ngh/mL)'],
            }
        elif target_pk_parameter == 'CL':
            unique_llm_parameters_dict = {
                "f1": ['Absolute antipyrine clearance (ml/min)', 'Antipyrine clearance (ml/min/kg)', 'CL', 'CL ( L ∙ k g - 1 ∙ h - 1 )', 'CL ((mL/kg)/min)', 'CL ((mL/min)/kg)', 'CL (L h/kg)', 'CL (L/h)', 'CL (L/h/Kg)', 'CL (L/h/kg)', 'CL (L/min/kg)', 'CL (L/minkg)', 'CL (l kg−1 h−1)', 'CL (l/h)', 'CL (l/h/kg)', 'CL (l/hour/kg)', 'CL (lh− kg−1)', 'CL (litre h−1)', 'CL (mL minute–1 kg–1)', 'CL (mL/d)', 'CL (mL/d/kg)', 'CL (mL/h)', 'CL (mL/h.kg)', 'CL (mL/h/kg)', 'CL (mL/hr/kg)', 'CL (mL/kg/min)', 'CL (mL/kg·min)', 'CL (mL/min)', 'CL (mL/min/ kg)', 'CL (mL/min/Kg)', 'CL (mL/min/kg)', 'CL (mL/min/kg) i.v.', 'CL (mL/minkg)', 'CL (mLmin−1)', 'CL (ml/h/kg)', 'CL (ml/min per kg)', 'CL (ml/min)', 'CL (ml/min/kg)', 'CL (n = 11)', 'CL B (L/kg h)', 'CL D 2', 'CL D 3', 'CL L/h', 'CL NR', 'CL NR (mL/min/ kg)', 'CL NR (mL/min/kg)', 'CL R', 'CL R (mL/min/kg)', 'CL bile (mL/min)', 'CL c (mL/min/kg)', 'CL h (mL/min)', 'CL l/h mean±SD', 'CL ml/day/kg', 'CL ml/kg/h', 'CL p', 'CL renal (μl/(minkg))', 'CL s', 'CL s (ml/min per kg)', 'CL t (l/h/kg)', 'CL tot /F', 'CL total (μl/(minkg))', 'CL(L/h)', 'CL/BW (L/h·kg)', 'CL/F', 'CL/F (L h−1)', 'CL/F (L/h)', 'CL/F (L/h/kg)', 'CL/F (L·h−1·kg−1)', 'CL/F (l/h/kg)', 'CL/F (mL minute–1 kg–1)', 'CL/F (mL minute−1 kg−1)', 'CL/F (mL/Kg/h)', 'CL/F (mL/min)', 'CL/F (ml/h/kg)', 'CL/F Gmean (CV%) L/h', 'CL/F L/h', 'CL/F L/kg/h', 'CL/F [L/h]', 'CL/F mL/day', 'CL/F mL/day/kg', 'CL/F mL/min', 'CL/F mean (SD) L/h', 'CL/F mean (SD) L/h ‡', 'CL/F(mL/h)', 'CL/F(ml/h/kg)', 'CL/F_obs', 'CL/fm (L/h)', 'CL/fm [(mg)/(μg/mL)/h] b', 'CL1 (mL/h/kg)', 'CL2', 'CL2 (mL/h/kg)', 'CL2/F (mL minute–1 kg–1)', 'CL3', 'CLCHDF (mL/min)', 'CLCRRT (L/h)', 'CLCVVH (L/h)', 'CLCr(mL/min)', 'CLNR (L/h)', 'CLNR (mL/min/kg)', 'CLNR (ml/min per kg)', 'CLNR (ml/min/kg)', 'CLP (mL min−1 kg−1)', 'CLP (mL/min·kg)', 'CLPMX (L/h)', 'CLR', 'CLR (mL min−1 kg−1)', 'CLR (mL/min/kg)', 'CLR (ml/min per kg)', 'CLR (ml/min/kg)', 'CLT(mL/minkg)', 'CLTOT (L/h)', 'CLTotal (Lh−1)', 'CL_F (L/h/kg)', 'CL_obs', 'CL_obs (mL·h−1·kg−1)', 'CLb (mL/min/kg)', 'CLb (mL/min/kg) b', 'CLd (mL minute–1 kg–1)', 'CLd/F (mL minute–1 kg–1)', 'CLhepatic (L h− 1 kg− 1)', 'CLi a', 'CLint, in vivo ((mL/min)/kg)', 'CLint, in vivo (mL/min/kg)', 'CLint, microsome ((μL/min)/mg proteins)', 'CLint, microsome (µL/min/mg proteins)', 'CLnon-CRRT (L/h)', 'CLnorm (n = 11)', 'CLo (μ L/min)', 'CLoral (ml/h/g)', 'CLp (mL/min/kg)', 'CLp (ml/min/kg)', 'CLp/F (mL/(hkg))', 'CLr (L/h)', 'CLr (ml/min/kg)', 'CLr L/h', 'CLr mean (SD) L/h', 'CLrenal (L h− 1 kg− 1)', 'CLs', 'CLs (L/h)', 'CLss/F (L/h)', 'CLss/F L/h', 'CLt/F (L/h)', 'CLtot', 'CLtot (L h− 1 kg− 1)', 'CLtot (L/h)', 'CLtot (mL/h/kg)', 'CLtot (mL/min)', 'CLtot (mL/min/kg)', 'CLtot/F (L/h/kg)', 'CLtotal (L/h/kg)', 'CLtotal (mL/min/kg)', 'CLz (L/h/kg)', 'CLz/F (L/h/kg)', 'Cl', 'Cl (IV) or Cl/F (PO) (mL/h/kg)', 'Cl (L h−1)', 'Cl (L minute−1)', 'Cl (L/h/Kg)', 'Cl (L/h/kg)', 'Cl (L/hkg)', 'Cl (L/hr/kg)', 'Cl (Lh−1)', 'Cl (lh−1 kg−1)', 'Cl (mL kg–1 minute–1)', 'Cl (mL kg−1 minute−1)', 'Cl (mL minute−1 kg−1)', 'Cl (mL min−1 kg−1)', 'Cl (mL/h)', 'Cl (mL/h/kg)', 'Cl (mL/h/m2)', 'Cl (mL/kgmin)', 'Cl (mL/min)', 'Cl (mL/min/kg)', 'Cl (ml/h/kg)', 'Cl (ml/kg/min)', 'Cl (ml/min/kg)', 'Cl Renal (mL/min)', 'Cl Total (mL/min)', 'Cl [mL/(h kg)]', 'Cl area', 'Cl b (mL/min/kg)', 'Cl iv (L/kgh)', 'Cl l/h/kg', 'Cl r', 'Cl t (L/hkg)', 'Cl t [ml/min/kg]', 'Cl tot', 'Cl total (ml/kg/hr)', 'Cl(L/h/kg)', 'Cl/F', 'Cl/F (1/(hkg))', 'Cl/F (L/h)', 'Cl/F (L/h/kg)', 'Cl/F (PO) or Cl (IV) (L/minkg)', 'Cl/F (ml/min/kg)', 'Cl/F L/h', 'Cl/F L/h/kg', 'Cl/F/weight (weight normalized) (L/h/kg)', 'Cl/F_obs (g M−1 h−1)', 'ClB', 'ClB (L/hkg)', 'ClB (L/h·kg)', 'ClB/F', 'ClNR', 'ClP (L/h/kg) a', 'ClR', 'ClS', 'ClT', 'ClT (mLh−1/kg)', 'ClTotal', 'Cl_F (L/h·kg)', 'Cl_F (mL/min/kg)', 'Cl_F_pred', 'Cl_obs (mL/h.kg)', 'Cl_obs (mL/min/kg)', 'Clc', 'Cld', 'Clearance (L/h)', 'Clearance (L/h/kg)', 'Clearance (l/h)', 'Clearance (mL/min)', 'Clearance (mL/min/kg)', 'Clearance (mg/kg/min)', 'Clearance (ml/h)', 'Clearance (ml/h/kg)', 'Clearance (ml/kg/hr)', 'Clearance (ml/min)', 'Clearance [mL/min/kg]', 'Clearance, mL/min', 'Clint (μl/min/mg protein)', 'Clint(mL/min/kg)', 'Clnr', 'Clp (L h−1 kg−1)', 'Clp (mL/min)', 'Clp (ml/min/kg)', 'Clpl (L/h)', 'Clr', 'Clrapid', 'Cls', 'Clslow', 'Clt (L/hkg)', 'Clt (l/h)', 'Clt/F (L/h)', 'Cltot', 'Cltot (mL/min/kg)', 'Cltot (ml/kg/h)', 'Haemofilter clearance (CLCVVH)', 'Half-Time Serum Clearance', 'Hu Mic CLint', 'Interpatient variability in CL', 'Total clearance (CLtot)', 'Total clearance L/min'],
                # "f2": ['Absolute antipyrine clearance (ml/min)', 'Antipyrine clearance (ml/min/kg)', 'C L/F (L/h/kg)', 'CL', 'CL ( L ∙ k g - 1 ∙ h - 1 )', 'CL (L/h)', 'CL (L/h/Kg)', 'CL (L/h/kg)', 'CL (L/h/m2)', 'CL (L/hkg)', 'CL (L/min/kg)', 'CL (L/minkg)', 'CL (Lmin−1 kg−1)', 'CL (l kg−1 h−1)', 'CL (l/h)', 'CL (l/h/kg)', 'CL (l/hour/kg)', 'CL (l/min/kg)', 'CL (litre h−1)', 'CL (mL h−1 kg−1)', 'CL (mL minute–1 kg–1)', 'CL (mL/(min*kg))', 'CL (mL/d/kg)', 'CL (mL/h)', 'CL (mL/h/kg)', 'CL (mL/min/ kg)', 'CL (mL/min/kg)', 'CL (mL·day−1)', 'CL (ml/kg/h)', 'CL (ml/min per kg)', 'CL (ml/min)', 'CL (ml/min/kg)', 'CL B (L/kg h)', 'CL D 2', 'CL D 3', 'CL L/h', 'CL NR', 'CL NR (mL/min/ kg)', 'CL NR (mL/min/kg)', 'CL R', 'CL R (mL/min/kg)', 'CL c (mL/min/kg)', 'CL l/h mean±SD', 'CL s', 'CL(L/h)', 'CL(L/h/kg)', 'CL(uL/min/mg)', 'CL/F', 'CL/F (L h−1)', 'CL/F (L/h)', 'CL/F (L/h/kg)', 'CL/F (L·h−1·kg−1)', 'CL/F (l/h)', 'CL/F (l/h/kg)', 'CL/F (mL minute–1 kg–1)', 'CL/F (mL/h)', 'CL/F (mL/min)', 'CL/F (ml/h/kg)', 'CL/F (ml/min)', 'CL/F Gmean (CV%) L/h', 'CL/F L/h', 'CL/F [L/h]', 'CL/F mean (SD) L/h', 'CL/F mean (SD) L/h ‡', 'CL/F(mL/h)', 'CL/F_obs', 'CL/fm (L/h)', 'CL/fm [(mg)/(μg/mL)/h] b', 'CL2', 'CL3', 'CLCHDF (mL/min)', 'CLCVVHDF (L/h)', 'CLCr(mL/min)', 'CLNR', 'CLNR (ml/min per kg)', 'CLP (mL min−1 kg−1)', 'CLP (mL/min·kg)', 'CLR', 'CLR (mL min−1 kg−1)', 'CLR (ml/min per kg)', 'CLR (ml/min)', 'CLR (ml/min/m2)', 'CLT (l/kg 24 h)', 'CLT (ml/kg/h)', 'CLT(mL/minkg)', 'CLTotal (Lh−1)', 'CL_obs', 'CL_obs (mL·h−1·kg−1)', 'CLb (mL/min/kg) b', 'CLd (mL minute–1 kg–1)', 'CLd/F (mL minute–1 kg–1)', 'CLi a', 'CLimp (L/h)', 'CLimp/m2 (L/h·m2)', 'CLo (μ L/min)', 'CLoral (ml/h/g)', 'CLother (L/h)', 'CLp (ml/min/kg)', 'CLr (ml/min/kg)', 'CLr L/h', 'CLr mean (SD) L/h', 'CLs', 'CLs (ml kg−1 min−1)', 'CLss/F (L/h)', 'CLss/F L/h', 'CLss/F(L/h)', 'CLtot (mL/min)', 'CLtot (mL/min/kg)', 'CLtot/F (L/h/kg)', 'CLtotal (L/h)', 'CLtotal (L/h/kg)', 'CLtotal (mL/min/kg)', 'CLz (L/h/kg)', 'CLz/F (L/h/kg)', 'Cl', 'Cl (L/h/Kg)', 'Cl (L/h/kg)', 'Cl (Lh−1)', 'Cl (iv) (mL/min/kg)', 'Cl (l/kg/h)', 'Cl (mL kg−1 minute−1)', 'Cl (mL minute−1 kg−1)', 'Cl (mL/h)', 'Cl (mL/h/kg)', 'Cl (mL/h/m2)', 'Cl (mL/kgmin)', 'Cl (mL/min)', 'Cl (mL/min/kg)', 'Cl (ml/kg/min)', 'Cl (ml/min/kg)', 'Cl [mL/min]', 'Cl t (L/hkg)', 'Cl tot', 'Cl tot (L/kg/h)', 'Cl total (ml/kg/hr)', 'Cl/F', 'Cl/F (L/h)', 'Cl/F (L/h/kg)', 'Cl/F (L/m2/h)', 'Cl/F (PO) or Cl (IV) (L/minkg)', 'Cl/F (l/h)', 'Cl/F (mL/h/kg)', 'Cl/F (mg)/(ng/mL)/h', 'Cl/F (ml/min/kg)', 'Cl/F L/h', 'Cl/F L/h/kg', 'Cl/F/weight (weight normalized) (L/h/kg)', 'ClB (L/h·kg)', 'ClB (mL minute–1 kg–1)', 'ClB/F', 'ClNR', 'ClP (L/h/kg) a', 'ClR', 'ClR (ml/(minkg))', 'ClS', 'ClT', 'ClT (Lhkg −1)', 'ClT (mLh−1/kg)', 'ClT/F (ml/(minkg))', 'ClTotal', 'Cl_F (mL/min/kg)', 'Cl_obs (mL/h.kg)', 'Cl_obs (mL/min/kg)', 'Cld', 'Cld (mL/min/kg)', 'Clearance (Clp, mL/min/kg)', 'Clearance (L/h)', 'Clearance (L/h/kg)', 'Clearance (Total) (ml/kg h)', 'Clearance (mL/min)', 'Clearance (ml/h)', 'Clearance (ml/h/kg)', 'Clearance [mL/min/kg]', 'Clnr', 'Clp (L h−1 kg−1)', 'Clp (mL/min/kg)', 'Clpl (L/h)', 'Clr', 'Cls', 'Cls (mL/min/kg)', 'Clt (l/h)', 'Cltot (ml/kg/h)', 'Interpatient variability in CL', 'Intrinsic clearance(µL/min/mg)', 'Renal clearance', 'Systemic clearance (L/h)', 'Total blood clearance', 'Total body clearance CL (ml/h/kg)', 'Total plasma clearance', 'Vitreal clearance (µl/day)'],  
                # "f3": ['Absolute antipyrine clearance (ml/min)', 'Antipyrine clearance (ml/min/kg)', 'C L/F (L/h/kg)', 'CL', 'CL ( L ∙ k g - 1 ∙ h - 1 )', 'CL (L/h)', 'CL (L/h/Kg)', 'CL (L/h/kg)', 'CL (L/h/m2)', 'CL (L/hkg)', 'CL (L/min/kg)', 'CL (L/minkg)', 'CL (Lmin−1 kg−1)', 'CL (l kg−1 h−1)', 'CL (l/h)', 'CL (l/h/kg)', 'CL (l/hour/kg)', 'CL (l/min/kg)', 'CL (litre h−1)', 'CL (mL h−1 kg−1)', 'CL (mL minute–1 kg–1)', 'CL (mL/(min*kg))', 'CL (mL/d/kg)', 'CL (mL/h)', 'CL (mL/h/kg)', 'CL (mL/min/ kg)', 'CL (mL/min/kg)', 'CL (mL·day−1)', 'CL (ml/kg/h)', 'CL (ml/min per kg)', 'CL (ml/min)', 'CL (ml/min/kg)', 'CL B (L/kg h)', 'CL D 2', 'CL D 3', 'CL L/h', 'CL NR', 'CL NR (mL/min/ kg)', 'CL NR (mL/min/kg)', 'CL R', 'CL R (mL/min/kg)', 'CL c (mL/min/kg)', 'CL l/h mean±SD', 'CL s', 'CL(L/h)', 'CL(L/h/kg)', 'CL(uL/min/mg)', 'CL/F', 'CL/F (L h−1)', 'CL/F (L/h)', 'CL/F (L/h/kg)', 'CL/F (L·h−1·kg−1)', 'CL/F (l/h)', 'CL/F (l/h/kg)', 'CL/F (mL minute–1 kg–1)', 'CL/F (mL/h)', 'CL/F (mL/min)', 'CL/F (ml/h/kg)', 'CL/F (ml/min)', 'CL/F Gmean (CV%) L/h', 'CL/F L/h', 'CL/F [L/h]', 'CL/F mean (SD) L/h', 'CL/F mean (SD) L/h ‡', 'CL/F(mL/h)', 'CL/F_obs', 'CL/fm (L/h)', 'CL/fm [(mg)/(μg/mL)/h] b', 'CL2', 'CL3', 'CLCHDF (mL/min)', 'CLCVVHDF (L/h)', 'CLCr(mL/min)', 'CLNR', 'CLNR (ml/min per kg)', 'CLP (mL min−1 kg−1)', 'CLP (mL/min·kg)', 'CLR', 'CLR (mL min−1 kg−1)', 'CLR (ml/min per kg)', 'CLR (ml/min)', 'CLR (ml/min/m2)', 'CLT (l/kg 24 h)', 'CLT (ml/kg/h)', 'CLT(mL/minkg)', 'CLTotal (Lh−1)', 'CL_obs', 'CL_obs (mL·h−1·kg−1)', 'CLb (mL/min/kg) b', 'CLd (mL minute–1 kg–1)', 'CLd/F (mL minute–1 kg–1)', 'CLi a', 'CLimp (L/h)', 'CLimp/m2 (L/h·m2)', 'CLo (μ L/min)', 'CLoral (ml/h/g)', 'CLother (L/h)', 'CLp (ml/min/kg)', 'CLr (ml/min/kg)', 'CLr L/h', 'CLr mean (SD) L/h', 'CLs', 'CLs (ml kg−1 min−1)', 'CLss/F (L/h)', 'CLss/F L/h', 'CLss/F(L/h)', 'CLtot (mL/min)', 'CLtot (mL/min/kg)', 'CLtot/F (L/h/kg)', 'CLtotal (L/h)', 'CLtotal (L/h/kg)', 'CLtotal (mL/min/kg)', 'CLz (L/h/kg)', 'CLz/F (L/h/kg)', 'Cl', 'Cl (L/h/Kg)', 'Cl (L/h/kg)', 'Cl (Lh−1)', 'Cl (iv) (mL/min/kg)', 'Cl (l/kg/h)', 'Cl (mL kg−1 minute−1)', 'Cl (mL minute−1 kg−1)', 'Cl (mL/h)', 'Cl (mL/h/kg)', 'Cl (mL/h/m2)', 'Cl (mL/kgmin)', 'Cl (mL/min)', 'Cl (mL/min/kg)', 'Cl (ml/kg/min)', 'Cl (ml/min/kg)', 'Cl [mL/min]', 'Cl t (L/hkg)', 'Cl tot', 'Cl tot (L/kg/h)', 'Cl total (ml/kg/hr)', 'Cl/F', 'Cl/F (L/h)', 'Cl/F (L/h/kg)', 'Cl/F (L/m2/h)', 'Cl/F (PO) or Cl (IV) (L/minkg)', 'Cl/F (l/h)', 'Cl/F (mL/h/kg)', 'Cl/F (mg)/(ng/mL)/h', 'Cl/F (ml/min/kg)', 'Cl/F L/h', 'Cl/F L/h/kg', 'Cl/F/weight (weight normalized) (L/h/kg)', 'ClB (L/h·kg)', 'ClB (mL minute–1 kg–1)', 'ClB/F', 'ClNR', 'ClP (L/h/kg) a', 'ClR', 'ClR (ml/(minkg))', 'ClS', 'ClT', 'ClT (Lhkg −1)', 'ClT (mLh−1/kg)', 'ClT/F (ml/(minkg))', 'ClTotal', 'Cl_F (mL/min/kg)', 'Cl_obs (mL/h.kg)', 'Cl_obs (mL/min/kg)', 'Cld', 'Cld (mL/min/kg)', 'Clearance (Clp, mL/min/kg)', 'Clearance (L/h)', 'Clearance (L/h/kg)', 'Clearance (Total) (ml/kg h)', 'Clearance (mL/min)', 'Clearance (ml/h)', 'Clearance (ml/h/kg)', 'Clearance [mL/min/kg]', 'Clnr', 'Clp (L h−1 kg−1)', 'Clp (mL/min/kg)', 'Clpl (L/h)', 'Clr', 'Cls', 'Cls (mL/min/kg)', 'Clt (l/h)', 'Cltot (ml/kg/h)', 'Interpatient variability in CL', 'Intrinsic clearance(µL/min/mg)', 'Renal clearance', 'Systemic clearance (L/h)', 'Total blood clearance', 'Total body clearance CL (ml/h/kg)', 'Total plasma clearance', 'Vitreal clearance (µl/day)'],
            }
        elif target_pk_parameter == 'MRT':
            unique_llm_parameters_dict = {
                "f1": ['MRT', 'MRT (area) (h)', 'MRT (d)', 'MRT (h)', 'MRT (hours)', 'MRT (h−1)', 'MRT (min)', 'MRT (minute)', 'MRT (minutes)', 'MRT 0-8h (min)', 'MRT 0-infinity (h)', 'MRT 0-last time (h)', 'MRT 0-∞ (h)', 'MRT 0-∞,h', 'MRT 0–∞', 'MRT 0−∞ (h)', 'MRT [min]', 'MRT h mean±SD', 'MRT(0-t)', 'MRT(0-t) (h)', 'MRT(0-∞)', 'MRT(0-∞) (h)', 'MRT(0–t)', 'MRT(0–t),h', 'MRT(0–∞),h', 'MRT(0−∞)', 'MRT(h)', 'MRT0-24 (h)', 'MRT0-inf (h)', 'MRT0-last (h)', 'MRT0-t (h)', 'MRT0-∞ (h)', 'MRT0-∞,h', 'MRT0–t (h)', 'MRT0–∞ (h)', 'MRT0−inf (h)', 'MRT0−t (h)', 'MRT0−t/h', 'MRT0−∞/h', 'MRTINF (h)', 'MRTlast (h)', 'MRT∞ (hour)'],
                # "f2": ['MRT', 'MRT (d)', 'MRT (h)', 'MRT (hours)', 'MRT (h−1)', 'MRT (min)', 'MRT (minutes)', 'MRT 0-8h (min)', 'MRT 0-∞,h', 'MRT a (h)', 'MRT h mean±SD', 'MRT(0-t) (h)', 'MRT(0-t)(h)', 'MRT(0–t),h', 'MRT(0–∞),h', 'MRT(0−t) (min)', 'MRT(0−∞),h', 'MRT(h)', 'MRT0-24 (h)', 'MRT0-last (h)', 'MRT0-t (h)', 'MRT0-∞ (h)', 'MRT0–t (h)', 'MRT0−t/h', 'MRT0−∞ (h)', 'MRT0−∞,h', 'MRT0−∞/h', 'MRTINF (h)', 'MRTlast (h)'],
                # "f3": ['MRT', 'MRT (d)', 'MRT (h)', 'MRT (hours)', 'MRT (h−1)', 'MRT (min)', 'MRT (minutes)', 'MRT 0-8h (min)', 'MRT 0-∞,h', 'MRT a (h)', 'MRT h mean±SD', 'MRT(0-t) (h)', 'MRT(0-t)(h)', 'MRT(0–t),h', 'MRT(0–∞),h', 'MRT(0−t) (min)', 'MRT(0−∞),h', 'MRT(h)', 'MRT0-24 (h)', 'MRT0-last (h)', 'MRT0-t (h)', 'MRT0-∞ (h)', 'MRT0–t (h)', 'MRT0−t/h', 'MRT0−∞ (h)', 'MRT0−∞,h', 'MRT0−∞/h', 'MRTINF (h)', 'MRTlast (h)'],
            }
        elif target_pk_parameter == 'CMAX':
            unique_llm_parameters_dict = {
                "f1": ['C MAX (μg/ml)', 'C max', 'C max (mg/L)', 'C max (mg/l)', 'C max (nM)', 'C max (ng/mL)', 'C max (ng/ml)', 'C max (nmol/L)', 'C max (plasma ng/mL; skin ng/g)', 'C max (s.c.) μg/ml', 'C max (µg/mL)', 'C max (µgmL−1)', 'C max (μM)', 'C max (μg mL−1)', 'C max (μg/L)', 'C max (μg/l)', 'C max (μg/mL)', 'C max (μg/ml)', 'C max (μgg−1)', 'C max (μgml−1)', 'C max (μmol/L)', 'C max [ng/mL]', 'C max b (μg/mL)', 'C max c (μg/mL)', 'C max calculated (IU anti-FXa/mL)', 'C max measured (IU anti-FXa/mL)', 'C max ng/mL', 'C max(ng/ml)', 'C max,mg/L', 'C max,ng/mL', 'C max,µg mL−1', 'C max,μg/mL', 'C max,μg/ml', 'C max,μgmL−1', 'C max,μgml−1', 'C max,μmol/L', 'C max-milk,mg/L', 'C max/C 0,μg/mL', 'C max/mIUL−1', 'C maxnorm (ng/ml)', 'C max→∞ (ng/mL)', 'CMAX (ng/ml)', 'Cmax', 'Cmax ( n g ∙ m L - 1 )', 'Cmax (PO) or C0 (IV) (ng/mL)', 'Cmax (mg/L)', 'Cmax (mg/l)', 'Cmax (nM)', 'Cmax (nM)a', 'Cmax (ng eq/mL)', 'Cmax (ng mL–1)', 'Cmax (ng/mL)', 'Cmax (ng/mL) maximum plasma concentration', 'Cmax (ng/mL) p.o.', 'Cmax (ng/ml)', 'Cmax (ng·mL−1)', 'Cmax (pg mL−1)', 'Cmax (pg/mL)', 'Cmax (ug/L)', 'Cmax (µg/mL) - Observed', 'Cmax (µg/mL) Non-ischemic Limb*', 'Cmax (µg/ml)', 'Cmax (μM)', 'Cmax (μg /mL)', 'Cmax (μg mL−1)', 'Cmax (μg/ mL)', 'Cmax (μg/L)', 'Cmax (μg/mL', 'Cmax (μg/mL or g)', 'Cmax (μg/mL)', 'Cmax (μg/ml)', 'Cmax (μgmL−1)', 'Cmax (μgml−1)', 'Cmax /dose,g/mL', 'Cmax [ng/mL]', 'Cmax [ng/ml]', 'Cmax a (μM)', 'Cmax c (mM)', 'Cmax c (μM)', 'Cmax mean (SD) µg/mL', 'Cmax ng/mL', 'Cmax ng/ml', 'Cmax per dose µg/mL/mg', 'Cmax pg/mL', 'Cmax µg/L', 'Cmax µg/mL', 'Cmax μg/L', 'Cmax μg/mL', 'Cmax μg/ml', 'Cmax μg/ml mean±SD', 'Cmax(PIP)/C0(DOX) (ng/ml)', 'Cmax(ng/ml)', 'Cmax(µg/mL)', 'Cmax(μg/L)', 'Cmax, pg/mL', 'Cmax,ng/mL', 'Cmax,μg/mL', 'Cmax,μg/ml', 'Cmax/C0 (ng/mL)', 'Cmax/D (PO) or Co/D (IV) (ng/mL/μg/kg)', 'Cmax/Dose (μM/mg/kg)', 'Cmax/dose (dose normalized) (ng/mL)/(mg/kg)', 'Cmax/dose (μg/mL/mg)', 'Cmax/μgmL−1', 'Cmax1 (nmol/L)', 'Cmaxss μg/L', 'Day 1 C max (μg/mL)', 'Dose Normalized Cmax (ng/mL/mg)', 'Maximum concentration (anti-D titer)', 'N C max po μM/(mg/kg)', 'No.^PK profiles a (po, 20mg/kg)^No.^C max b', 'Peak concentration after', 'Plasma Concentration (g/dL)', 'RacCmax', 'SS C max (μg/mL)'],
                # "f2": ['AMP C max (ng/ml)', 'C max', 'C max (mg/L)', 'C max (nM)', 'C max (ng ml−1)', 'C max (ng/mL)', 'C max (ng/ml)', 'C max (pg/mL)', 'C max (plasma ng/mL; skin ng/g)', 'C max (µg/l)', 'C max (µg/mL)', 'C max (µg/ml)', 'C max (μM)', 'C max (μg mL−1)', 'C max (μg/L)', 'C max (μg/l)', 'C max (μg/mL)', 'C max (μg/ml)', 'C max (μgg−1)', 'C max (μmol/L)', 'C max [ng/mL]', 'C max a (µg/mL)', 'C max c (μg/mL)', 'C max estimated(mg/l)', 'C max ng/mL', 'C max,ng/mL', 'C max,μg/L', 'C max,μg/mL', 'C max,μg/ml', 'C max,μmol/L', 'C max-milk', 'C max/mIUL−1', 'C maxnorm (ng/ml)', 'C max→∞ (ng/mL)', 'CMAX (ng/ml)', 'Cmax', 'Cmax ( n g ∙ m L - 1 )', 'Cmax (%ID/g)', 'Cmax (Bq·µL−1)', 'Cmax (mg mL−1)', 'Cmax (mg/L)', 'Cmax (mg/l)', 'Cmax (nM)a', 'Cmax (ng eq/mL)', 'Cmax (ng mL–1)', 'Cmax (ng mL−1)', 'Cmax (ng/g)', 'Cmax (ng/mL)', 'Cmax (ng/mL) maximum plasma concentration', 'Cmax (ng·mL−1)', 'Cmax (pg mL−1)', 'Cmax (pg/mL)', 'Cmax (ug/L)', 'Cmax (ug/mL)', 'Cmax (μg mL–1)', 'Cmax (μg/L)', 'Cmax (μg/mL)', 'Cmax (μgmL−1)', 'Cmax (μmol L−1)', 'Cmax /dose,g/mL', 'Cmax 1st dose (μg/mL)', 'Cmax [ng/mL]', 'Cmax mean (SD) µg/mL', 'Cmax ng/mL', 'Cmax ng/ml', 'Cmax per dose µg/mL/mg', 'Cmax µg/L', 'Cmax µg/mL', 'Cmax μg/L', 'Cmax μg/ml', 'Cmax μg/ml mean±SD', 'Cmax(PIP)/C0(DOX) (ng/ml)', 'Cmax(mg/L)', 'Cmax(ng/mL)', 'Cmax(p·o·)/C0(i.v.) (ng/ml)', 'Cmax(μg/L)', 'Cmax,ng/mL', 'Cmax,μg/mL', 'Cmax,μg/ml', 'Cmax/C0 (ng/mL)', 'Cmax/D (ng/mL·mg)', 'Cmax/dose (dose normalized) (ng/mL)/(mg/kg)', 'Cmaxss μg/L', 'Day 1 C max (μg/mL)', 'ISF C max (mg/L)', 'Maximum concentration (anti-D titer)', 'Plasma C max (mg/L)', 'SS C max (μg/mL)', 'Vitreal Cmax (nM)', 'c C max (ng/mL)', 'cmax [pg/mL]'],
                # "f3": ['AMP C max (ng/ml)', 'C max', 'C max (mg/L)', 'C max (nM)', 'C max (ng ml−1)', 'C max (ng/mL)', 'C max (ng/ml)', 'C max (pg/mL)', 'C max (plasma ng/mL; skin ng/g)', 'C max (µg/l)', 'C max (µg/mL)', 'C max (µg/ml)', 'C max (μM)', 'C max (μg mL−1)', 'C max (μg/L)', 'C max (μg/l)', 'C max (μg/mL)', 'C max (μg/ml)', 'C max (μgg−1)', 'C max (μmol/L)', 'C max [ng/mL]', 'C max a (µg/mL)', 'C max c (μg/mL)', 'C max estimated(mg/l)', 'C max ng/mL', 'C max,ng/mL', 'C max,μg/L', 'C max,μg/mL', 'C max,μg/ml', 'C max,μmol/L', 'C max-milk', 'C max/mIUL−1', 'C maxnorm (ng/ml)', 'C max→∞ (ng/mL)', 'CMAX (ng/ml)', 'Cmax', 'Cmax ( n g ∙ m L - 1 )', 'Cmax (%ID/g)', 'Cmax (Bq·µL−1)', 'Cmax (mg mL−1)', 'Cmax (mg/L)', 'Cmax (mg/l)', 'Cmax (nM)a', 'Cmax (ng eq/mL)', 'Cmax (ng mL–1)', 'Cmax (ng mL−1)', 'Cmax (ng/g)', 'Cmax (ng/mL)', 'Cmax (ng/mL) maximum plasma concentration', 'Cmax (ng·mL−1)', 'Cmax (pg mL−1)', 'Cmax (pg/mL)', 'Cmax (ug/L)', 'Cmax (ug/mL)', 'Cmax (μg mL–1)', 'Cmax (μg/L)', 'Cmax (μg/mL)', 'Cmax (μgmL−1)', 'Cmax (μmol L−1)', 'Cmax /dose,g/mL', 'Cmax 1st dose (μg/mL)', 'Cmax [ng/mL]', 'Cmax mean (SD) µg/mL', 'Cmax ng/mL', 'Cmax ng/ml', 'Cmax per dose µg/mL/mg', 'Cmax µg/L', 'Cmax µg/mL', 'Cmax μg/L', 'Cmax μg/ml', 'Cmax μg/ml mean±SD', 'Cmax(PIP)/C0(DOX) (ng/ml)', 'Cmax(mg/L)', 'Cmax(ng/mL)', 'Cmax(p·o·)/C0(i.v.) (ng/ml)', 'Cmax(μg/L)', 'Cmax,ng/mL', 'Cmax,μg/mL', 'Cmax,μg/ml', 'Cmax/C0 (ng/mL)', 'Cmax/D (ng/mL·mg)', 'Cmax/dose (dose normalized) (ng/mL)/(mg/kg)', 'Cmaxss μg/L', 'Day 1 C max (μg/mL)', 'ISF C max (mg/L)', 'Maximum concentration (anti-D titer)', 'Plasma C max (mg/L)', 'SS C max (μg/mL)', 'Vitreal Cmax (nM)', 'c C max (ng/mL)', 'cmax [pg/mL]'],
            }    
        elif target_pk_parameter == 'TMAX':
            unique_llm_parameters_dict = {
                "f1": ['Cmpd^Tmax (h)^^^^^','No.^PK profiles a (po, 20mg/kg)^No.^T max b', 'T (max) (h)', 'T MAX (h)', 'T max', 'T max (1/h)', 'T max (h)', 'T max (min)', 'T max b (h)', 'T max h', 'T max po (h)', 'T max,h', 'T max/h', 'T max^number^(h)', 'TMAX (h)', 'Time to reach maximum concentration (days)', 'Time-to-peak levels(h)', 'Tmax', 'Tmax (h)', 'Tmax (h)b', 'Tmax (hours)', 'Tmax (hr)', 'Tmax (mg/L)', 'Tmax (min)', 'Tmax a (hr)', 'Tmax c (hr)', 'Tmax h', 'Tmax h †', 'Tmax hours†', 'Tmax median (range) h', 'Tmax median (range) h §', 'Tmax median (range) ha', 'Tmax min', 'Tmax(h)', 'Tmax, h', 'Tmax,h', 'Tmaxss median (range) h', 'a Tmax (h)', 't max', 't max (h)', 't max (h); Median (range)', 't max (min)', 't max (s.c.) day', 't max(h)', 't max,h', 't max,hr', 't max-milk,h', 't max/h', 'tmax', 'tmax (h)', 'tmax (hour)', 'tmax (hours)', 'tmax (min)', 'tmax (minutes)', 'tmax [h]', 'tmax [min]', 'tmax days median (min max)', 'tmax h median (range)', 'tmax §', 'tmax,h'],
                # "f2": ['AMP T max (min)', 'T max', 'T max (h)', 'T max (min)', 'T max h', 'T max,h', 'T max/h', 'T max^number^(h)', 'TMAX (h)', 'Time to reach maximum concentration (days)', 'Tmax', 'Tmax (day)', 'Tmax (h)', 'Tmax (h)b', 'Tmax (hr)', 'Tmax (mg/L)', 'Tmax (min)', 'Tmax (minutes)', 'Tmax h', 'Tmax h †', 'Tmax median (range) h', 'Tmax median (range) h §', 'Tmax median (range) ha', 'Tmax(h)', 'Tmax(min)', 'Tmax,h', 'Tmaxss median (range) h', 't max (h)', 't max (min)', 't max(h)', 't max,h', 't max,hr', 't max-milk,h', 't max/h', 'tmax (h)', 'tmax (min)', 'tmax (minutes)', 'tmax [h]', 'tmax a (h)', 'tmax h median (range)', 'tmax §,h', 'tmax,h'],
                # "f3": ['AMP T max (min)', 'T max', 'T max (h)', 'T max (min)', 'T max h', 'T max,h', 'T max/h', 'T max^number^(h)', 'TMAX (h)', 'Time to reach maximum concentration (days)', 'Tmax', 'Tmax (day)', 'Tmax (h)', 'Tmax (h)b', 'Tmax (hr)', 'Tmax (mg/L)', 'Tmax (min)', 'Tmax (minutes)', 'Tmax h', 'Tmax h †', 'Tmax median (range) h', 'Tmax median (range) h §', 'Tmax median (range) ha', 'Tmax(h)', 'Tmax(min)', 'Tmax,h', 'Tmaxss median (range) h', 't max (h)', 't max (min)', 't max(h)', 't max,h', 't max,hr', 't max-milk,h', 't max/h', 'tmax (h)', 'tmax (min)', 'tmax (minutes)', 'tmax [h]', 'tmax a (h)', 'tmax h median (range)', 'tmax §,h', 'tmax,h'],
            }
        else:
            print(f"⚠️ Unknown target PK parameter: {target_pk_parameter}")
            continue

        similarity_llm_query_cache = {}
        dataset_array = np.array(dataset_list)

        # Use single fold for now
        test_idx, val_idx = train_test_split(
            np.arange(len(dataset_array)), test_size=0.3, random_state=42, shuffle=True
        )
        splits = [(test_idx, val_idx)]
        total = 1

        fold_scores = []
        total_runtime_param = 0
        total_pipeline2_runtime = 0

        # ---------------------
        # Fold-level evaluation. We did one fold.
        # ---------------------
        for fold_idx, (test_idx, val_idx) in tqdm(enumerate(splits), total=total, leave=False):
            with open(save_log_file_name, "a") as f:
                f.write(
                    f"## {time.strftime('%Y-%m-%d %H:%M:%S')} ## "
                    f"Starting run for fold {fold_idx+1}, "
                    f"threshold: {threshold:.2f}, "
                    f"PK parameter: {target_pk_parameter}, "
                    f"model: {pipeline2_model_name}\n"
                )

            scores_log = []
            fold_datasets = dataset_array[test_idx]

            unique_llm_parameters = unique_llm_parameters_dict[f"f{fold_idx+1}"].copy()
            initial_llm_parameters = unique_llm_parameters.copy()

            overall_start_time = time.time()

            # ---------------------
            # Table-level loop
            # ---------------------
            for dataset in tqdm(fold_datasets, desc=f"📊 Tables Processing for fold {fold_idx+1}...", leave=False):
                try:
                    df_input = dataset["input_table"]
                    ref_df = dataset["labeled_table"]

                    if ref_df.empty:
                        continue

                    # 🔍 Pipeline 1: find PK parameter locations
                    pk_locations, unique_llm_parameters, similarity_llm_query_cache, scores_caches, embedding_cache = find_pk_parameter_locations(
                        df_input,
                        initial_llm_parameters,
                        unique_llm_parameters,
                        similarity_llm_query_cache,
                        scores_caches,
                        embedding_cache,
                        weights,
                        concept_name=target_pk_parameter,
                        threshold=threshold,
                        device=device,
                    )

                    # handle header locations
                    try:
                        if any(row == -1 for row, col in pk_locations):
                            for idx, (row, col) in enumerate(pk_locations):
                                pk_locations[idx] = (col, df_input.columns.tolist()[0])

                            df_input = df_input.T.reset_index()
                            df_input.columns = df_input.iloc[0]
                            df_input = df_input[1:]
                    except Exception as e:
                        print(f"⚠️ Error processing PK locations: {e}")
                        continue

                    # 🔍 Extract rows for structured extraction
                    target_rows = extract_rows(df_input, pk_locations)

                    # 🧠 Pipeline 2: LLM structured extraction
                    pipeline2_start_time = time.time()
                    gen_df_text = generate_extraction_csv(
                        target_rows,
                        target_pk_parameter,
                        final_extraction_config,
                        pipeline2_model_name,
                        dataset["footnote"],
                        dataset["caption"],
                        dataset["article_title"],
                        dataset["article_abstract"],
                    )
                    total_pipeline2_runtime += time.time() - pipeline2_start_time

                    try:
                        gen_df = convert_llm_table2df(gen_df_text)
                    except Exception as e:
                        print(f"⚠️ Error converting LLM table: {e}")
                        continue

                    if gen_df.empty:
                        continue

                    # 🧹 Unification
                    gen_df = unification_tables(gen_df)
                    ref_df = unification_tables(ref_df)

                    # 📏 Evaluation
                    reordered_df, mapping, extra_cols = table_eval.reorder_generated_table_by_header(
                        ref_df.columns, gen_df
                    )
                    row_match_map, extra_rows = table_eval.optimal_row_assignment(ref_df, reordered_df)
                    aligned_gen_df = table_eval.align_generated_table_rows(ref_df, reordered_df, row_match_map)

                    aligned_gen_df.reset_index(drop=True, inplace=True)
                    ref_df.reset_index(drop=True, inplace=True)

                    tp, fp, fn, hc = table_eval.compare_cells(
                        ref_df, aligned_gen_df, extra_columns=extra_cols, extra_rows=extra_rows, threshold=0.8
                    )

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    scores_log.append({
                        "filename": dataset["filename"],
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "hallucinated_cells": hc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    })
                    if len(scores_log) % 30 == 0:
                        # calcuate average scores
                        avg_scores = {
                            'tp': np.mean([score['tp'] for score in scores_log]),
                            'fp': np.mean([score['fp'] for score in scores_log]),
                            'fn': np.mean([score['fn'] for score in scores_log]),
                            'precision': np.mean([score['precision'] for score in scores_log]),
                            'recall': np.mean([score['recall'] for score in scores_log]),
                            'f1': np.mean([score['f1'] for score in scores_log])
                        }
                        print(f"\nAverage Scores for Threshold {threshold:.2f} and fold {fold_idx+1}:")
                        print(f"Total Tables Processed: {len(scores_log)}")
                        print(f"True Positives (TP): {avg_scores['tp']:.2f}")
                        print(f"False Positives (FP): {avg_scores['fp']:.2f}")
                        print(f"False Negatives (FN): {avg_scores['fn']:.2f}")
                        print(f"Precision: {avg_scores['precision']:.2f}")
                        print(f"Recall: {avg_scores['recall']:.2f}")
                        print(f"F1 Score: {avg_scores['f1']:.2f}")
                        print("======" * 50)

                except Exception as e:
                    print(f"💥 Error processing table {dataset['filename']}: {e}")
                    continue

            # ---------------------
            # Fold aggregation
            # ---------------------
            avg_scores_fold_dataset = {
                "tp": np.mean([s["tp"] for s in scores_log]),
                "fp": np.mean([s["fp"] for s in scores_log]),
                "fn": np.mean([s["fn"] for s in scores_log]),
                "hc": np.mean([s["hallucinated_cells"] for s in scores_log]),
                "hc_rate": np.sum([score['hallucinated_cells'] for score in scores_log]) / np.sum([score['fp'] for score in scores_log]) * 100 if np.sum([score['fp'] for score in scores_log]) != 0 else 0,

                "precision": np.mean([s["precision"] for s in scores_log]),
                "recall": np.mean([s["recall"] for s in scores_log]),
                "f1": np.mean([s["f1"] for s in scores_log]),
                "dataset_count": len(scores_log),
                "fold_idx": fold_idx + 1,
                "dataset_logs": scores_log,
            }
            fold_scores.append(avg_scores_fold_dataset)

            # print the final scores for the fold
            print(f"\nAverage Scores for Threshold {threshold:.2f}, Fold {fold_idx+1}:")
            print(f"Total Tables Processed: {avg_scores_fold_dataset['dataset_count']:.2f}")
            print(f"Hallucination Rate: {avg_scores_fold_dataset['hc_rate']:.2f}%\n")
            print(f"Precision: {avg_scores_fold_dataset['precision']:.2f}\n")
            print(f"Recall: {avg_scores_fold_dataset['recall']:.2f}\n")
            print(f"F1 Score: {avg_scores_fold_dataset['f1']:.2f}\n")
            print("======" * 50)
            with open(save_log_file_name, "a") as f:
                f.write(f"\nAverage Scores for Threshold {threshold:.2f}, Fold {fold_idx+1}:\n")
                f.write(f"Total Tables Processed: {avg_scores_fold_dataset['dataset_count']:.2f}\n")
                f.write(f"Hallucination Rate: {avg_scores_fold_dataset['hc_rate']:.2f}%\n")
                f.write(f"Precision: {avg_scores_fold_dataset['precision']:.2f}\n")
                f.write(f"Recall: {avg_scores_fold_dataset['recall']:.2f}\n")
                f.write(f"F1 Score: {avg_scores_fold_dataset['f1']:.2f}\n")
                f.write("======" * 50 + "\n")


        # ---------------------
        # Parameter aggregation
        # ---------------------
        avg_scores_parameter = {
            "tp": np.mean([s["tp"] for s in fold_scores]),
            "fp": np.mean([s["fp"] for s in fold_scores]),
            "fn": np.mean([s["fn"] for s in fold_scores]),
            "hc": np.mean([s["hc"] for s in fold_scores]),
            "hc_rate": np.mean([s["hc_rate"] for s in fold_scores]),

            "precision": np.mean([s["precision"] for s in fold_scores]),
            "recall": np.mean([s["recall"] for s in fold_scores]),
            "f1": np.mean([s["f1"] for s in fold_scores]),
            "avg_dataset_count": np.mean([s["dataset_count"] for s in fold_scores]),
            "folds_logs": fold_scores,
        }

        total_runtime_param += time.time() - overall_start_time
        tables_count_param = int(avg_scores_parameter["avg_dataset_count"])
        avg_scores_parameter["avg_runtime_sec"] = (
            total_runtime_param / tables_count_param if tables_count_param > 0 else 0
        )
        avg_scores_parameter["avg_pipeline2_runtime_sec"] = (
            total_pipeline2_runtime / tables_count_param if tables_count_param > 0 else 0
        )


        print(f"\nFinal Scores for Threshold {threshold:.2f}, PK Parameter {target_pk_parameter}:")
        print("++++++" * 50)

        scores_log_all[str(threshold)][target_pk_parameter] = avg_scores_parameter

        print(f"\nFinal Scores for Threshold {threshold:.2f}, PK Parameter {target_pk_parameter}:")
        print(f"Total Tables Processed: {avg_scores_parameter['avg_dataset_count']:.2f}")
        print(f"Average Runtime per Table: {avg_scores_parameter['avg_runtime_sec']:.2f} s")
        print(f"Average Pipeline 2 Runtime per Table: {avg_scores_parameter['avg_pipeline2_runtime_sec']:.2f} s")
        print(f"True Positives (TP): {avg_scores_parameter['tp']:.2f}")
        print(f"False Positives (FP): {avg_scores_parameter['fp']:.2f}")
        print(f"False Negatives (FN): {avg_scores_parameter['fn']:.2f}")
        print(f"Hallucinated Cells (HC): {avg_scores_parameter['hc']:.2f}")
        print(f"Hallucination Rate: {avg_scores_parameter['hc_rate']:.2f}%")
        print(f"Precision: {avg_scores_parameter['precision']:.2f}")
        print(f"Recall: {avg_scores_parameter['recall']:.2f}")
        print(f"F1 Score: {avg_scores_parameter['f1']:.2f}")
        print("++++++" * 50)


        # Save scores all log to a file
        with open(save_log_file_name, "a") as f:
            f.write(f"Final Scores for Threshold {threshold:.2f}, PK Parameter {target_pk_parameter}:\n")
            f.write(f"Total Tables Processed: {avg_scores_parameter['avg_dataset_count']:.2f}\n")
            f.write(f"Average Runtime per Table: {avg_scores_parameter['avg_runtime_sec']:.2f} seconds\n")
            f.write(f"Average Pipeline 2 Runtime per Table: {avg_scores_parameter['avg_pipeline2_runtime_sec']:.2f} seconds\n")
            f.write(f"True Positives (TP): {avg_scores_parameter['tp']:.2f}\n")
            f.write(f"False Positives (FP): {avg_scores_parameter['fp']:.2f}\n")
            f.write(f"False Negatives (FN): {avg_scores_parameter['fn']:.2f}\n")
            f.write(f"Hallucinated Cells (HC): {avg_scores_parameter['hc']:.2f}\n")
            f.write(f"Hallucination Rate: {avg_scores_parameter['hc_rate']:.2f}%\n")
            f.write(f"Precision: {avg_scores_parameter['precision']:.2f}\n")
            f.write(f"Recall: {avg_scores_parameter['recall']:.2f}\n")
            f.write(f"F1 Score: {avg_scores_parameter['f1']:.2f}\n")
            f.write(f"## {time.strftime('%Y-%m-%d %H:%M:%S')} ## Ending run for threshold: {threshold:.2f}, PK parameter: {target_pk_parameter}, model: {pipeline2_model_name}\n")
            f.write("======" * 50 + "\n")
    # ---------------------
    # Save final results
    # ---------------------
    with open(f"{save_log_file_name.split('.txt')[0]}.json", "w") as f:
        json.dump(scores_log_all, f, indent=4)

    return scores_log_all


if __name__ == "__main__":

    target_pk_parameters = ['half-life', 'AUC','CL', "MRT", "CMAX", "TMAX"]
    threshold = 0.69
    pipeline2_model_name = "llama3"
    save_log_file_name = f"experiments/evaluation_logs/eval_log_{int(time.time())}.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # or "cuda" if GPU is available

    run_evaluation(
        target_pk_parameters,
        threshold,
        pipeline2_model_name,
        save_log_file_name,
        weights=(0.6, 0.2, 0.2),
        device=device
    )
