"""
pk_final_prompts.py
Pipeline 2: Prompt templates and example builders for PK parameter data structured extraction.
"""

final_extraction_examples_prompt = """Extract pharmacokinetic (PK) data (PK parameter and associated information) from any tables may appear inside a scientific document (the document can also contain a title, abstract, table caption and table footnotes). Return one and only one comma-separated table with the header in the exact column order shown below—no commentary, no extra columns, no blank lines. My tables are in the specific text representation format which I combined my target row to the header with @ sign for each cell and header can combine to other header with '^' if the table is multi-header table. I want to convert this into a table format with the following columns (if not exists any data only left it with None): 
pk_parameter,pk_parameter_unit,pk_parameter_value,animal,drug,drug_dosage,route_of_administration,animal_matrix/commodity

Extraction rules

Extract PK parameters that relate to {pk_parameter} only ({param_aliases}). Accept every synonym, abbreviation, symbol. Do not translate or normalise; copy it exactly (e.g. “t½(γ)”, “AUCMC_t-∞”, “CL”, “Vd(ss)”).
route_of_administration including but not limited to Oral (PO), Intravenous (IV), Intramuscular (IM), Subcutaneous (SC), SubQ (SubQ), Sublingual (SL), Buccal (BUCC), Topical (TOP), Transdermal (TD), Inhalation (INH), Rectal (PR), Intranasal (IN), Ophthalmic (OPHTH), Otic (OTIC), Intrathecal (IT), Intraosseous (IO), Intraperitoneal (IP), Intravitreal (IVT), Intrapleural (IPP), Intraarterial (IA), Intracardiac (IC), Intradermal (ID), Subungual (SUB), Vaginal (VAG), Dermal (DERM), Transrectal (TR), Intramammary (IMM).
Animals only the animal name and not the breed including but not limited to dog, cat, chicken, cattle, swine, pig, sheep, goat, buffalo, deer, rabbit, horse, turkey, duck, geese, goose, guinea fowl, quail, salmon, tuna, trout, catfish, tilapia, cod, herring, sardines, shrimp, crab, lobster, mussels, oysters, clams, scallops, pheasant, partridge. Could be patient, human, female, male, childeren, adult, etc.
animal_matrix/commodity animal products including but not limited to plasma, urine, tissue, milk, liver, muscle, kidney, brain, heart, fat, egg, leather, wool, honey, blood, gelatin, lung, spleen, pancreas, intestine, stomach, gallbladder, thymus, gizzard
drug_dosage including dose values (including decimals and ± signs) followed by weight units (g, grams, ...) or volume units (l, liters,...) with optional prefixes (milli, micro, nano, pico,...).
pk_parameter_value: For each animal column with a numeric entry on that row, copy the full cell content exactly, including “±” to the SD, ranges, parentheses, superscripts, etc. Do not collect p-value, 
These information may be presented in the table, table footnote, table caption, document title, and document abstract. If they are not presented in the table, table footnote, and table caption, they may be presented in the document title and document abstract but they should be related to the table.
Multi-header tables are concatenated into a single header with '^' seperators.
Missing data: Use "None" for the missing data or empty cells in the input table
No multi drug or animal or route_of_administration or animal_matrix/commodity in a single row. Separete them into multiple rows.
"""

# 1. Define a config mapping each PK parameter to its prompt template and its list of examples:
final_extraction_config = {
    "half-life": {
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="half-life",
            param_aliases="half-life, HL, T1/2, T½, T12 with combination of elimination, termination, alpha(α), beta(β), gamma(γ), lambda(λ), delta(δ) and etc."
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "HLgamma",
                    "unit": "min",
                },
            ],
            [
                {
                    "variant": "t1/2eli",
                    "unit": "h",
                }
            ],
            [
                {
                    "variant": "t½ absorption",
                    "unit": "hours",
                }
            ],
            [
                {
                    "variant": "half-life",
                    "unit": "days",
                },
                {
                    "variant": "T1/2 lambda",
                    "unit": "min",
                },
                {
                    "variant": "T12delta",
                    "unit": "weeks",
                },
            ],
            [
                {
                    "variant": "t½(γ)",
                    "unit": "d",
                },
                {
                    "variant": "t½ k 01",
                    "unit": "h",
                }

            ],

        ],
    },

    "AUC": {
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="AUC",
            param_aliases="AUC, area under the curve, AUC0-∞, AUC0-t, AUCinf, AUCt, AUCtau, AUClast, AUCall, AUCss, AUC24h, AUC12h, AUC8h, AUC6h, AUC4h, AUC2h"
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "AUC 0-∞",
                    "unit": "mg*h/L",
                },
            ],
            [
                {
                    "variant": "AUC0-t",
                    "unit": "ug/mL h",
                }
            ],
            [
                {
                    "variant": "AUCinf",
                    "unit": "μg·min/mL",
                }
            ],
            [
                {
                    "variant": "Oral AUC",
                    "unit": "hour ng mL−1",
                },
                {
                    "variant": "AUClast",
                    "unit": "μg*day/ml",
                },
                {
                    "variant": "Ae0–24h",
                    "unit": "mIUhL−1",
                },
            ],
            [
                {
                    "variant": "AUCss",
                    "unit": "ng·h−1·mL−1",
                },
                {
                    "variant": "AUC% extrap",
                    "unit": "%",
                }

            ],
        ]  
    },
    "CMAX": {
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="CMAX",
            param_aliases="Cmax, maximum concentration, peak concentration, peak plasma concentration, maximum plasma concentration, Cmax0-∞, Cmax estimated, C max"
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "Cmax",
                    "unit": "mg/mL",
                },
            ],
            [
                {
                    "variant": "maximum concentration",
                    "unit": "ng mL−1",
                }
            ],
            [
                {
                    "variant": "Cmax maximum plasma concentration",  
                    "unit": "mg/L",
                }
            ],
            [
                {
                    "variant": "C max estimated",
                    "unit": "μmol L−1",
                },    
            {   
                    "variant": "C max",
                    "unit": "μgg−1",
                },
                {
                    "variant": "maximum plasma concentration",
                    "unit": "mIUL−1",
                }
            ],
            [
                {
                    "variant": "cmax",
                    "unit": "μg/L",
                },
                {
                    "variant": "Cmax0-∞",
                    "unit": "nM",
                }
            ]
        ]
    },

    "TMAX": {
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="TMAX",
            param_aliases="Tmax, time to peak concentration, time to maximum concentration, Tmax0-∞, Tmax estimated"
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "Tmax",
                    "unit": "h",
                },
            ],
            [
                {
                    "variant": "time to peak concentration",
                    "unit": "hours",
                }
            ],
            [
                {
                    "variant": "Time to reach maximum concentration",
                    "unit": "day",
                }
            ],
            [
                {
                    "variant": "time to maximum concentration",
                    "unit": "days",
                },
                {
                    "variant": "Tmax maximum concentration",
                    "unit": "h",
                },
                {
                    "variant": "Tmax",
                    "unit": "h",
                }
            ],
            [
                {
                    "variant": "Tmax0-∞",
                    "unit": "min",
                },
                {
                    "variant": "Tmax estimated",
                    "unit": "h",
                }
            ]
        ]
    },
    # MRT
    "MRT": {
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="MRT",
            param_aliases="MRT, mean residence time, MRT0-∞, MRT0-t, MRTinf, MRTt, MRTtau, MRTlast, MRTall, MRTss, MRT24h, MRT12h, MRT8h, MRT6h, MRT4h, MRT2h"
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "MRT",
                    "unit": "h",
                },
            ],
            [
                {
                    "variant": "mean residence time",
                    "unit": "hours",
                }
            ],
            [
                {
                    "variant": "MRT0-∞",
                    "unit": "h",
                }
            ],
            [
                {
                    "variant": "MRT0-t",
                    "unit": "h",
                },
                {
                    "variant": "MRTinf",
                    "unit": "mins",
                },
                {
                    "variant": "MRT0-last",
                    "unit": "hour",
                }
            ],
            [
                {
                    "variant": "MRTtau",
                    "unit": "h",
                },
                {
                    "variant": "MRTblood",
                    "unit": "min",
                },
            ],
        ]

    },
    # Bioavailability
    "bioavailability": {
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="Bioavailability",
            param_aliases="Bioavailability, F, F%."
        ),
        "examples": [
            [
                {
                    "variant": "Bioavailability",
                    "unit": "%",
                },
            ],
            [
                {
                    "variant": "F",
                    "unit": "%",
                }
            ],
            [
                {
                    "variant": "F",
                    "unit": "%",
                }
            ],
            [
                {
                    "variant": "F%",
                    "unit": "%",
                },
                {
                    "variant": "F oral",
                    "unit": "",
                },
                {
                    "variant": "F IV",
                    "unit": "%",
                }

            ],
            [
                {
                    "variant": "F bioavailability",
                    "unit": "%",
                },
                {
                    "variant": "F bio",

                    "unit": "%",
                }
            ],

        ]
    },

    # Volume of distribution
    "VD":{
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="VD",
            param_aliases="Vd, volume of distribution, Vd/F, Vss, Vz, Vc, Vp, Varea, Vss/F, Vdss, Vdss/F, apparent volume of distribution, volume of distribution at steady state"
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "V SS",
                    "unit": "L/kg",
                },
            ],
            [
                {
                    "variant": "volume of distribution",
                    "unit": "mL kg–1",
                }
            ],
            [
                {
                    "variant": "Vss_obs",
                    "unit": "lkg-1",
                }
            ],
            [
                {
                    "variant": "Vd/F",
                    "unit": "L",
                },
                {
                    "variant": "V d(area)",
                    "unit": "l/kg",
                },
                {
                    "variant": "Vc)",
                    "unit": "mL/m2",
                }
            ],
            [
                {
                    "variant": "Vz",
                    "unit": "l",
                },
                {
                    "variant": "vdiss",
                    "unit": "mLkg-1",
                }
            ]
        ]
    },
    # Clearance
    "CL":{
        "prompt": final_extraction_examples_prompt.format(
            pk_parameter="Cl",
            param_aliases="Cl, clearance, Absolute antipyrine clearance, Antipyrine clearance, C L/F , CL, CL NR, CL R,  CL/F , CL/F Gmean (CV%), CL/F, CL/fm, CL2, CL3, CLNR ,CLR, CLT, CL_obs, CLoral, CLtot/F, CLz/F, Cl, Cl/F, ClB, ClB/F, Cl_obs , Cld, Cld, Clpl, Cls, Clt"
        ),
        "examples": [
            # Example 1 all the concepts list
            [
                {
                    "variant": "Cl",
                    "unit": "ml/min",
                },
            ],
            [
                {
                    "variant": "Cls",
                    "unit": "mL kg−1 minute−1",
                }
            ],
            [
                {
                    "variant": "Cl/F",
                    "unit": "L/h/kg",
                }
            ],
            [
                {
                    "variant": "Absolute antipyrine clearance",
                    "unit": "ml/min per kg",
                },
                {
                    "variant": "CLR",
                    "unit": "l/kg 24 h",
                },
                {
                    "variant": "Clpl",
                    "unit": "L/h",
                }
            ],
            [
                {
                    "variant": "CLz/F",
                    "unit": "ml/min/m2",
                },
                {
                    "variant": "CLtot/F",
                    "unit": "Lh−1 kg−1g",
                }
            ]
        ]
    }


}    

# 2. Write a builder function that, for a given pk_name, emits the list of user/assistant messages:
def build_final_extraction_examples(pk_name: str,
                                    pk_config: dict = final_extraction_config):
    """ """
    prompt = pk_config[pk_name]["prompt"]
    examples = pk_config[pk_name]["examples"]

    final_extraction_examples = [
    {
            "role": "user",
            "content": f"""{prompt}
This is my custom format table:
'<Patient Number^{examples[0][0]["variant"]} ({examples[0][0]["unit"]})^@Patient Number^><71@1><180@2><133@3><38@5><362@6><368@7><113@8><150@9><177 ± 124@This study><180 ± 96@Trauma3><111 ± 32@Healthy10><186 ± 55@General anesthesia11><185 ± 53@Renal failure12><123 ± 24@Burns13><143 ± 85@ICU patients14><110 ± 21@Median><0.42@P-value>'

This is footnote of my table in document:
nan

This is caption of my table in document:
Morphine Pharmacokinetics and Comparison to Previous Studies (Means ± SD)legend

This is title of my document:
The pharmacokinetics of morphine and lidocaine in nine severe trauma patients

This is abstract of my document:
Study Objective: To study the pharmacokinetic parameters of morphine and lidocaine after a single intravenous (IV) bolus in severe trauma patients. Design: Clinical case study. Setting: Department of Anesthesiology and Intensive Care of a university hospital. Patients: Nine patients, ages 24 to 91 years (mean 54.4 yrs), admitted to the hospital with severe trauma (Injury Severity Score >20) were included in the study. Interventions: After initial evaluation and stabilization, a single IV dose of morphine 0.025 mg/kg and lidocaine 1.5 mg/kg was given separately, and blood samples were drawn for each drug serum concentration. Measurements and Main Results: Morphine pharmacokinetics was studied in eight patients, lidocaine pharmacokinetics in seven patients, and both drugs were studied in six patients. Morphine clearance 2.5 to 10 ml/kg/min (6 ± 2.6, mean ± SD) and volume of distribution 0.28 to 3.30 L/kg (1.4 ± 1.0) were found to be lower than values described previously for healthy volunteers (33.5 ± 9 ml/kg/min and 5.16 ± 1.40 L/kg, respectively), and are similar to those described in trauma patients (5 ± 2.9 ml/kg/min and 0.9 ± 0.2 L/kg, respectively). In contrast, lidocaine clearance 4.5 to 9.4 ml/kg/min (6.7 ± 1.7) and volume of distribution 0.39 to 1.20 L/kg (0.72 ± 0.28) were similar to the value described in healthy volunteers (10 ml/kg/min and 1.32 L/kg, respectively). Conclusion: Changes in pharmacokinetics of drugs eliminated by the liver may occur in patients with severe trauma. The preserved lidocaine clearance indicates an almost normal hepatic blood flow and suggests that other mechanisms may be involved in the lower morphine clearance. The findings may have applications for the treatment of severe trauma patients and suggest that drug monitoring might be needed in some instances so as to avoid toxicity.


Output Produce nothing except the final CSV lines in the order specified.
"""},
        {
            "role": "assistant",
            "content": f"""pk_parameter,pk_parameter_unit,pk_parameter_value,animal,drug,drug_dosage,route_of_administration,animal_matrix/commodity
{examples[0][0]["variant"]},{examples[0][0]["unit"]},71,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},180,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},133,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},38,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},362,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},368,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},113,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},150,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},177±124,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},180±96,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},111±32,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},186±55,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},185±53,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},123±24,patient,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},143±85,patients,morphine,0.025 mg/kg,iv,serum
{examples[0][0]["variant"]},{examples[0][0]["unit"]},110 ± 21,patients,morphine,0.025 mg/kg,iv,serum
"""
        },
        # Example 2
        {
            "role": "user",
            "content": f"""{prompt}

This is my custom format table:
'<Compd^{examples[1][0]["variant"]} ({examples[1][0]["unit"]}) d^@Compd^><n.d. e@1a><45@1><29.34@5h><2.88@6k><40.5@6b><69.3@6d><25.74@9c></@Baricitinib>'

This is footnote of my table in document:
nan

This is caption of my table in document:
Physicochemical properties and metabolic stability of preferred compounds.

This is title of my document:
Discovery of imidazopyrrolopyridines derivatives as novel and selective inhibitors of JAK2

This is abstract of my document:
Herein, we describe the design, synthesis, and structure−activity relationships of a series of imidazopyrrolopyridines derivatives that selectively inhibit Janus kinase 2 (JAK2). These screening cascades revealed that 6k was a preferred compound, with IC50 values of 10 nM for JAK2. Moreover, 6k was a selective JAK2 inhibitor with 19-fold, >30-fold and >30-fold selectivity over JAK1, JAK3 and TYK2 respectively. In cytokine-stimulated cell-based assays, 6k exhibited a higher JAK2 selectivity over JAK1 isoforms. Indeed, at a dose of 20 mg/kg compound 6k, pSTAT3 and pSTAT5 expression was reduced to levels comparable to those of control animals untreated with GM-CSF. Additionally, 6k showed a relatively good bioavailability (F = 38%), a suitable half-life time (T1/2 = 1.9 h), a satisfactory metabolic stability, suggesting that 6k might be a promising inhibitor of JAK2 for further development research for the treatment of MPNs.Image 1

Output Produce nothing except the final CSV lines in the order specified.
"""},
        {
            "role": "assistant",
            "content": f"""pk_parameter,pk_parameter_unit,pk_parameter_value,animal,drug,drug_dosage,route_of_administration,animal_matrix/commodity
{examples[1][0]["variant"]},{examples[1][0]["unit"]},n.d. e,None,Compound 1a,None,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},45,None,Compound 1,None,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},29.34,None,Compound 5h,None,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},2.88,None,Compound 6k,20 mg/kg,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},40.5,None,Compound 6b,None,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},69.3,None,Compound 6d,None,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},25.74,None,Compound 9c,None,None,None
{examples[1][0]["variant"]},{examples[1][0]["unit"]},/,None,Baricitinib,None,None,None"""
        },
        # Example 3
               {
            "role": "user",
            "content": f"""{prompt}

This is my custom format table:
'<{examples[2][0]["variant"]} ({examples[2][0]["unit"]})@PK parameters a $ PK parameters a $ ><9.87±0.41a@PK parameters a$Compd: 9 a,b,c $ PK parameters a$sprague_dawley Rat i.v dose (5mg/kg) $ ><11.2±1.95@PK parameters a$Compd: 9 a,b,c $ PK parameters a$sprague_dawley Rat oral dose (10mg/kg) $ ><ND@PK parameters a$Compd: 9 a,b,c $ PK parameters a$G.Pig oral dose (100mg/kg) $ ><5.50±0.16@PK parameters a$Compd: 11l a,b $ PK parameters a$sprague_dawley Rat IV dose (5mg/kg) $ ><6.77±0.87@PK parameters a$Compd: 11l a,b $ PK parameters a$sprague_dawley Rat oral dose (10mg/kg) $ >'

This is footnote of my table in document:
–: not applicable; ND: Not determined.

This is caption of my table in document:
Oral PK of 9 and 11l in sprague_dawley Rat and guinea pig.

This is title of my document:
Tricyclic 4,4-dimethyl-3,4-dihydrochromeno[3,4-d]imidazole derivatives as microsomal prostaglandin E2 synthase-1 (mPGES-1) inhibitors: SAR and in vivo efficacy in hyperalgesia pain model

This is abstract of my document:
A series of substituted tricyclic 4,4-dimethyl-3,4-dihydrochromeno[3,4-d]imidazole derivatives have been synthesized and their mPGES-1 biological activity has been disclosed in detail. Structure-activity relationship (SAR) optimization provided inhibitors with excellent mPGES-1 potency and low to moderate PGE2 release A549 cell potency. Among the mPGES-1 inhibitors studied, 7, 9 and 11l provided excellent selectivity over COX-2 (>200-fold) and >70-fold selectivity for COX-1 except 11l, which exhibited dual mPGES-1/COX-1 activity. Furthermore, the above tested mPGES-1 inhibitors demonstrated good metabolic stability in liver microsomes, high plasma protein binding (PPB) and no significant inhibition observed in clinically relevant CYP isoforms. Besides, selected mPGES-1 tool compounds 9 and 11l provided good in vivo pharmacokinetic profile and oral bioavailability (%F=33 and 85). Additionally, the representative mPGES-1 tool compounds 9 and 11l revealed moderate in vivo efficacy in the LPS-induced thermal hyperalgesia guinea pig pain model.
     
Output Produce nothing except the final CSV lines in the order specified.
"""},
        {
            "role": "assistant",
            "content": f""""pk_parameter,pk_parameter_unit,pk_parameter_value,animal,drug,drug_dosage,route_of_administration,animal_matrix/commodity
{examples[2][0]["variant"]},{examples[2][0]["unit"]},9.87±0.41,Rat,Compound 9,5mg/kg,i.v,None
{examples[2][0]["variant"]},{examples[2][0]["unit"]},11.2±1.95,Rat,Compound 9,10mg/kg,oral,None
{examples[2][0]["variant"]},{examples[2][0]["unit"]},ND,Pig,Compound 9,100mg/kg,oral,None
{examples[2][0]["variant"]},{examples[2][0]["unit"]},5.50±0.16,Rat,Compound 11l,5mg/kg,IV,None
{examples[2][0]["variant"]},{examples[2][0]["unit"]},6.77±0.87,Rat,Compound 11l,10mg/kg,oral,None
"""
        },
        # Example 4 edited
        {
            "role": "user",
            "content": f"""{prompt}
This is my custom format table:
'<^{examples[3][0]["variant"]} ({examples[3][0]["unit"]})^IO@^><5.85@Paclitaxel><4.59@Abraxane><6.2@CPT-11><1408.0@nLs-CPT-11 (Pn)>','<^{examples[3][1]["variant"]} ({examples[3][1]["unit"]})^IP@^><2134.0@Paclitaxel><0.069@Abraxane><127.0@CPT-11><13.6@nLs-CPT-11 (Pn)>', ''<^{examples[3][2]["variant"]} ({examples[3][2]["unit"]})^topical@^><2198.0@Paclitaxel><44.19@Abraxane><58.12@CPT-11><81.02@nLs-CPT-11 (Pn)>'
This is footnote of my table in document:
AUC∞, area under the concentration versus time curve in plasma based on the sum of exponential terms; MRT, mean residence time; CL, clearance; V d, volume of distribution; Pn, polyphosphate-stabilized formulation; SOS, sucroseoctasulfate-stabilized formulation.

This is caption of my table in document:
Preclinical Pharmacokinetic Parameters of Various Chemotherapeutic Agents Administered as the Nanoparticle Formulation or Unencapsulated Small Molecule. All compounds are 65 Milligram Dose in Goats. Administration Route is Intravenous (IO), Intraperitoneal (IP) and topical.

This is title of my document:
Pharmacokinetics and in vivo drug release rates in liposomal nanocarrier development in goats

This is abstract of my document:
Liposomes represent a widely varied and malleable class of drug carriers generally characterized by the presence of one or more amphiphile bilayers enclosing an interior aqueous space. Thus, the pharmacological profile of a particular liposomal drug formulation is a function not only of the properties of the encapsulated drug, but to a significant extent of the pharmacokinetics, biodistribution, and drug release rates of the individual carrier. Various physicochemical properties of the liposomal carriers, the drug encapsulation and retention strategies utilized, and the properties of the drugs chosen for encapsulation, all play an important role in determining the effectiveness of a particular liposomal drug. These properties should be carefully tailored to the specific drug, and to the application for which the therapeutic is being designed. Liposomal carriers are also amenable to additional modifications, including the conjugation of targeting ligands or environment-sensitive triggers for increasing the bioavailability of the drug specifically at the site of disease. This review describes the rationale for selecting optimal strategies of liposomal drug formulations with respect to drug encapsulation, retention, and release, and how these strategies can be applied to maximize therapeutic benefit in vivo.

Output Produce nothing except the final CSV lines in the order specified.
"""
    },
        {
            "role": "assistant",
            "content": f"""pk_parameter, pk_parameter_unit, pk_parameter_value, animal, drug, drug_dosage, route_of_administration, animal_matrix/commodity
{examples[3][0]["variant"]}, {examples[3][0]["unit"]}, 5.85, goats, Paclitaxel, 65 Milligram, IO, plasma
{examples[3][0]["variant"]}, {examples[3][0]["unit"]}, 4.59, goats, Abraxane, 65 Milligram, IO, plasma
{examples[3][0]["variant"]}, {examples[3][0]["unit"]}, 6.2, goats, CPT-11, 65 Milligram, IO, plasma
{examples[3][0]["variant"]}, {examples[3][0]["unit"]}, 1408.0, goats, nLs-CPT-11 (Pn), 65 Milligram, IO, plasma
{examples[3][1]["variant"]}, {examples[3][1]["unit"]}, 2134.0, goats, Paclitaxel, 65 Milligram, IP, plasma
{examples[3][1]["variant"]}, {examples[3][1]["unit"]}, 0.069, goats, Abraxane, 65 Milligram, IP, plasma
{examples[3][1]["variant"]}, {examples[3][1]["unit"]}, 127.0, goats, CPT-11, 65 Milligram, IP, plasma
{examples[3][1]["variant"]}, {examples[3][1]["unit"]}, 13.6, goats, nLs-CPT-11 (Pn), 65 Milligram, IP, plasma
{examples[3][2]["variant"]}, {examples[3][2]["unit"]}, 2198.0, goats, Paclitaxel, 65 Milligram, topical, plasma
{examples[3][2]["variant"]}, {examples[3][2]["unit"]}, 44.19, goats, Abraxane, 65 Milligram, topical, plasma
{examples[3][2]["variant"]}, {examples[3][2]["unit"]}, 58.12, goats, CPT-11, 65 Milligram, topical, plasma
{examples[3][2]["variant"]}, {examples[3][2]["unit"]}, 81.02, goats, nLs-CPT-11 (Pn), 65 Milligram, topical, plasma
"""
        },
        # Example 5
{
            "role": "user",
            "content": f"""{prompt}
This is my custom format table:
'<{examples[4][0]["variant"]} ({examples[4][0]["unit"]})@Parameter $ Parameter $ ><89.65@Parameter$Estradiol cypionate $ Parameter$Mean $ ><76.04@Parameter$Estradiol cypionate $ Parameter$SD $ ><23.02@Parameter$Estradiol cypionate $ Parameter$Minimum $ ><70.96@Parameter$Estradiol cypionate $ Parameter$Median $ ><292.84@Parameter$Estradiol cypionate $ Parameter$Maximum $ >', '<{examples[4][1]["variant"]} ({examples[4][1]["unit"]})@Parameter $ Parameter $ ><11.17@Parameter$Estradiol cypionate $ Parameter$Mean $ ><19.94@Parameter$Estradiol cypionate $ Parameter$SD $ ><13.22@Parameter$Estradiol cypionate $ Parameter$Minimum $ ><51.16@Parameter$Estradiol cypionate $ Parameter$Median $ ><241.54@Parameter$Estradiol cypionate $ Parameter$Maximum $ >',

This is footnote of my table in document:
SD, standard deviation; AUC, area under the liver concentration versus time curve; Cmax, maximum liver concentration; tmax, time to reach Cmax; t½, elimination half-life; Vd/F, apparent volume of distribution; CL/F, apparent clearance; MRT, mean residence time; Kel, elimination rate constant.

This is caption of my table in document:
Pharmacokinetics of estradiol cypionate in healthy female dogs volunteers (n = 12) after a single Subcutaneous administration of 25 mg/mL medroxyprogesterone acetate +5 mg/mL estradiol cypionate in injectable suspension.

This is title of my document:
Quantification of estradiol cypionate in liver by liquid chromatography coupled with tandem mass spectrometry: Application in a pharmacokinetic study in healthy female dogs volunteers

This is abstract of my document:
The combination of medroxyprogesterone acetate 25 mg + estradiol cypionate 5 mg is a highly effective, monthly injectable contraceptive. For the first time, this study presents the development and validation of a sensitive method for estradiol cypionate analysis in human liver by liquid chromatography coupled with tandem mass spectrometry (LC-MS/MS). Aliquots (500 μL) of liver were extracted with ethyl ether (100%) and derivatized with dansyl chloride. Its separation was performed on a Jones Chromatography Genesis C8 column and the quantification was performed with a mass spectrometer equipped with an electrospray interface operating in negative ion mode. The run time was 6 min and the calibration curve was linear over the range of 0.005−0.15 ng/mL. The method was applied to evaluate the pharmacokinetics of estradiol cypionate in liver collected up to 1008 h (42 days) after a single Subcutaneous administration of 25 mg/mL medroxyprogesterone acetate +5 mg/mL estradiol cypionate to healthy female dogs volunteers (n = 12). The estradiol cypionate maximum liver concentration (Cmax) was 0.14 ± 0.08 ng/mL reached at 16.83 ± 21.07 h and the area under the liver concentration versus time curve (AUC0-last) was 14.07 ± 6.32 ng.h/mL. Elimination half-life (t½), apparent volume of distribution (Vd/F), apparent clearance (CL/F) and mean residence time (MRT) were 89.65 ± 76.04 h, 28038 ± 9636 L, 49.02 ± 10.62 L/h and 576.05 ± 238.32 h, respectively, showing that the estradiol cypionate release from the administration site was prolonged and there was no drug accumulation.

Output Produce nothing except the final CSV lines in the order specified.

"""},
        {
            "role": "assistant",
            "content": f"""pk_parameter,pk_parameter_unit,pk_parameter_value,animal,drug,drug_dosage,route_of_administration,animal_matrix/commodity
{examples[4][0]["variant"]},{examples[4][0]["unit"]},89.65±76.04,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][0]["variant"]},{examples[4][0]["unit"]},23.02,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][0]["variant"]},{examples[4][0]["unit"]},70.96,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][0]["variant"]},{examples[4][0]["unit"]},292.84,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][1]["variant"]},{examples[4][1]["unit"]},11.17±19.94,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][1]["variant"]},{examples[4][1]["unit"]},13.22,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][1]["variant"]},{examples[4][1]["unit"]},51.16,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
{examples[4][1]["variant"]},{examples[4][1]["unit"]},241.54,dogs,estradiol cypionate,5 mg/mL,Subcutaneous,liver
"""
        },

]
    return prompt, final_extraction_examples
