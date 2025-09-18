"""
pk_form_prompts.py
Prompt templates and example builders for PK parameter form detection.
"""


pk_form_extraction_prompt = """Extract all forms of {pk_parameter} (in various forms, like {param_aliases}) based on the table provided. Write the exact names in the format of <$form$> using '$$' symbols like $form1$, $form2$, etc., without adding any extra text and without further information.
Only provide {pk_parameter} exactly in the form that presented in the table without any changes. If it is between '^^' like 'random1^form1^random2' as multi header just provide what related to the form1 like $form1$ and ignore the rest. It can be more than 1 forms of {pk_parameter} in the table.
Do not include any forms where:
{non_params_alias}"""

pk_extraction_config = {
    "half-life": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="half-life",
            param_aliases="half-life, HL, T1/2, T½, T12 with combination of elimination, termination, alpha(α), beta(β), gamma(γ), lambda(λ), delta(δ) and etc.",
            non_params_alias=""
        ),
        'examples': [
            "$effective T1/2 delta (days)$,$half-life lambda2 (h)$,$t1/2 (h)$",
            "$HLδ (h)$",
            "$HL_lambda2$,$HL_ lambdaz$,$T1/2 δ$",
            "$HLα$,$T1/2 Eli$",
            "$effective T1/2 z$,$T12γ$",
            "$T12lambda2$,$half-lifeα$,$effective T1/2lambdaz$,$HL λz$"
        ],
    },
    "AUC": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="AUC",
            param_aliases="AUC0–∞, AUC0–24, AUCinf, AUC12–24, partial AUC, AUC% extrap",
            non_params_alias=""
        ),
        'examples': [
            "$AUC0-∞ (ng.h/mL)$,$AUC% extrap (%)$",
            "$Oral AUC (μMh)$",
            "$AUEC0-72$",
            "$AUC0-inf (ng*hrs/ml)$",
            "$AUClast (ng h mL-1)$",
            "$AUC0-192 g hrs per ml$"
        ]
    },
    "CMAX": {
        "prompt" : pk_form_extraction_prompt.format(
            pk_parameter="CMAX",
            param_aliases="Cmax, peak concentration, maximum concentration",
            non_params_alias="1- C0, Co, Cop, or any expression where C is followed by a number. 2-Any unrelated capital letters like A, B, or terms like Tmax"
        ),
        "examples": [
            "$Cmax (ng/mL)$",
            "",
            "$peak concentration$",
            "",
            "$cmax pgx per ml$",
            "$Maximum concentration pg/ml$"
        ],

    },
    "TMAX": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="TMAX",
            param_aliases="Tmax, time to peak drug concentration, time to maximum concentration",
            non_params_alias=""
        ),
        "examples": [
            "$Tmax$",
            "",
            "$time to peak drug concentration$",
            "",
            "$time to maximum concentration$",
            "$Tmax (hours)$"
        ],
    },
    "MRT": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="MRT",
            param_aliases="MRT, mean residence time",
            non_params_alias="1-Any form of half-life: t1/2, T1/2, t ½, t½, t1/2α, t1/2β, t ½ λ z, etc"
        ),
        "examples": [
            "$MRT (h)$",
            "",
            "$MRT (min)$",
            "",
            "",
            "$Mean residence time$"
        ]
    },
    "CL": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="Cl",
            param_aliases="Cl, clearance, Absolute antipyrine clearance, Antipyrine clearance, C L/F , CL, CL NR, CL R,  CL/F , CL/F Gmean (CV%), CL/F, CL/fm, CL2, CL3, CLNR ,CLR, CLT, CL_obs, CLoral, CLtot/F, CLz/F, Cl, Cl/F, ClB, ClB/F, Cl_obs , Cld, Cld, Clpl, Cls, Clt",
            non_params_alias="1- C0, C1, Cinitial, Cinitial (μg/mL), or any variant of concentration at time 0 or other timepoints."
        ),
        "examples": [
            "$CL/F (L/h)$",
            "$CL (L/h/kg)$",
            "$Cl$,$CLz/F$",
            "",
            "$Cl_obs (g/min)$",
            "$Cl/Vd$"
        ]
    },
    "VD": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="VD",
            param_aliases="Vd, volume of distribution, Vd/F, Vss, Vz, Vc, Vp, Varea, Vss/F, Vdss, Vdss/F, apparent volume of distribution, volume of distribution at steady state",
            non_params_alias="1-Clearance terms: CL, CL/F, Cl, etc. 2- Ambiguous compartments like V1, V2, Vc, Vp"
        ),
        "examples": [
            "$Vd/F (L)$",
            "$Vc (L/kg)$",
            "$V(ss)$,$Vd/F$",
            "$Volume of distribution$",
            "$Vz$",
            "$Varea$"
        ],
    },
    "bioavailability": {
        "prompt": pk_form_extraction_prompt.format(
            pk_parameter="bioavailability",
            param_aliases="Bioavailability, F, F%. It could be only F or biavaliablity.",
            non_params_alias="1-Any pharmacokinetic ratios (e.g., CL/F, V/F, AUC/F, Vz/F, Vc/F, Vss/F), 2-Any terms that refer to excretion or recovery of dose (e.g., Ae, GI, urine, feces, % of dose, Recovery). 3-Unrelated abbreviations (e.g., IVR, RE)"
        ),
        "examples": [
            "",
            "$%F$",
            "$F bio$",
            "",
            "$F%$",
            "$Bioavailability (%)$"
        ],
    }
}


def build_pk_prompt_examples(pk_name="half-life", pk_config=pk_extraction_config):

    # EXAMPLES:
    PK_extraction_examples = [ 
        # Example 1
        {
            "role": "user",
            "content": f"""This is the table:
Parameter^Parameter^,Parameter^Estradiol cypionate^Parameter^Mean^,Parameter^Estradiol cypionate^Parameter^SD^,Parameter^Estradiol cypionate^Parameter^Minimum^,Parameter^Estradiol cypionate^Parameter^Median^,Parameter^Estradiol cypionate^Parameter^Maximum^
effective T1/2 delta (days),14.07,6.32,3.25,15.76,21.27
AUC0-∞ (ng.h/mL),17.01,9.62,5.29,16.23,41.19
AUC% extrap (%),15.27,18.07,2.68,6.42,52.81
Cmax (ng/mL),0.14,0.08,0.04,0.14,0.32
half-life lambda2 (h),16.83,21.07,2.0,9.0,72.0
t1/2 (h),89.65,76.04,23.02,70.96,292.84
Vd/F (L),28038.0,9636.0,13644.0,28983.0,43543.0
CL/F (L/h),49.02,10.62,33.96,48.11,70.47
Kel (h−1),0.012,0.007,0.002,0.01,0.03
MRT (h),576.05,238.32,259.05,514.92,982.88
Tmax,2.0,1.0,1.0,2.0,4.0
Ae0–24h (%dose),4.87±1.23,15.5±8.00 ***,3.73±1.15,17.3±6.80 ***
GI24h (%dose),13.6±4.92,14.7±6.87,16.1±6.62,15.0±6.27

question: {pk_config[pk_name]['prompt']}"""},
    {
            "role": "assistant",
            "content": pk_config[pk_name]['examples'][0]
        },
        # Example 2
        {
            "role": "user",
            "content": f"""This is the table:
Compound^,Compound^HLδ (h)^,Compound^V c (L/kg)^,Compound^CL (L/h/kg)^,Compound^%F^,Compound^Oral AUC (μMh)^
9b,0.6,2.1,4.0,82,5.78
9c,0.8,13.3,11.5,100,3.38
9g,0.6,3.5,4.2,85,8.64
9k,0.4,1.8,3.0,52,6.33

question: {pk_config[pk_name]['prompt']}"""
        },
        { 
            "role": "assistant",
            "content": pk_config[pk_name]['examples'][1]
        },
        # Example 3
        {
            "role": "user",
            "content": f"""This is the table:
Parameter,Unit,chicken
,,Intramuscular
V(ss),l,7.48799847231819
time to peak drug concentration,day,14.0475596124332
HL_lambda2,hours,17.9428514196301
peak concentration,mgxl-1,90.1222911324324
AUEC0-72,µgxhrµl-1,637
MRT (min),WTD,130.879903809212
HL_ lambdaz,minutes,34.5430042690434
Cl,pg/s,8.12618202580096
T1/2 δ,h,15.0462053996788
F bio,%,124.5124
CLz/F,mL/h/kg,1.1256
Vd/F,mL/kg,114.2656

question: {pk_config[pk_name]['prompt']}"""},
        {
            "role": "assistant",
            "content": pk_config[pk_name]['examples'][2]
        },
        # Example 4
        {
            "role": "user",
            "content": f"""This is the table:
^Agent^,^^Agent^Volume of distribution^,^HLα^Agent^Normal^,animal^T1/2 Eli^Agent^CrCl<10 mL/min^,^AUC0-inf (ng*hrs/ml)^animal^CrCl<10 mL/min
Amikacin,0.3,2.5–3,30,142
Arbekacin,nan,2.3,nan,nan
Dibekacin,0.16,2.1,5,nan
Gentamicin,0.22–0.3,2.5–3,30–50,12-13
Isepamicin,0.11,2.3,47,41
Kanamycin,0.27,2–5,72–96,13
Netilmycin,0.26,2–2.3,40,41
Sisomicin,0.22,2–3,35–80,51
Streptomycin,nan,2.5,100,21
Tobramycin,0.33,2.5–3,56,45

question: {pk_config[pk_name]['prompt']}"""},
        {
            "role": "assistant",
            "content": pk_config[pk_name]['examples'][3]
        },
        # Example 5
        {
            "role": "user",
            "content": f"""This is the table:
Parameter,Unit,Cat
,,PO
F%,WTD,165.381244143094
AUClast (ng h mL-1),,861
T12γ,minutes,44.868250662551
cmax pgx per ml,,74.4595976411061
Vz,l,28.0939168948168
effective T1/2 z ,hrs,25.901475889529
Cl_obs (g/min),20.8808456744995
time to maximum concentration,days,10.6236202895275

question: {pk_config[pk_name]['prompt']}"""},
        {
            "role": "assistant",
            "content": pk_config[pk_name]['examples'][4]
        },
        # Example 6
        {
            "role": "user",
            "content": f"""This is the table:
Parameter,Unit,chicken,chicken,chicken,chicken
,,PO,IM,IV,SC
T12lambda2,days,8.89472434572088,6.10637470068754,37.3356045918512,1.95982279027228
Cl/Vd,L/h,31.1483261141686,29.438661417949,27.5657528318013,6.07159075945379
half-lifeα,days,10.2339816343675,19.1010518839447,2.66823733319522,5.28515833515929
Varea,l,35.3617553329068,60.0476319963462,65.29583763447,42.6410653696861
Maximum concentration pg/ml,,61.9009572708798,8.10452581390612,36.4599225207599,159.711472537316
effective T1/2lambdaz,min,14.4421708494395,6.41109061985708,9.31356816337494,23.2400988578029
HL λz,hr,46.7079859007555,5.18555577454753,44.8031271861549,40.18102926917
AUC0-192 g hrs per ml,,222,912,279,532
Mean residence time,WTD,185.115998697057,75.8098105551152,62.6956066702552,121.509813595095
Bioavailability (%),,185.121697057,75.841105551152,62.6954566702552,121.51813595095
Tmax (hours),,2.5,3.0,1.5,4.0

question: {pk_config[pk_name]['prompt']}"""},

        { 
            "role": "assistant",
            "content": pk_config[pk_name]['examples'][5]
        }
]
    return pk_config[pk_name]['prompt'], PK_extraction_examples
