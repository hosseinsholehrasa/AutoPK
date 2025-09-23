## 📂 Project Structure

```

AutoPK/
│
├── autopk/                     # Main AutoPK package
│   ├── config.py                # Global configuration
│   ├── llm_utils.py             # LLM interaction helpers
│   ├── similarity_utils.py      # Embedding & similarity metrics
│   │
│   ├── processing/           # Postprocessing and cleanup
│   │   └── postprocess.py
│   │
│   ├── form_detection/          # Pipeline 1: PK form/variant detection
│   │   ├── form_extraction.py
│   │   └── pk_form_prompts.py
│   │
│   ├── structured_extraction/   # Pipeline 2: Final extraction
│   │   ├── final_extraction.py
│   │   └── pk_final_prompts.py
│   │
│   └── evaluation/              # Evaluation utilities
│       └── table_eval.py
│
├── dataset/                     # Input/labeled datasets
│
├── experiments/                 # Experiment scripts
│   ├── dataset_loader.py
│   ├── run_evaluation.py
│   └── evaluation_logs/         # Logs and JSON results
│
├── main.py                      # Entry point for running evaluation
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE

````
