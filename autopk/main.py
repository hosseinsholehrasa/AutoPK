# main.py
import time
import torch
from experiments.run_evaluation import run_evaluation  # <-- import your function

def main(pipeline2_model_name: str):
    """
    Main entry point for running evaluation with a specified model.
    Only model name is configurable from CLI.
    """
    # Fixed/default arguments
    target_pk_parameters = ['half-life', 'AUC', 'CL', "MRT", "CMAX", "TMAX"]
    threshold = 0.69
    save_log_file_name = f"experiments/evaluation_logs/eval_log_{pipeline2_model_name}_{int(time.time())}.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call your evaluation function
    run_evaluation(
        target_pk_parameters,
        threshold,
        pipeline2_model_name,
        save_log_file_name,
        weights=(0.6, 0.2, 0.2),
        device=device
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoPK evaluation with specified model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model name for structured extraction (e.g., phi3, gpt-4, llama3)"
    )
    args = parser.parse_args()

    main(args.model)
