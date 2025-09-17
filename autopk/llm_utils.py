"""
llm_utils.py
Wrapper utilities for interacting with LLMs (OpenAI or local models).
"""

import re
import openai
from autopk.config import OPENAI_API_KEY, OPENAI_BASE_URL

# ==============================
# Client Initialization
# ==============================

if OPENAI_BASE_URL:  # e.g. LiteLLM proxy
    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:  # Default OpenAI
    client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ==============================
# Utility Functions
# ==============================

def extract_value_dollar(txt: str):
    """
    Extract values enclosed in $...$ markers from text.
    Example: "The value is $42$." -> ["42"]
    """
    return re.findall(r'\$(.*?)(?:\$|$)', txt)


def llm_chat_with_history(history, model_name: str = "gpt-4o-mini", max_tokens: int = 512): # TODO: add kwargs including temperature and also return the response object
    """
    Run a chat completion and return reply + updated history.
    """
    if model_name in ['gemma3', 'deepseek-r1-distill-qwen-32b', 'llama3', 'phi3', 'gpt-4o-mini']:
        response = client.chat.completions.create(
            model=model_name,
            messages=history,
            max_tokens=max_tokens,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()
    elif model_name in ['llama3.2-8b_local']:
        reply = ""  # TODO: hook to local inference
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    updated_history = history + [{"role": "assistant", "content": reply}]
    return reply, updated_history
