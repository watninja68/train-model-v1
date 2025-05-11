import re
from typing import List, Dict, Any, Union, Optional
import time
import json
import requests

# --- Helper Functions (Keep as is) ---


def extract_xml_answer(text: str) -> str:
    """Extracts content between <answer> tags."""
    parts = text.split("<answer>")
    if len(parts) > 1:
        answer = parts[-1].split("</answer>")[0]
        return answer.strip()
    # Fallback or alternative extraction if needed
    # Log if tag missing
    print(f"Warning: <answer> tag not found in text: {text[:100]}...")
    # Attempt to extract based on structure if needed, or return original/empty
    return text.strip()  # Or potentially return an empty string or raise an error


def extract_hash_answer(text: str) -> str | None:
    """Extracts content before #### (Not used in the main flow here)."""
    if "####" not in text:
        return None
    return text.split("####")[0].strip()

# --- Core Evaluation Logic ---


def llm_evaluation_reward_func(prompts, completions, answer=None, **kwargs) -> list[float]:
    """
    Reward function that uses a local Ollama instance to evaluate responses.

    Args:
        prompts: List of prompt message lists (e.g., [[{'role':'user', 'content':'...'}], ...])
        completions: List of model completions (e.g., [[{'content': '<reasoning>...</reasoning><answer>...</answer>'}], ...])
        answer: Optional list of gold standard answers

    Returns:
        List of reward scores between 0.0 and 2.0
    """
    # Ensure completions is in the expected format (list of lists of dicts)
    if not completions or not isinstance(completions, list) or not isinstance(completions[0], list):
        raise ValueError(
            "completions must be a list of lists containing dictionaries.")

    responses = []
    for comp_list in completions:
        if comp_list and isinstance(comp_list[0], dict) and 'content' in comp_list[0]:
            responses.append(comp_list[0]['content'])
        else:
            # Handle unexpected format within completions list
            print(
                f"Warning: Unexpected item format in completions: {comp_list}")
            responses.append("")  # Append an empty string or handle as error

    # Extract the final answer part from the potentially structured response
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # --- Ollama Configuration ---
    # !! IMPORTANT: Replace with models you have pulled locally in Ollama !!
    # Example: ollama pull llama3:8b
    # Example: ollama pull mistral
    models = [
        "llama3.1:latest"
    ]
    ollama_base_url = "http://localhost:11434"  # Default Ollama URL

    # ollama_base_url = "http://192.168.0.104:11434"  # Default Ollama URL
    results = []

    for idx, response_to_evaluate in enumerate(extracted_responses):
        # Original question from the prompt history
        # Assumes the last message in the prompt list is the user's question
        if prompts[idx] and isinstance(prompts[idx][-1], dict) and 'content' in prompts[idx][-1]:
            question = prompts[idx][-1]['content']
        else:
            print(
                f"Warning: Could not extract question from prompt at index {idx}. Using placeholder.")
            question = "[Question Unavailable]"

        # Reference answer if provided
        reference = answer[idx] if answer is not None and idx < len(
            answer) else None

        # Score this response using Ollama
        score = evaluate_with_llm(
            question, response_to_evaluate, reference, models, ollama_base_url)
        results.append(score)

        # Optional: add delay to avoid overwhelming the local Ollama instance
        if idx < len(extracted_responses) - 1:
            time.sleep(0.2)  # Shorter delay might be fine for local calls

    return results


def evaluate_with_llm(question: str, response: str, reference: Optional[str], models: List[str], ollama_base_url: str) -> float:
    """
    Use a local Ollama instance to evaluate the quality of a response.

    Args:
        question: The original question
        response: The model's response to evaluate
        reference: Optional reference answer
        models: List of Ollama model names to try
        ollama_base_url: The base URL of the Ollama API

    Returns:
        Score between 0.0 and 2.0
    """
    # Create evaluation prompt
    if reference:
        prompt = f"""You are an expert evaluator. Your task is to evaluate the quality of a response to a question based on a reference answer.

Question:
{question}

Reference Answer:
{reference}

Response to Evaluate:
{response}

---
Instructions:
Score the "Response to Evaluate" strictly based on its factual correctness and completeness compared to the "Reference Answer". Use the following scale:
- 0.0: Completely incorrect or irrelevant.
- 0.5: Partially correct but missing key information or containing significant errors.
- 1.0: Mostly correct but with minor omissions or inaccuracies.
- 1.5: Correct and comprehensive, essentially matching the reference answer's quality.
- 2.0: Excellent - correct, comprehensive, and potentially providing valid additional details or clarity beyond the reference.

Output only a single floating-point number representing the score (e.g., "1.5") and nothing else. Do not add explanations or any other text.
Score:"""
    else:
        # No reference answer provided
        prompt = f"""You are an expert evaluator. Your task is to evaluate the quality of a response to a question.

Question:
{question}

Response to Evaluate:
{response}

---
Instructions:
Score the "Response to Evaluate" strictly based on its factual correctness, relevance, and completeness in answering the "Question". Use the following scale:
- 0.0: Completely incorrect, irrelevant, nonsensical, or fails to address the question.
- 0.5: Partially addresses the question but contains significant errors or omissions.
- 1.0: Adequately addresses the question but may have some minor errors or lack depth.
- 1.5: Good response - correctly and thoroughly addresses the main points of the question.
- 2.0: Excellent response - comprehensive, accurate, insightful, and clearly answers the question.

Output only a single floating-point number representing the score (e.g., "1.6") and nothing else. Do not add explanations or any other text.
Score:"""

    # Try each specified Ollama model until one works
    for model in models:
        try:
            print(f"Attempting evaluation with Ollama model: {model}")
            score = call_ollama_api(prompt, model, ollama_base_url)
            if score is not None:
                print(f"Received score: {score} from model {model}")
                return score
            # If we got None but no exception, it means the model responded but maybe not with a score
            print(
                f"Model {model} responded, but score extraction failed. Trying next model.")
        except requests.exceptions.ConnectionError:
            print(
                f"Error: Could not connect to Ollama at {ollama_base_url}. Is Ollama running?")
            # If connection fails for one model, it will likely fail for others too, maybe stop early?
            break  # Stop trying models if connection failed
        except Exception as e:
            print(f"Error evaluating with Ollama model {model}: {str(e)}")
            # Continue to the next model

    # If all models fail or connection fails
    print("All Ollama evaluation attempts failed. Returning default score 0.0.")
    return 0.0


def call_ollama_api(prompt: str, model: str, base_url: str) -> Optional[float]:
    """
    Call the local Ollama API (/api/chat) to get an evaluation score.

    Args:
        prompt: The evaluation prompt.
        model: The Ollama model name (e.g., "llama3:8b").
        base_url: The base URL of the Ollama API.

    Returns:
        Score between 0.0 and 2.0, or None if the call failed or score extraction failed.
    """
    url = f"{base_url}/api/chat"  # Use the chat endpoint

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,  # Get the full response at once
        "options": {
            "temperature": 0.1  # Low temperature for more deterministic scoring
            # "num_predict": 10 # Limit token generation - might cut off scores? Test this.
        }
    }

    try:
        response = requests.post(
            url, json=payload, timeout=60)  # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()

        # Extract content from the Ollama chat response structure
        content = result.get("message", {}).get("content", "")
        if not content:
            print(f"Warning: Empty content received from Ollama model {model}")
            return None

        # Log the raw response for debugging
        print(f"Ollama ({model}) raw response: '{content.strip()}'")

        # Extract the numeric score (handles potential surrounding text/whitespace)
        # This regex looks for one or more digits, optionally followed by a dot and more digits
        score_match = re.search(r"(\d+\.?\d*)", content.strip())

        if score_match:
            try:
                score_str = score_match.group(1)
                score = float(score_str)
                # Ensure score is within the expected 0.0 to 2.0 bounds
                score = max(0.0, min(score, 2.0))
                return score
            except ValueError:
                print(
                    f"Could not convert extracted score '{score_str}' to float. Raw content: '{content.strip()}'")
                return None
        else:
            print(
                f"No numeric score found in Ollama response: '{content.strip()}'")
            return None

    except requests.exceptions.RequestException as e:
        # Catch connection errors, timeouts, HTTP errors etc.
        print(f"Ollama API call failed for model {model}: {str(e)}")
        raise  # Re-raise the exception to be caught by evaluate_with_llm


# --- Test Data (Keep as is) ---
test_prompts = [
    [{"role": "user", "content": "give me code to calcualate sum of 2 numbers in rust "}],
    [{"role": "user", "content": "Calculate 15 + 27."}],
    [{"role": "user", "content": "Explain quantum computing in simple terms."}]
]
test_completions = [
    [{
        "content": "<reasoning>\nThe capital of France is Paris. This is a well-known geographical fact.\n</reasoning>\n<answer>\nParis\n</answer>"
    }],
    [{
        "content": "<reasoning>\nTo calculate 15 + 27, I add the numbers:\n15 + 27 = 42\n</reasoning>\n<answer>\n42\n</answer>"
    }],
    [{
        "content": "<reasoning>\nQuantum computing uses quantum bits or qubits which can exist in multiple states at once, unlike classical bits.\n</reasoning>\n<answer>\nQuantum computing uses qubits that can represent multiple states simultaneously thanks to superposition, enabling certain calculations to be performed much faster than classical computers.\n</answer>"
    }]
]

# Reference answers - only for the first two questions
test_answers = ["Paris", "42", None]  # Third one has no reference

# --- Run the evaluation ---
# !! IMPORTANT !!
# 1. Make sure Ollama is installed and running (e.g., `ollama serve` in your terminal).
# 2. Make sure you have pulled the models specified in the `models` list above.
#    (e.g., `ollama pull llama3:8b`, `ollama pull mistral`)
# 3. Verify the `ollama_base_url` if your Ollama isn't running on the default port/host.

print("Starting evaluation using local Ollama...")
try:
    scores = llm_evaluation_reward_func(
        test_prompts, test_completions, test_answers)
    print("\n--- Evaluation Scores ---")
    print(scores)
except Exception as e:
    print(f"\nAn error occurred during evaluation: {e}")
