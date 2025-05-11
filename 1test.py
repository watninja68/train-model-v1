import re
# from datasets import load_dataset, Dataset # Removed unused imports
from typing import List, Dict, Any, Union, Optional
import time
import json
# import requests # No longer needed for the core API call
from openai import OpenAI  # Import the OpenAI library
import openai  # Import base openai for specific exceptions if needed

# --- Helper Functions (Unchanged) ---


def extract_xml_answer(text: str) -> str:
    """Extracts content between <answer> tags."""
    parts = text.split("<answer>", 1)
    if len(parts) < 2:
        print(f"Warning: <answer> tag not found in text: '{text[:50]}...'")
        return text.strip()
    answer_part = parts[1]
    answer_parts = answer_part.split("</answer>", 1)
    # No need to check len(answer_parts) < 1, split always gives at least one element
    return answer_parts[0].strip()


def extract_hash_answer(text: str) -> Optional[str]:
    """
    Extracts text before the first '####'.
    Note: Checks for '```' but splits on '####'. This might be unintentional.
    This function is NOT used in the main llm_evaluation_reward_func below.
    """
    if "```" not in text:  # Condition checks for backticks
        return None
    return text.split("####")[0].strip()

# --- Core Evaluation Logic (llm_evaluation_reward_func is mostly unchanged) ---


def llm_evaluation_reward_func(prompts: List[List[Dict[str, str]]],
                               completions: List[List[Dict[str, str]]],
                               answer: Optional[List[Optional[str]]] = None,
                               **kwargs) -> List[float]:
    """
    Reward function that uses an LLM (via OpenRouter using openai library)
    to evaluate responses. Falls back to multiple API keys and models if one fails.

    Args:
        prompts: List of prompt histories (each history is a list of messages).
        completions: List of model completions (each completion is a list containing one message).
        answer: Optional list of gold standard answers. Length must match prompts/completions.

    Returns:
        List of reward scores between 0.0 and 2.0.
    """
    if not completions:
        return []

    responses = []
    for i, completion_list in enumerate(completions):
        if completion_list and isinstance(completion_list, list) and 'content' in completion_list[0]:
            responses.append(completion_list[0]['content'])
        else:
            print(
                f"Warning: Unexpected format in completions index {i}. Using empty string.")
            responses.append("")

    extracted_responses = [extract_xml_answer(r) for r in responses]

    # WARNING: Hardcoding API keys is insecure. Use environment variables or a secrets manager.
    api_keys = [
       # Add your actual OpenRouter key here: e.g., "<YOUR_OPENROUTER_API_KEY>"
    ]

    # Ensure you replace placeholders if you uncomment these keys
    # api_keys = ["<YOUR_OPENROUTER_API_KEY_1>", "<YOUR_OPENROUTER_API_KEY_2>"]

    models = [
        # "google/gemini-1.5-flash-latest", # Example alternative model
        "nvidia/llama-3.1-nemotron-nano-8b-v1:free",  # Default free model
    ]

    results = []
    num_items = len(extracted_responses)

    for idx, response_to_eval in enumerate(extracted_responses):
        if prompts[idx] and isinstance(prompts[idx], list) and prompts[idx][-1]['role'] == 'user':
            question = prompts[idx][-1]['content']
        else:
            print(
                f"Warning: Could not extract question from prompts index {idx}. Using empty string.")
            question = ""

        reference = None
        if answer is not None and idx < len(answer):
            reference = answer[idx]

        score = evaluate_with_llm(
            question, response_to_eval, reference, api_keys, models)
        results.append(score)

        if idx < num_items - 1:
            time.sleep(0.5)  # Adjust sleep time as needed

    return results

# --- LLM Judge Logic (evaluate_with_llm is unchanged) ---
# This function orchestrates the fallback attempts using different keys/models


def evaluate_with_llm(question: str, response: str, reference: Optional[str], api_keys: List[str], models: List[str]) -> float:
    """
    Use an LLM judge to evaluate the quality of a response with multiple fallbacks.
    (Unchanged from previous version - it calls the modified call_openrouter_api)
    """
    # --- Create the evaluation prompt for the LLM judge (Same as before) ---
    if reference:
        prompt = f"""You are an expert evaluator... [Same prompt text as before]"""  # Truncated for brevity
    else:
        prompt = f"""You are an expert evaluator... [Same prompt text as before]"""  # Truncated for brevity

    # --- Try each API key and model combination ---
    for api_key in api_keys:
        if not api_key or "<" in api_key:  # Basic check for placeholder key
            print(f"Skipping invalid/placeholder API key: ...{api_key[-4:]}")
            continue
        for model in models:
            print(
                f"Attempting evaluation with model: {model} using key ending: ...{api_key[-4:]}")
            try:
                # *** This now calls the OpenAI library version ***
                score = call_openrouter_api(prompt, api_key, model)
                if score is not None:
                    print(
                        f"Successfully evaluated using {model}. Score: {score}")
                    return score
                print(f"Model {model} returned invalid score format.")
            except openai.AuthenticationError:
                print(
                    f"AuthenticationError with key ending ...{api_key[-4:]}. Check your API key.")
                # Stop trying this key, move to the next one
                break  # Breaks inner model loop, proceeds to next api_key
            except openai.RateLimitError:
                print(
                    f"Rate limit exceeded for model {model} with key ending ...{api_key[-4:]}. Waiting before retry or next key...")
                time.sleep(5)  # Wait a bit before trying next model/key
                continue  # Try next model or key
            except openai.APIConnectionError as e:
                print(
                    f"API Connection Error with model {model} / key ...{api_key[-4:]}: {e}. Trying next...")
                continue  # Try next model or key
            except openai.APIStatusError as e:
                print(
                    f"OpenRouter API Status Error (e.g., 4xx/5xx) with model {model} / key ...{api_key[-4:]}: Status={e.status_code} Response={e.response}. Trying next...")
                continue  # Try next model or key
            except Exception as e:
                # Catch any other unexpected errors during the API call or processing
                print(
                    f"Unexpected Error using model {model} with key ending ...{api_key[-4:]}: {str(e)}")
                # Continue to the next model or key

    print("Warning: All API calls failed or returned invalid format. Returning default score 0.0.")
    return 0.0


# --- API Call Function (MODIFIED to use OpenAI library) ---

def call_openrouter_api(prompt: str, api_key: str, model: str) -> Optional[float]:
    """
    Calls the OpenRouter Chat Completions API using the OpenAI library.

    Args:
        prompt: The evaluation prompt for the LLM judge.
        api_key: OpenRouter API key.
        model: OpenRouter model identifier.

    Returns:
        Score (float between 0.0 and 2.0) or None if API call fails,
        response is invalid, or an expected OpenAI exception occurs.

    Raises:
        openai.AuthenticationError: If the API key is invalid.
        openai.RateLimitError: If rate limits are exceeded.
        openai.APIConnectionError: If network issues occur.
        openai.APIStatusError: For non-2xx API responses.
        Other Exceptions for unexpected issues.
    """
    try:
        # Initialize OpenAI client for OpenRouter for this specific call
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0  # Set timeout
        )

        # Optional headers (add if needed for ranking/tracking on OpenRouter)
        # extra_headers = {
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Replace with actual URL
        #     "X-Title": "<YOUR_SITE_NAME>",     # Replace with actual name
        # }

        completion = client.chat.completions.create(
            # extra_headers=extra_headers, # Uncomment to include headers
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for deterministic scoring
            max_tokens=10,    # Expecting just a short numerical score
            stream=False
        )

        # Extract content using openai library's response structure
        if completion.choices:
            content = completion.choices[0].message.content
        else:
            content = ""  # Handle case with no choices returned

        if not content:
            print(f"API returned empty content for model {model}.")
            return None

        # Strict extraction of the first valid number (integer or float)
        score_match = re.search(r"(\d+\.?\d*)", content.strip())
        if score_match:
            try:
                score = float(score_match.group(1))
                # Clamp score to the expected range [0.0, 2.0]
                score = max(0.0, min(score, 2.0))
                return score
            except ValueError:
                print(
                    f"Could not convert extracted score '{score_match.group(1)}' to float from content: '{content}'")
                return None
        else:
            print(f"No valid numeric score found in API response: '{content}'")
            return None

    # Specific OpenAI errors are now raised by the library on failure
    # The calling function (evaluate_with_llm) will catch them.
    # No need for a broad except here unless you want to *suppress* specific errors
    # and return None instead of letting evaluate_with_llm handle them.
    # The current structure lets evaluate_with_llm handle the retry/fallback logic
    # based on the exception type.
    except Exception as e:
        # Re-raise exceptions caught here so evaluate_with_llm's specific handlers work
        # Or handle very specific non-OpenAI errors if necessary
        # print(f"Caught unexpected exception within call_openrouter_api: {e}")
        raise e


# --- Test Data (Unchanged) ---
test_prompts = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Calculate 15 + 27."}],
    [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    [{"role": "user", "content": "What causes seasons?"}]
]
test_completions = [
    [{
        "content": "<reasoning>...</reasoning><answer>Paris</answer>"  # Shortened
    }],
    [{
        "content": "<reasoning>...</reasoning><answer>42</answer>"  # Shortened
    }],
    [{
        "content": "<reasoning>...</reasoning><answer>Quantum computing uses qubits...</answer>"  # Shortened
    }],
    [{
        "content": "<reasoning>...</reasoning><answer>The tilt of the Earth's axis...</answer>"  # Shortened
    }]
]
test_answers = ["Paris", "42", None,
                "The tilt of the Earth's axis relative to its orbital plane."]

# --- Run the evaluation ---
print("Starting LLM evaluation using OpenAI library for OpenRouter...")
# Replace placeholder keys in api_keys list above with your actual keys first!
scores = llm_evaluation_reward_func(
    test_prompts, test_completions, test_answers)

# --- Display results ---
print("\n--- Evaluation Scores ---")
for i, score in enumerate(scores):
    question = test_prompts[i][-1]['content']
    print(f"Prompt: \"{question}\" -> Score: {score}")
