
# from huggingface_hub import notebook_login

# notebook_login()
from vllm import SamplingParams
import requests
from trl import GRPOConfig, GRPOTrainer
import json
import time
from typing import List, Dict, Any, Union, Optional
from datasets import load_dataset, Dataset
import re
from unsloth import FastLanguageModel
import torch
max_seq_length = 1024  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "```" not in text:
        return None
    return text.split("####")[0].strip()

# uncomment middle messages for 1-shot prompting


def get_gsm8k_questions(split="train", limit=None) -> Dataset:
    data = load_dataset('Advaith1612/paired-coding')[split]

    # Limit to the first 1000 examples
    if limit and len(data) > limit:
        data = data.select(range(limit))

    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

    return data


dataset = get_gsm8k_questions(limit=10)

# Reward functions


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def llm_evaluation_reward_func(prompts, completions, answer=None, **kwargs) -> list[float]:
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
- 0.125: Partially addresses the question but contains significant errors or omissions.
- 0.5: Adequately addresses the question but may have some minor errors or lack depth.
- 0.6: Good response - correctly and thoroughly addresses the main points of the question.
- 0.75: Excellent response - comprehensive, accurate, insightful, and clearly answers the question.

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


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=6,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        llm_evaluation_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        # int_reward_func,
        # correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()


text = tokenizer.apply_chat_template([
    {"role": "user", "content": "Calculate pi."},
], tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = model.fast_generate(
    [text],
    sampling_params=sampling_params,
    lora_request=None,
)[0].outputs[0].text

output


model.save_lora("grpo_saved_lora")


text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Calculate pi."},
], tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)


# Merge to 16bit
# if True: model.save_pretrained_merged("test_llama", tokenizer, save_method = "merged_16bit",)
if True:
    model.push_to_hub_merged("PoppingFace468/test_llama",
                             tokenizer, save_method="merged_16bit", token="")

# Merge to 4bit
if False:
    model.save_pretrained_merged(
        "model", tokenizer, save_method="merged_4bit",)
if False:
    model.push_to_hub_merged("hf/model", tokenizer,
                             save_method="merged_4bit", token="")

# Just LoRA adapters
if False:
    model.save_pretrained_merged("model", tokenizer, save_method="lora",)
if False:
    model.push_to_hub_merged("hf/model", tokenizer,
                             save_method="lora", token="")


# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer,
                           quantization_method="f16", token="")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf(
        "model", tokenizer, quantization_method="q4_k_m")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer,
                           quantization_method="q4_k_m", token="")
