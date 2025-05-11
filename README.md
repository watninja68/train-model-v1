# LLM Fine-Tuning and Evaluation Engine

This project provides a suite of tools and scripts for fine-tuning Large Language Models (LLMs) using Generative Reward Policy Optimization (GRPO) and evaluating their performance with other LLMs acting as judges. It supports evaluation using both OpenRouter and local Ollama instances.

## Overview

The primary goal of this repository is to enable experimentation with LLM fine-tuning and to provide robust evaluation mechanisms. It leverages Unsloth for efficient training and VLLM for fast inference. The evaluation framework includes multiple reward functions, including correctness checks, format validation, and LLM-based scoring.

## Features

* **LLM Fine-Tuning:** GRPO training implemented for models like Llama 3.1 8B using Unsloth.
* **LLM-based Evaluation:**
    * Uses OpenRouter API for evaluation with various models.
    * Supports local Ollama instances for evaluation (e.g., `llama3.1:latest`).
* **Flexible Reward Functions:** Includes reward functions for:
    * Correctness against gold answers.
    * Strict and soft XML-based response formatting.
    * Presence of numerical answers.
    * LLM-judged quality scores.
* **Environment Setup:** Comprehensive shell scripts for setting up the environment, including dependencies, CUDA, Python, and Ollama.
* **Modular Script Execution:** A master script (`run.sh`) to manage and execute various setup and task-specific scripts.

## Project Structure

Directory structure:
└── watninja68-train-model-v1/ <br>
    ├── README.md  <br>
    ├── 1test.py <br>
    ├── gpro1_test_withopenRoutere.py <br>
    ├── gpro_with_ollama.py
    ├── ollama_setup.sh
    ├── per.sh
    ├── requirement.txt
    ├── temp_setup.sh
    └── test.py
## Setup Instructions

**Prerequisites:**

* Linux-based OS (tested on Ubuntu).
* NVIDIA GPU with appropriate drivers if using CUDA for training/inference.

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/watninja68/train-model-v1
    cd https://github.com/watninja68/train-model-v1
    ```

2.  **Initial Permissions and Python Setup:**
    The `per.sh` script makes other necessary scripts executable and then triggers the Python environment setup.
    ```bash
    chmod +x per.sh
    ./per.sh
    ```
    This will:
    * Install `tmux`.
    * Make `run.sh`, `ollama_setup.sh`, `temp_setup.sh`, and scripts in `runs/` executable.
    * Execute `runs/python.sh` via `run.sh python` to install Python 3.10 and the `uv` package manager.

3.  **Comprehensive Environment Setup (`temp_setup.sh`):**
    This script automates several setup stages. Review the script before running.
    ```bash
    chmod +x temp_setup.sh
    ./temp_setup.sh
    ```
    This script performs the following:
    * Ensures Python is set up (by calling `./run.sh python`).
    * Pulls a default Ollama model (`ollama pull gemma3:27b`).
    * Creates a Python virtual environment (`.venv`) using `uv`.
    * Activates the virtual environment.
    * Installs basic Python packages (`re`, `typing`, `time`, `json`, `requests`).
    * Installs `torch`, `torchvision`, `torchaudio`.
    * Installs dependencies from `requirement.txt` using `uv`.
    * Installs a specific version of `transformers`.
    * *Attempts to run a Python script (e.g., `llama3_1_(8b)_grpo.py`).
    * Adjust this to `gpro1_test_withopenRoutere.py` or `gpro_with_ollama.py`.*

4.  **Manual Setup (Alternative/Granular):**
    If you prefer a step-by-step manual setup:

    * **System Updates (Optional, requires sudo):**
        ```bash
        ./run.sh first
        ```
    * **CUDA Toolkit (If NVIDIA GPU is present, requires sudo):**
        ```bash
        ./run.sh nvcc # Installs CUDA 12.4
        ```
    * **Ollama Setup:**
        Ensure Ollama is installed and the server is running.
        ```bash
        chmod +x ollama_setup.sh
        ./ollama_setup.sh
        ```
        Then, pull the models you intend to use for evaluation (e.g., `llama3.1:latest`):
        ```bash
        ollama pull llama3.1:latest
        ollama pull <other-models-you-need>
        ```
    * **Python Environment with `uv`:**
        Ensure `runs/python.sh` has been run (done by `per.sh` or `temp_setup.sh`).
        ```bash
        uv venv # Create virtual environment
        source .venv/bin/activate # Activate
        uv pip install -r requirement.txt
        # Install other specific packages like torch if not fully covered
        uv pip install torch torchvision torchaudio
        uv pip install --no-deps git+[https://github.com/huggingface/transformers@v4.49.0-Gemma-3](https://github.com/huggingface/transformers@v4.49.0-Gemma-3)
        ```

## Usage

Ensure your Python virtual environment is activated (`source .venv/bin/activate`) before running Python scripts.

### Master Script Runner (`run.sh`)

The `run.sh` script can execute any executable script within the `./runs/` directory.
* Run all executable scripts in `./runs/`:
    ```bash
    ./run.sh
    ```
* Run a specific script by providing a filter (substring of the script name):
    ```bash
    ./run.sh python  # Runs ./runs/python.sh
    ./run.sh nvcc    # Runs ./runs/nvcc.sh
    ```
* Dry run (shows what would be executed without actually running):
    ```bash
    ./run.sh --dry
    ./run.sh python --dry
    ```

### Training Scripts

* **GRPO Fine-tuning with OpenRouter Evaluation (`gpro1_test_withopenRoutere.py`):**
    This script is adapted from a Colab notebook. You'll need to configure OpenRouter API keys (see Configuration section).
    ```bash
    uv run gpro1_test_withopenRoutere.py
    ```
* **GRPO Fine-tuning with Ollama Evaluation (`gpro_with_ollama.py`):**
    This script uses a local Ollama instance for evaluation. Ensure Ollama is running and the specified models are pulled.
    ```bash
    uv run gpro_with_ollama.py
    ```

### Evaluation Scripts (Standalone)

* **OpenRouter Evaluation (`1test.py`):**
    Tests the LLM-as-a-judge functionality using the OpenRouter API.
    ```bash
    uv run 1test.py
    ```
* **Ollama Evaluation (`test.py`):**
    Tests the LLM-as-a-judge functionality using a local Ollama instance.
    ```bash
    uv run test.py
    ```

## Key Scripts and Their Roles

* **`1test.py`:**
    * Implements `llm_evaluation_reward_func` using the OpenAI library to call OpenRouter models.
    * Includes logic for cycling through multiple API keys and fallback models.
    * Contains test data (`test_prompts`, `test_completions`, `test_answers`) for demonstration.
    * **Security Note:** Hardcoded API keys are a security risk. Use environment variables or a secure secrets management solution for production.

* **`test.py`:**
    * Implements `llm_evaluation_reward_func` to evaluate responses using a local Ollama instance.
    * Calls `evaluate_with_llm` which constructs prompts for an Ollama model to score responses.
    * Handles connection and API call errors for Ollama.

* **`gpro1_test_withopenRoutere.py`:**
    * Based on an Unsloth Colab notebook for Llama 3.1 8B GRPO fine-tuning.
    * Prepares the GSM8K dataset.
    * Defines several reward functions: `correctness_reward_func`, `int_reward_func`, `strict_format_reward_func`, `soft_format_reward_func`, `xmlcount_reward_func`.
    * Integrates an LLM-based evaluation (intended to be similar to `1test.py`) within the GRPO training loop using OpenRouter.
    * Includes model saving and inference examples.

* **`gpro_with_ollama.py`:**
    * Similar structure to `gpro1_test_withopenRoutere.py` but adapted for Ollama.
    * Uses the `Advaith1612/paired-coding` dataset.
    * The `llm_evaluation_reward_func` in this script is configured to use local Ollama models for scoring during GRPO training.

## Configuration


### Ollama Configuration

* **Ollama Server:** Ensure the Ollama server is running. By default, scripts connect to `http://localhost:11434`. This can be changed in `test.py` and `gpro_with_ollama.py` (variable `ollama_base_url`).
* **Models:** Pull the Ollama models specified in the scripts.
    * `test.py` and `gpro_with_ollama.py` default to `llama3.1:latest`.
    * `temp_setup.sh` pulls `gemma3:27b`.
    ```bash
    ollama pull llama3.1:latest
    ollama list # To see pulled models
    ```

### Model Choices in Scripts

* The Python scripts often define a list of `models` to try for evaluation. You can modify these lists to use different OpenRouter or Ollama models.

## Reward Functions

The project utilizes several types of reward functions to guide the GRPO training and for evaluation:

* **`extract_xml_answer` / `extract_hash_answer`:** Helper functions to parse model outputs, typically extracting content within `<answer>` tags or based on `####` delimiters.
* **`correctness_reward_func`:** Compares the extracted answer to a gold standard answer.
* **`int_reward_func`:** Checks if the extracted answer is a digit.
* **`strict_format_reward_func` / `soft_format_reward_func`:** Validate if the model's output adheres to a specific XML-like structure (`<reasoning>...</reasoning><answer>...</answer>`).
* **`xmlcount_reward_func`:** Provides a fine-grained reward based on the correct usage and placement of XML tags.
* **`llm_evaluation_reward_func`:** (Present in `1test.py`, `test.py`, `gpro_with_ollama.py`) - This is the core LLM-as-a-judge function. It sends the question, the model's response, and optionally a reference answer to another LLM (OpenRouter or Ollama) which then returns a numerical quality score.

## `tmux` Configuration

The file `runs/sudo` appears to be a `tmux` configuration file, likely misnamed. If you use `tmux`, you might want to rename it to `~/.tmux.conf` or source it from your existing `tmux` configuration.
The line `bind r source-file <WHERE YOUR TMUX CONF GOES> \; display-message tmux.conf reloaded` within the file should be updated to point to its actual path, for example:
```tmux
bind r source-file ~/.tmux.conf \; display-message ".tmux.conf reloaded"
or if you place it in ~/.config/tmux/tmux.conf:

Code snippet

bind r source-file ~/.config/tmux/tmux.conf \; display-message "~/.config/tmux/tmux.conf reloaded"
