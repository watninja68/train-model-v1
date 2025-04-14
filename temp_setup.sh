#!/usr/bin/env bash
./run.sh python
ollama pull gemma3:27b
uv venv
source .venv/bin/activate
uv pip install re typing time json requests 
uv pip   install torch torchvision torchaudio
uv pip   install -r requirement.txt
uv pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
uv run "llama3_1_(8b)_grpo.py"
