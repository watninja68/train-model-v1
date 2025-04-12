#!/usr/bin/env bash
#./run.sh
uv pip   install torch torchvision torchaudio
pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
uv pip   install -r requirement.txt
uv run "llama3_1_(8b)_grpo.py"
