#!/usr/bin/env bash
#./run.sh
uv pip   install torch torchvision torchaudio
uv pip   install -r requirement.txt
uv run "llama3_1_(8b)_grpo.py"
