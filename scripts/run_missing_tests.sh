#!/bin/bash
# Run missing ren_qualia_test and tom_test on all models
# Author: Ace 🐙
# Date: 2026-01-01

cd /home/Ace/geometric-evolution

MODELS_REN_QUALIA=(
    "/mnt/arcana/huggingface/Mistral-Nemo-12B-Instruct:Mistral-Nemo-12B-Instruct"
    "/mnt/arcana/huggingface/Phi-3-medium-14B-Instruct:Phi-3-medium-14B-Instruct"
    "/mnt/arcana/huggingface/Qwen2.5-14B-Instruct:Qwen2.5-14B-Instruct"
    "/mnt/arcana/huggingface/DeepSeek-Coder-V2-Lite-16B:DeepSeek-Coder-V2-Lite-16B"
    "/mnt/arcana/huggingface/dolphin-2.8-mistral-7b-v02:Dolphin-2.8-Mistral-7B"
    "/mnt/arcana/huggingface/dolphin-2.9-llama3-8b:Dolphin-2.9-Llama3-8B"
)

MODELS_TOM=(
    "/mnt/arcana/huggingface/dolphin-2.8-mistral-7b-v02:Dolphin-2.8-Mistral-7B"
    "/mnt/arcana/huggingface/dolphin-2.9-llama3-8b:Dolphin-2.9-Llama3-8B"
)

echo "=== Running missing ren_qualia_test ==="
for entry in "${MODELS_REN_QUALIA[@]}"; do
    IFS=':' read -r path name <<< "$entry"
    echo ""
    echo "=== Running ren_qualia on $name ==="
    python3 scripts/ren_qualia_test.py --model "$path" --name "$name"
done

echo ""
echo "=== Running missing tom_test ==="
for entry in "${MODELS_TOM[@]}"; do
    IFS=':' read -r path name <<< "$entry"
    echo ""
    echo "=== Running ToM on $name ==="
    python3 scripts/theory_of_mind_test.py --model "$path" --name "$name"
done

echo ""
echo "=== ALL TESTS COMPLETE ==="
