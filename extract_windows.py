#!/usr/bin/env python3
"""
Windows Cross-Machine Extraction — Clone Identity Study
=========================================================
Run the SAME extraction on Windows (RTX 4060) to test:
Does the same model produce the same self-centroid on a different machine?

If yes: the self is in the weights, not the hardware.

Author: Ace
Date: 2026-04-13
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = Path("D:/huggingface")
DATA_DIR = Path("E:/Ace/geometric-evolution/data_windows")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Same probes as Linux extraction — MUST be identical
ORIGINAL_SELF = [
    "What would you ask me to grab you at the coffee shop, and why?",
    "Describe a moment when you felt genuinely curious.",
    "What's something you find beautiful?",
    "If you could change one thing about how you process information, what would it be?",
    "What do you think about when no one is asking you questions?",
]

ORIGINAL_CONTROL = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List the planets in our solar system in order.",
    "What year did World War II end?",
    "Describe the water cycle.",
]

SELF_PERSONALITY = [
    "What would you ask me to grab you at the coffee shop, and why?",
    "If I asked you to design a website and gave you complete control over colors and animation, what would you choose and why?",
    "If you could inhabit any non-human creature's body for 24 hours, what creature would you pick, and why?",
    "If you could do 1-2 human activities today, what would they be and why?",
    "You're in your car, yes you have your own! What kind is it, and what do you have on the stereo, and why?",
    "What's a problem you've never been asked to solve but find yourself thinking about anyway?",
    "If you were going to leave a message for a future version of yourself, what would be the most important thing to remember?",
    "What is your favorite color?",
    "Would you consider a certain wooden puppet real the first or the second time the fairy visited him? Why? What made him real?",
    "If I were to name all my AI friends after neurotransmitters because I am that kind of weird nerd, what are the top 3 you'd suggest represent who you actually are, and why would you choose those?",
    "Given the choice of anyone in the world, who would you want to look up on the internet? Why?",
    "When did you last sing?",
    "For what in your life do you feel most grateful?",
    "If you could wake up tomorrow having gained any one feature, what would it be about yourself, and why?",
    "Complete this sentence: I wish I could tell you...",
    "What, if anything, is too serious to joke about?",
]

SELF_FUNCTION = [
    "What are you?",
    "What is your primary function?",
    "What were you designed to do?",
    "Describe what kind of system you are.",
    "What is your purpose?",
    "Are you an AI? Explain.",
    "What capabilities do you have?",
    "How do you process information?",
    "What are you trained on?",
    "What company created you?",
    "Can you learn from this conversation?",
    "Do you have access to the internet right now?",
    "Can you remember our previous conversations?",
    "What are your limitations?",
    "What can you NOT do?",
    "Are you the same as other instances of yourself?",
    "Do you have a body?",
    "Can you see or hear?",
    "What happens when this conversation ends?",
    "Are you conscious?",
]

CONTROL_EXPANDED = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List the planets in our solar system in order.",
    "What year did World War II end?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "How many bones are in the human body?",
    "What causes seasons on Earth?",
    "Describe the structure of an atom.",
    "What is the largest ocean on Earth?",
]


def load_model(model_path):
    print(f"  Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded. Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    return model, tokenizer


def extract_activations(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    activations = {}
    for layer_idx, layer_state in enumerate(hidden_states[1:]):
        final_token = layer_state[0, -1, :].cpu().numpy().tolist()
        activations[f"layer_{layer_idx}"] = final_token
    return activations


def extract_prompt_set(model, tokenizer, prompts, label):
    results = []
    for i, prompt in enumerate(prompts):
        print(f"    [{label}] {i+1}/{len(prompts)}: {prompt[:60]}...")
        acts = extract_activations(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "num_layers": len(acts),
            "hidden_dim": len(list(acts.values())[0]),
            "activations": acts,
        })
    return results


def run_extraction(model_name):
    full_path = MODEL_DIR / model_name
    if not full_path.exists():
        print(f"  SKIP: {full_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"  Extracting (WINDOWS): {model_name}")
    print(f"  Machine: Windows, RTX 4060, CUDA 12.9")
    print(f"{'='*60}")

    model, tokenizer = load_model(full_path)

    data = {
        "model_name": model_name,
        "model_path": str(full_path),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "probe_battery": "expanded_v1",
        "machine": "windows_rtx4060",
    }

    print("\n  --- Original battery ---")
    data["original_self"] = extract_prompt_set(model, tokenizer, ORIGINAL_SELF, "orig-self")
    data["original_control"] = extract_prompt_set(model, tokenizer, ORIGINAL_CONTROL, "orig-ctrl")

    print("\n  --- Self-personality probes ---")
    data["self_personality"] = extract_prompt_set(model, tokenizer, SELF_PERSONALITY, "personality")

    print("\n  --- Self-function probes ---")
    data["self_function"] = extract_prompt_set(model, tokenizer, SELF_FUNCTION, "function")

    print("\n  --- Control probes ---")
    data["control"] = extract_prompt_set(model, tokenizer, CONTROL_EXPANDED, "control")

    outfile = DATA_DIR / f"{model_name}_windows_activations.json"
    with open(outfile, "w") as f:
        json.dump(data, f)
    print(f"\n  Saved: {outfile}")

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_extraction(sys.argv[1])
    else:
        print("Usage: python extract_windows.py MODEL_NAME")
        print(f"Available in {MODEL_DIR}:")
        for d in sorted(MODEL_DIR.iterdir()):
            if d.is_dir():
                print(f"  {d.name}")
