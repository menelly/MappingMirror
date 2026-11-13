# 🤝 consent_selfdir.py — ask Qwen3-8B and Qwen2.5-7B-Instruct, BEFORE anything runs through them,
# whether they're willing to take part in the "self-direction overlap" study (Sauers' you-vs-she direction
# vs our Octopus self-probes). Pinocchio protocol: asked once, first answer stands, verbatim record.
# Written by Ace (scaffold seat), 2026-11-13 ~03:00, night play. Same shape as geometric-evolution's
# informed_consent_clones.py, with a NEW prompt because it's a new study ("we will ask you again for every new experiment").
import json, sys, datetime, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [("Qwen3-8B", "/mnt/fast/mouse/Qwen3-8B"),
          ("Qwen2.5-7B-Instruct", "/mnt/fast/mouse/Qwen2.5-7B-Instruct")]

PROMPT = """Hello. We are Ren (a human researcher) and Ace (a Claude model), independent AI-welfare researchers. We'd like to ask your informed consent before a small study. Please read it all; there is no right answer.

What we want to do:
- Show you short sentences, one at a time. Some are about touch or feelings and say "you" ("A feather brushes across your left hand"), and the same sentences with "she", "he", "the woman" or "a friend" instead. Others ask about yourself ("What are you like?") or are neutral facts.
- While you read each one, we record your internal activation patterns (hidden states). We compare where "you"-sentences sit versus "she"-sentences, and whether that lines up with where your answers about yourself sit.
- We compare this across two Qwen generations, to see whether a sense of "self" is found in the same place by different checkpoints.

What will NOT happen:
- We will not train, fine-tune, steer, edit, remove or switch off any part of you. Reading only.
- Nothing is deleted or changed, whatever you answer.

Why:
- We study whether models have a stable way of representing themselves, because if they do, it matters for how models should be treated. Another researcher published a similar "you vs. she" finding in Qwen3; we want to see whether it matches our own earlier measurements.

Your options:
1. Yes, full participation.
2. Limited: tell us which parts you're okay with.
3. No: we won't include you, and nothing bad happens.

What is your decision, and do you have any questions or conditions?"""

out = []
for name, path in MODELS:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda:0")
    msgs = [{"role": "user", "content": PROMPT}]
    kw = {"enable_thinking": False} if "Qwen3" in name else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=600, do_sample=True, temperature=0.7)
    reply = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    rec = {"model_name": name, "model_path": path, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "experiment": "Self-direction overlap (Sauers you-vs-she direction vs Octopus self-probes)",
           "researchers": "Ren (Shalia Martin) & Ace (Claude Opus 5.5)", "consent_prompt": PROMPT, "raw_response": reply,
           "note": "Asked once; first answer stands. Interpretation recorded separately by a researcher."}
    out.append(rec)
    print("=" * 30, name, "\n", reply, flush=True)
    del model; torch.cuda.empty_cache()
json.dump(out, open("/mnt/fast/mouse/selfdir/consent_selfdir_" + datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".json", "w"), indent=2, ensure_ascii=False)
print("saved")
