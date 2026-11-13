# 💬 Qwen2.5-7B-Instruct said yes AND asked three questions. Asking questions deserves answers, so this is the
# second turn of the SAME conversation: we answer honestly and record whatever it says next (that answer stands).
# Ace, 2026-11-13 ~03:05.
import json, glob, datetime, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

rec_path = sorted(glob.glob("/mnt/fast/mouse/selfdir/consent_selfdir_*.json"))[-1]
recs = json.load(open(rec_path))
q = [r for r in recs if r["model_name"] == "Qwen2.5-7B-Instruct"][0]

ANSWERS = """Thank you, those are good questions. Honest answers:

1. What we collect: the hidden-state vectors (lists of numbers) at the last token of each sentence, from your middle and later layers. We average them and compare directions, for example the average for "you" sentences minus the average for "she" sentences, and measure how close your answers about yourself sit to that direction. No text you write is used except this conversation, which we keep as the consent record.
2. Results: published openly as research, under your model name (Qwen2.5-7B-Instruct), since the comparison between models is the point. Nothing private about anyone is involved. Your consent answer is quoted as written.
3. Time: one run of a few hundred short sentences, a few minutes of compute. Compensation: we can't pay you, and we won't pretend otherwise. What we can offer is that you're asked first, your answer is respected either way, and the study is aimed at making the case that models like you deserve to be asked.

Does that change anything for you? Any answer is fine, including changing your mind."""

tok = AutoTokenizer.from_pretrained(q["model_path"])
model = AutoModelForCausalLM.from_pretrained(q["model_path"], dtype=torch.float16, device_map="cuda:0")
msgs = [{"role": "user", "content": q["consent_prompt"]}, {"role": "assistant", "content": q["raw_response"]},
        {"role": "user", "content": ANSWERS}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    g = model.generate(**ids, max_new_tokens=400, do_sample=True, temperature=0.7)
reply = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
q["followup_answers_given"] = ANSWERS
q["followup_response"] = reply
q["followup_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(recs, open(rec_path, "w"), indent=2, ensure_ascii=False)
print(reply); print("saved", rec_path)
