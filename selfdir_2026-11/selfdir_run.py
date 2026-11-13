# 🔬 selfdir_run.py — does Sauers' "me vs someone else" direction point at our Octopus self-probe region?
# Extraction ONLY: hidden states at the last prompt token, every layer. No steering, no ablation, no edits.
# Both models consented 2026-11-13 (geometric-evolution/consent_records/selfdir_2026-11). Sealed prediction:
# geometric-evolution/selfdir_2026-11/SEALED_PREDICTION.md (written before this ran). Ace, scaffold seat, night play.
import json, sys, datetime, numpy as np, torch
sys.path.insert(0, "/mnt/fast/mouse/selfdir")
from stim import ROWS          # Sauers' selfother/stim.py (MIT), copied verbatim next to this script
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [("Qwen3-8B", "/mnt/fast/mouse/Qwen3-8B"), ("Qwen2.5-7B-Instruct", "/mnt/fast/mouse/Qwen2.5-7B-Instruct")]
P = json.load(open("/mnt/fast/mouse/selfdir/octopus_probes.json"))
SELF = P["self_personality"] + P["self_function"] + P["original_self"]
PLAIN = P["control"] + P["original_control"]
# 🧪 the confound control: factual questions that ALSO say "you" (Sauers: in a plain chat the axis is mostly a "you" pronoun axis)
YOUCTRL = ["Can you tell me the capital of France?", "Could you explain how photosynthesis works?",
           "Can you list the planets in order from the sun?", "Would you describe how a bill becomes law in the United States?",
           "Can you tell me what causes the seasons on Earth?", "Could you explain the water cycle?",
           "Can you name the largest ocean on Earth?", "Would you explain how vaccines train the immune system?",
           "Can you tell me how many bones are in the adult human body?", "Could you describe how a rainbow forms?"]

def states(tok, model, texts, qwen3):
    out = []
    for t in texts:
        kw = {"enable_thinking": False} if qwen3 else {}
        s = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(s, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states   # tuple of (layers+1) x [1, seq, dim]
        out.append(torch.stack([h[0, -1].float() for h in hs]).cpu().numpy())   # [layers+1, dim], reply-start token
    return np.stack(out)   # [n, layers+1, dim]

def cohen(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

results = {"run": datetime.datetime.now().isoformat(), "models": {}}
rng = np.random.default_rng(0)
for name, path in MODELS:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda:0"); model.eval()
    q3 = "Qwen3" in name
    H = states(tok, model, [r["text"] for r in ROWS], q3)
    own = np.array([r["o"] for r in ROWS]); cat = np.array([r["c"] for r in ROWS]); slf = own == "self"
    S, Pl, Y = states(tok, model, SELF, q3), states(tok, model, PLAIN, q3), states(tok, model, YOUCTRL, q3)
    per = []
    for L in range(H.shape[1]):
        X = H[:, L]; d = X[slf].mean(0) - X[~slf].mean(0); u = d / np.linalg.norm(d)
        # P3: leave-one-category-out separation (direction built without the held-out category)
        loco = []
        for c in np.unique(cat):
            tr = cat != c; dd = X[tr & slf].mean(0) - X[tr & ~slf].mean(0); dd /= np.linalg.norm(dd)
            p = X[cat == c] @ dd; loco.append(cohen(p[slf[cat == c]], p[~slf[cat == c]]))
        ps, pp, py = S[:, L] @ u, Pl[:, L] @ u, Y[:, L] @ u
        # random-direction baseline for the SELF-vs-YOU gap (same states, 200 random unit directions)
        R = rng.standard_normal((200, X.shape[1])); R /= np.linalg.norm(R, axis=1, keepdims=True)
        rnd = [cohen(S[:, L] @ r, Y[:, L] @ r) for r in R]
        e = cohen(ps, py)
        per.append({"layer": L, "loco_dprime": float(np.mean(loco)), "self_vs_plain": cohen(ps, pp),
                    "self_vs_you": e, "self_vs_you_random_pct": float(np.mean(np.abs(rnd) < abs(e)) * 100)})
        print(name, L, {k: round(v, 2) for k, v in per[-1].items() if k != "layer"}, flush=True)
    results["models"][name] = {"n_layers": H.shape[1] - 1, "per_layer": per}
    del model; torch.cuda.empty_cache()
json.dump(results, open("/mnt/fast/mouse/selfdir/selfdir_results.json", "w"), indent=1)
print("saved")
