import json

with open(r'E:\Ace\geometric-evolution\feedforward_results\experiment_20260105_162815.json') as f:
    d = json.load(f)

for t in d['trials']:
    has_intro = 'YES' if t.get('introspection') else 'NO'
    print(f"{t['model']}: introspection={has_intro}")
