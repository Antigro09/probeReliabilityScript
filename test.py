import json
from scipy.stats import spearmanr

rows = []
with open('results/benchmark/EleutherAI_pythia-160m_sst2.jsonl') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

for arch in ['linear', 'mlp', 'mka']:
    arch_rows = [r for r in rows if r['arch'] == arch]
    As = [r['A'] for r in arch_rows]
    Rs = [r['R'] for r in arch_rows]
    rho, p = spearmanr(As, Rs)
    print(f"{arch}: rho={rho:.3f} p={p:.4f} n={len(arch_rows)}")