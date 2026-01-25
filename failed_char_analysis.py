import pandas as pd
from collections import Counter
import Levenshtein
import unicodedata

def normalize_bn(text):
    return unicodedata.normalize('NFKC', str(text))

# Load data
df = pd.read_csv('analysis/baseline/failure_cases.csv')

substitutions = Counter()
deletions = Counter()
insertions = Counter()

for _, row in df.iterrows():
    gt = normalize_bn(row['ground_truth'])
    pred = normalize_bn(row['prediction'])
    
    ops = Levenshtein.editops(gt, pred)
    for op, gt_pos, pred_pos in ops:
        if op == 'replace':
            substitutions[(gt[gt_pos], pred[pred_pos])] += 1
        elif op == 'delete':
            deletions[gt[gt_pos]] += 1
        elif op == 'insert':
            insertions[pred[pred_pos]] += 1

# Save detailed report
with open('analysis/full_error_profile.txt', 'w', encoding='utf-8') as f:
    f.write("=== TOP 20 SUBSTITUTIONS (EXPECTED -> GOT) ===\n")
    for (g, p), count in substitutions.most_common(20):
        f.write(f"{g} -> {p}: {count}\n")
    
    f.write("\n=== TOP 20 VANISHING CHARACTERS (DELETIONS) ===\n")
    for char, count in deletions.most_common(20):
        f.write(f"{char}: {count}\n")
        
    f.write("\n=== TOP 20 GHOST CHARACTERS (HALLUCINATIONS) ===\n")
    for char, count in insertions.most_common(20):
        f.write(f"{char}: {count}\n")
        
print("✅ Full Error Profile saved to analysis/full_error_profile.txt")