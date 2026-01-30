import Levenshtein
import unicodedata
import re

def normalize_bengali(text):
    if not isinstance(text, str):
        return ""
    # NFKC is the research standard for Bengali to handle conjuncts (যুক্তবর্ণ)
    text = unicodedata.normalize('NFKC', text)
    # Thesis Tip: Remove all whitespace for a "Pure Character" comparison
    text = re.sub(r'\s+', '', text)
    return text.strip()

def cer(pred, gt):
    pred_norm = normalize_bengali(pred)
    gt_norm = normalize_bengali(gt)
    
    # Standard Research Formula: Distance / Length of Reference
    if len(gt_norm) == 0:
        return 1.0 if len(pred_norm) > 0 else 0.0
        
    return Levenshtein.distance(pred_norm, gt_norm) / len(gt_norm)