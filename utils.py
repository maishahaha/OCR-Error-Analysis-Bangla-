import Levenshtein
import unicodedata

def normalize_bengali(text):
    if not isinstance(text, str):
        return ""
    # NFKC normalization handles combined characters and nuktas consistently
    return unicodedata.normalize('NFKC', text).strip()

def cer(pred, gt):
    pred_norm = normalize_bengali(pred)
    gt_norm = normalize_bengali(gt)
    return Levenshtein.distance(pred_norm, gt_norm) / max(len(gt_norm), 1)