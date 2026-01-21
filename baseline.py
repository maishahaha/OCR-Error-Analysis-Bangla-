import pandas as pd
import os
from utils import cer
from ocr_models import ocr_trocr   # import your OCR function

# Paths
dataset_path = 'dataset/images'
labels_csv = 'dataset/labels.csv'

# Load labels
labels = pd.read_csv(labels_csv)

results = []

for idx, row in labels.iterrows():
    image_path = os.path.join(dataset_path, row['image_name'])
    gt_text = row['ground_truth_text']
    
    # Catch both values
    pred_text, confidence = ocr_trocr(image_path)
    
    # Calculate CER (using your normalized utils function)
    error_rate = cer(pred_text, gt_text)
    
    results.append({
        'image': row['image_name'],
        'ground_truth': gt_text,
        'prediction': pred_text,
        'confidence': confidence,
        'CER': error_rate
    })

# Save baseline results
pd.DataFrame(results).to_csv('baseline_results.csv', index=False)

print("Baseline OCR completed! Results saved to baseline_results.csv")
