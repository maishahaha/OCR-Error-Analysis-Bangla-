import pandas as pd
import os
from tqdm import tqdm
from utils import cer
from ocr_models import ocr_tesseract

# Configuration
IMG_DIR = 'dataset/images'
LABEL_CSV = 'dataset/labels.csv'
OUTPUT_CSV = 'tesseract_results.csv'
SUMMARY_TXT = 'tesseract_summary.txt'

# Load data
df = pd.read_csv(LABEL_CSV)
results = []

print(f"🕵️ Running Tesseract Baseline on {len(df)} images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMG_DIR, row['image_name'])
    gt = row['ground_truth_text'] # Confirmed column name
    
    # Run OCR
    pred, conf = ocr_tesseract(img_path)
    
    # Calculate CER
    error_rate = cer(pred, gt)
    
    results.append({
        'image': row['image_name'],
        'ground_truth': gt,
        'prediction': pred,
        'confidence': round(conf, 4),
        'CER': round(error_rate, 4)
    })

# Save CSV
res_df = pd.DataFrame(results)
res_df.to_csv(OUTPUT_CSV, index=False)

# Generate Summary
mean_cer = res_df['CER'].mean()
median_cer = res_df['CER'].median()
perfect = (res_df['CER'] == 0).sum()

summary_content = f"""TESSERACT PERFORMANCE SUMMARY
========================================
Total Samples: {len(res_df)}

Overall Metrics:
Mean CER: {mean_cer:.4f}
Median CER: {median_cer:.4f}
Accuracy (CER=0): {perfect} ({ (perfect/len(res_df))*100 :.2f}%)

Error Distribution:
CER = 0: {perfect}
0 < CER ≤ 0.2: {len(res_df[(res_df['CER'] > 0) & (res_df['CER'] <= 0.2)])}
0.2 < CER ≤ 0.5: {len(res_df[(res_df['CER'] > 0.2) & (res_df['CER'] <= 0.5)])}
CER > 0.5: {len(res_df[res_df['CER'] > 0.5])}
"""

with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
    f.write(summary_content)

print(f"\n✅ Tesseract baseline complete!")
print(f"Results: {OUTPUT_CSV}")
print(f"Summary: {SUMMARY_TXT}")