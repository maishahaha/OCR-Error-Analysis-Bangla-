import pandas as pd
import unicodedata
import matplotlib.pyplot as plt

def normalize(text):
    return unicodedata.normalize('NFKC', str(text))

def categorize_error(row):
    gt = normalize(row['ground_truth'])
    pred = normalize(row['prediction'])
    conf = row['confidence']
    
    # 1. Hallucination: Prediction contains lots of non-Bengali characters
    bengali_chars = [c for c in pred if '\u0980' <= c <= '\u09FF']
    if len(pred) > 0 and (len(bengali_chars) / len(pred)) < 0.5:
        return "Hallucination/Noise"
    
    # 2. Length Mismatch: Prediction is significantly longer/shorter than GT
    if abs(len(gt) - len(pred)) > len(gt):
        return "Severe Over/Under Recognition"
    
    # 3. Low Confidence but legible
    if conf < 0.3:
        return "Low Confidence Uncertainty"
    
    return "Linguistic/Modifier Error"

# Load the failure cases
df = pd.read_csv('analysis/baseline/failure_cases.csv')

# Apply categorization
df['error_category'] = df.apply(categorize_error, axis=1)

# Summary Stats
summary = df['error_category'].value_counts()
print("FAILURE ANALYSIS SUMMARY")
print("=========================")
print(summary)

# Correlation between Confidence and CER
plt.figure(figsize=(10, 6))
plt.scatter(df['confidence'], df['CER'], alpha=0.5, color='red')
plt.title('Confidence vs. Character Error Rate (Failures)')
plt.xlabel('Confidence Score')
plt.ylabel('CER')
plt.grid(True)
plt.savefig('analysis/failure_correlation.png')

print("\nAnalysis complete. Correlation plot saved to analysis/failure_correlation.png")

# Save the text summary
with open('analysis/failure_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("FAILURE ANALYSIS SUMMARY\n")
    f.write("=========================\n")
    f.write(summary.to_string())
    f.write("\n\nNote: Categories based on automated heuristic analysis.")

print("✅ Analysis saved to analysis/failure_analysis.txt")