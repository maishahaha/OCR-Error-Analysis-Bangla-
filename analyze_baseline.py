import pandas as pd
import numpy as np

# -----------------------------
# Load baseline results
# -----------------------------
csv_path = "analysis/baseline/baseline_results.csv"
df = pd.read_csv(csv_path)

# Basic sanity
assert {"image", "ground_truth", "prediction", "CER"}.issubset(df.columns)

# -----------------------------
# Overall Metrics
# -----------------------------
mean_cer = df["CER"].mean()
median_cer = df["CER"].median()
zero_cer_pct = (df["CER"] == 0).mean() * 100
worst_5_pct = df["CER"].quantile(0.95)

# -----------------------------
# Error Buckets
# -----------------------------
buckets = {
    "CER = 0": (df["CER"] == 0).sum(),
    "0 < CER ≤ 0.2": ((df["CER"] > 0) & (df["CER"] <= 0.2)).sum(),
    "0.2 < CER ≤ 0.5": ((df["CER"] > 0.2) & (df["CER"] <= 0.5)).sum(),
    "CER > 0.5": (df["CER"] > 0.5).sum(),
}

# -----------------------------
# Save summary
# -----------------------------
summary_path = "analysis/baseline/baseline_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("BASELINE OCR PERFORMANCE SUMMARY\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Total samples: {len(df)}\n\n")

    f.write("Overall Metrics:\n")
    f.write(f"Mean CER: {mean_cer:.4f}\n")
    f.write(f"Median CER: {median_cer:.4f}\n")
    f.write(f"% CER = 0: {zero_cer_pct:.2f}%\n")
    f.write(f"Worst 5% CER ≥: {worst_5_pct:.4f}\n\n")

    f.write("Error Distribution:\n")
    for k, v in buckets.items():
        f.write(f"{k}: {v}\n")

print("✅ baseline_summary.txt created")

# -----------------------------
# Failure Cases (CER > 0.4)
# -----------------------------
failure_df = df[df["CER"] > 0.4].copy()
failure_df.sort_values("CER", ascending=False, inplace=True)

failure_path = "analysis/baseline/failure_cases.csv"
failure_df.to_csv(failure_path, index=False)

print(f"✅ failure_cases.csv created ({len(failure_df)} samples)")
