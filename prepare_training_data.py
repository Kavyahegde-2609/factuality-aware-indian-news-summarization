"""
Prepare NewsSumm for LED Fine-tuning
Subset for faster training on Kaggle
"""

import pandas as pd
from pathlib import Path

# Load your existing data
train_df = pd.read_csv('data/processed/train_full.csv')
val_df = pd.read_csv('data/processed/val_full.csv')
test_df = pd.read_csv('data/processed/test_full.csv')

print(f"Original sizes:")
print(f"  Train: {len(train_df):,}")
print(f"  Val: {len(val_df):,}")
print(f"  Test: {len(test_df):,}")

# Create training subset (10,000 samples for manageable training)
train_subset = train_df.sample(n=10000, random_state=42)
val_subset = val_df.sample(n=1000, random_state=42)

# Save
train_subset.to_csv('data/processed/train_10k.csv', index=False)
val_subset.to_csv('data/processed/val_1k.csv', index=False)

print(f"\nSubsets created:")
print(f"  Train subset: 10,000 samples → train_10k.csv")
print(f"  Val subset: 1,000 samples → val_1k.csv")
print(f"  Test: {len(test_df):,} samples (unchanged)")

print("\n Ready for fine-tuning!")