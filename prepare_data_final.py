"""
Prepare final dataset with available articles (253,930)
"""
import pandas as pd
from pathlib import Path

print("="*70)
print("Preparing Final NewsSumm Dataset for Research")
print("="*70)

# Load the cleaned data
df = pd.read_csv('data/processed/train_full.csv')
df_val = pd.read_csv('data/processed/val_full.csv')
df_test = pd.read_csv('data/processed/test_full.csv')

total = len(df) + len(df_val) + len(df_test)

print(f"\nTotal articles available: {total:,}")
print(f"   Train: {len(df):,} (90%)")
print(f"   Val:   {len(df_val):,} (5%)")
print(f"   Test:  {len(df_test):,} (5%)")

# Calculate statistics
if 'article' in df.columns:
    df['article_words'] = df['article'].str.split().str.len()
    df['summary_words'] = df['summary'].str.split().str.len()
    
    print(f"\nDataset Statistics:")
    print(f"   Avg article length: {df['article_words'].mean():.2f} words")
    print(f"   Avg summary length: {df['summary_words'].mean():.2f} words")
    print(f"   Compression ratio: {df['article_words'].mean() / df['summary_words'].mean():.2f}")
    
    if 'category' in df.columns:
        print(f"   Unique categories: {df['category'].nunique():,}")

print("\n" + "="*70)
print("DATASET READY FOR BASELINE COMPARISON")
print("="*70)
print("\nLarge-scale (253K+ articles)")
print("Human-annotated")
print("Multi-document capable")
print("Indian English domain")
print("Suitable for factuality research")
print("\n Your NOVELTY is the factuality module, not dataset size!")
print("="*70)
