"""
Final data preparation for NewsSumm
"""
import pandas as pd
from pathlib import Path

print("="*70)
print("Preparing NewsSumm for Research")
print("="*70)

# Load the CSV
df = pd.read_csv('data/raw/newsumm/NewsSumm Dataset.csv')

print(f"\n✅ Loaded {len(df):,} articles")
print(f"Columns: {df.columns.tolist()}")

# Standardize column names
df.columns = df.columns.str.strip().str.replace('\n', '')
df = df.rename(columns={
    'article_text': 'article',
    'human_summary': 'summary',
    'news_category': 'category',
    'published_date': 'date'
})

# Remove missing data
df = df.dropna(subset=['article', 'summary'])

print(f"\n✅ After cleaning: {len(df):,} articles")

# Create splits
train_size = int(0.90 * len(df))
val_size = int(0.05 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

# For quick experiments, create small versions
train_small = train_df.head(1000)
val_small = val_df.head(200)
test_small = test_df.head(200)

# Save
output_path = Path('data/processed')
output_path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_path / 'train_full.csv', index=False)
val_df.to_csv(output_path / 'val_full.csv', index=False)
test_df.to_csv(output_path / 'test_full.csv', index=False)

train_small.to_csv(output_path / 'train_small.csv', index=False)
val_small.to_csv(output_path / 'val_small.csv', index=False)
test_small.to_csv(output_path / 'test_small.csv', index=False)

print("\n" + "="*70)
print("SPLITS CREATED")
print("="*70)
print(f"Full - Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Small - Train: 1,000 | Val: 200 | Test: 200")
print("="*70)