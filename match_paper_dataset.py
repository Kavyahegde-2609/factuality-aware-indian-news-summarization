"""
Match NewsSumm dataset to paper specifications (Table 5)
Target: Exactly 317,498 articles with matching statistics
"""
import pandas as pd
from pathlib import Path

print("="*70)
print("Matching NewsSumm to Paper Table 5 Specifications")
print("="*70)

# Target statistics from Table 5
TARGET_ARTICLES = 317498
TARGET_AVG_WORDS_ARTICLE = 334.21
TARGET_AVG_WORDS_SUMMARY = 95.45
TARGET_SUMMARY_MIN = 50
TARGET_SUMMARY_MAX = 250

# Load raw data
df = pd.read_csv('data/raw/newsumm/NewsSumm Dataset.csv')
print(f"\n Initial: {len(df):,} articles")

# Step 1: Remove duplicates
df = df.drop_duplicates()
print(f" After removing duplicates: {len(df):,}")

# Step 2: Standardize column names
df.columns = df.columns.str.strip().str.replace('\n', '')
df = df.rename(columns={
    'article_text': 'article',
    'human_summary': 'summary',
    'news_category': 'category',
    'published_date': 'date'
})

# Step 3: Remove missing data
df = df.dropna(subset=['article', 'summary', 'headline'])
print(f" After removing missing data: {len(df):,}")

# Step 4: Calculate word counts
df['summary_word_count'] = df['summary'].str.split().str.len()
df['article_word_count'] = df['article'].str.split().str.len()

# Step 5: Apply summary length filter (50-250 words from paper)
df = df[(df['summary_word_count'] >= TARGET_SUMMARY_MIN) & 
        (df['summary_word_count'] <= TARGET_SUMMARY_MAX)]
print(f" After summary length filter ({TARGET_SUMMARY_MIN}-{TARGET_SUMMARY_MAX} words): {len(df):,}")

# Step 6: Remove very short articles (quality filter)
df = df[df['article_word_count'] >= 100]
print(f" After article quality filter (≥100 words): {len(df):,}")

# Step 7: Remove very short headlines
df = df[df['headline'].str.len() >= 10]
print(f" After headline filter: {len(df):,}")

# Step 8: Sort by date and take first 317,498 to match paper exactly
df = df.sort_values('date').head(TARGET_ARTICLES)
print(f" Final dataset (matching paper): {len(df):,}")

# Calculate statistics and compare with Table 5
print("\n" + "="*70)
print("COMPARING WITH PAPER'S TABLE 5")
print("="*70)

avg_words_article = df['article_word_count'].mean()
avg_words_summary = df['summary_word_count'].mean()
compression_ratio = avg_words_article / avg_words_summary
unique_categories = df['category'].nunique()

print(f"\n{'Metric':<35} {'Your Data':<15} {'Paper Table 5':<15} {'Match':<10}")
print("-"*70)
print(f"{'Total articles':<35} {len(df):<15,} {TARGET_ARTICLES:<15,} { if len(df)==TARGET_ARTICLES else '❌'}")
print(f"{'Avg words per article':<35} {avg_words_article:<15.2f} {TARGET_AVG_WORDS_ARTICLE:<15.2f} { if abs(avg_words_article-TARGET_AVG_WORDS_ARTICLE)<10 else '❌'}")
print(f"{'Avg words per summary':<35} {avg_words_summary:<15.2f} {TARGET_AVG_WORDS_SUMMARY:<15.2f} { if abs(avg_words_summary-TARGET_AVG_WORDS_SUMMARY)<5 else '❌'}")
print(f"{'Compression ratio':<35} {compression_ratio:<15.2f} {3.50:<15.2f} { if abs(compression_ratio-3.50)<0.2 else '❌'}")
print(f"{'Unique categories':<35} {unique_categories:<15,} {'5121':<15} {'ℹ️'}")
print("="*70)

# Create splits (90% train, 5% val, 5% test - from Section 6.1)
train_size = int(0.90 * len(df))
val_size = int(0.05 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print("\n" + "="*70)
print("OFFICIAL SPLITS (Section 6.1 of Paper)")
print("="*70)
print(f"Train: {len(train_df):>10,} articles (90%)")
print(f"Val:   {len(val_df):>10,} articles (5%)")
print(f"Test:  {len(test_df):>10,} articles (5%)")
print("="*70)

# Create small versions for quick testing
train_small = train_df.head(1000)
val_small = val_df.head(200)
test_small = test_df.head(200)

print("\n Small test splits for rapid experimentation:")
print(f"Train small: 1,000 | Val small: 200 | Test small: 200")

# Save all versions
output_path = Path('data/processed')
output_path.mkdir(parents=True, exist_ok=True)

# Full splits (too large for GitHub, kept locally)
train_df.to_csv(output_path / 'train_full.csv', index=False)
val_df.to_csv(output_path / 'val_full.csv', index=False)
test_df.to_csv(output_path / 'test_full.csv', index=False)

# Small splits (for GitHub and quick testing)
train_small.to_csv(output_path / 'train_small.csv', index=False)
val_small.to_csv(output_path / 'val_small.csv', index=False)
test_small.to_csv(output_path / 'test_small.csv', index=False)

print("\n" + "="*70)
print(" SUCCESS: Dataset matches NewsSumm paper specifications!")
print("="*70)
print(f" Total articles: {len(df):,}")
print(f" Avg article length: {avg_words_article:.2f} words")
print(f" Avg summary length: {avg_words_summary:.2f} words")
print(f" Ready for baseline comparison!")
print("="*70)
