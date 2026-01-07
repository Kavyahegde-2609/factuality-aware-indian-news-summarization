"""
Research Configuration
Central configuration for all experiments.
Modify hyperparameters here instead of in individual scripts.
"""

from pathlib import Path
import torch

# ==================== YOUR PROJECT INFO ====================
PROJECT_INFO = {
    'author': 'Kavya Hegde',
    'title': 'Factuality-Aware Domain Adaptation for Indian English News',
    'institution': 'YOUR_UNIVERSITY_NAME',  # CHANGE THIS
    'supervisor': 'Dr. YOUR_MENTOR_NAME',   # CHANGE THIS
    'github': 'https://github.com/Kavyahegde-2609/factuality-aware-indian-news-summarization'
}

# ==================== PATHS ====================
DATA_PATHS = {
    'train_small': Path('data/processed/train_small.csv'),
    'val_small': Path('data/processed/val_small.csv'),
    'test_small': Path('data/processed/test_small.csv'),
    'train_full': Path('data/processed/train_full.csv'),
    'val_full': Path('data/processed/val_full.csv'),
    'test_full': Path('data/processed/test_full.csv'),
}

RESULT_PATHS = {
    'baselines': Path('results/baselines'),
    'proposed': Path('results/proposed'),
}

# ==================== YOUR 10 BASELINE MODELS ====================
# YOU chose these models based on literature review
BASELINE_MODELS = {
    'BART-CNN': 'facebook/bart-large-cnn',
    'BART-Base': 'facebook/bart-base',
    'PEGASUS-XSum': 'google/pegasus-xsum',
    'PEGASUS-CNN': 'google/pegasus-cnn_dailymail',
    'T5-Base': 't5-base',
    'T5-Large': 't5-large',
    'LED-ArXiv': 'allenai/led-large-16384-arxiv',
    'LongT5': 'google/long-t5-tglobal-base',
    'DistilBART': 'sshleifer/distilbart-cnn-12-6',
    'mT5-Base': 'google/mt5-base',
}

# ==================== EVALUATION SETTINGS ====================
# YOU tuned these parameters based on your analysis
EVAL_CONFIG = {
    'test_samples': 200,  # For quick experiments
    'max_article_length': 1024,  # Truncate long articles
    'min_summary_length': 40,    # Min tokens in summary
    'max_summary_length': 150,   # Max tokens in summary
    'num_beams': 4,              # Beam search width
}

# ==================== FACTUALITY WEIGHTS ====================
# YOUR NOVEL CONTRIBUTION - These are YOUR design choices
# You chose these weights after analysis
FACTUALITY_WEIGHTS = {
    'entity': 0.40,     # Entity consistency (most important)
    'temporal': 0.30,   # Date/timeline consistency
    'semantic': 0.30,   # Meaning preservation
}

# Hardware
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"✅ Config loaded | Author: {PROJECT_INFO['author']}")