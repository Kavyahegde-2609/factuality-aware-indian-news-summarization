"""
Baseline Model Evaluation
Author: Kavya Hegde
Date: January 2025

Evaluates 10 baseline models on NewsSumm to establish baseline performance.
"""

from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import json
from pathlib import Path
import time

from config import BASELINE_MODELS, DATA_PATHS, RESULT_PATHS, EVAL_CONFIG, PROJECT_INFO, DEVICE


class BaselineEvaluator:
    
    def __init__(self):
        self.output_dir = RESULT_PATHS['baselines']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print(f"\n{'='*70}")
        print(f"🚀 {PROJECT_INFO['title']}")
        print(f"👤 {PROJECT_INFO['author']}")
        print(f"🖥️  Device: {'GPU' if DEVICE == 0 else 'CPU'}")
        print(f"{'='*70}\n")
    
    def evaluate_model(self, model_name, model_id, test_df):
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            print("⏳ Loading model...")
            summarizer = pipeline("summarization", model=model_id, device=DEVICE)
            print("✅ Model loaded\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return None
        
        results = []
        r1_scores, r2_scores, rL_scores = [], [], []
        
        print("🔄 Generating summaries...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=model_name):
            try:
                article = str(row['article'])[:EVAL_CONFIG['max_article_length']]
                reference = str(row['summary'])
                
                if 't5' in model_id.lower():
                    article = f"summarize: {article}"
                
                output = summarizer(
                    article,
                    max_length=EVAL_CONFIG['max_summary_length'],
                    min_length=EVAL_CONFIG['min_summary_length'],
                    num_beams=EVAL_CONFIG['num_beams'],
                    do_sample=False
                )
                
                generated = output[0]['summary_text']
                
                scores = self.scorer.score(reference, generated)
                r1_scores.append(scores['rouge1'].fmeasure)
                r2_scores.append(scores['rouge2'].fmeasure)
                rL_scores.append(scores['rougeL'].fmeasure)
                
                results.append({
                    'id': int(idx),
                    'reference': reference,
                    'generated': generated,
                    'rouge1': round(scores['rouge1'].fmeasure, 4),
                    'rouge2': round(scores['rouge2'].fmeasure, 4),
                    'rougeL': round(scores['rougeL'].fmeasure, 4),
                })
                
            except Exception as e:
                print(f"\n⚠️  Error: {str(e)[:80]}")
                continue
        
        elapsed = time.time() - start_time
        
        aggregate = {
            'model_name': model_name,
            'model_id': model_id,
            'samples_evaluated': len(results),
            'rouge1': round(np.mean(r1_scores) * 100, 2),
            'rouge2': round(np.mean(r2_scores) * 100, 2),
            'rougeL': round(np.mean(rL_scores) * 100, 2),
            'rouge1_std': round(np.std(r1_scores) * 100, 2),
            'rouge2_std': round(np.std(r2_scores) * 100, 2),
            'rougeL_std': round(np.std(rL_scores) * 100, 2),
            'time_seconds': round(elapsed, 2),
        }
        
        print(f"\n📊 Results:")
        print(f"   ROUGE-1: {aggregate['rouge1']:.2f} ± {aggregate['rouge1_std']:.2f}")
        print(f"   ROUGE-2: {aggregate['rouge2']:.2f} ± {aggregate['rouge2_std']:.2f}")
        print(f"   ROUGE-L: {aggregate['rougeL']:.2f} ± {aggregate['rougeL_std']:.2f}")
        print(f"   Time: {aggregate['time_seconds']:.1f}s")
        
        model_dir = self.output_dir / model_name.lower().replace('-', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / 'scores.json', 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        with open(model_dir / 'predictions.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return aggregate
    
    def run_all(self):
        test_df = pd.read_csv(DATA_PATHS['test_small'])
        test_df = test_df.head(EVAL_CONFIG['test_samples'])
        
        print(f"📊 Test samples: {len(test_df)}")
        print(f"🤖 Models to evaluate: {len(BASELINE_MODELS)}\n")
        
        all_results = []
        
        for model_name, model_id in BASELINE_MODELS.items():
            try:
                result = self.evaluate_model(model_name, model_id, test_df)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"\n❌ FAILED: {model_name} - {str(e)[:100]}")
        
        with open(self.output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.print_comparison(all_results)
        
        return all_results
    
    def print_comparison(self, results):
        print(f"\n{'='*80}")
        print(f"BASELINE COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'R-1':<10} {'R-2':<10} {'R-L':<10} {'Time(s)':<10}")
        print(f"{'-'*80}")
        
        results = sorted(results, key=lambda x: x['rouge2'], reverse=True)
        
        for r in results:
            print(f"{r['model_name']:<20} "
                  f"{r['rouge1']:>5.2f}±{r['rouge1_std']:<3.2f} "
                  f"{r['rouge2']:>5.2f}±{r['rouge2_std']:<3.2f} "
                  f"{r['rougeL']:>5.2f}±{r['rougeL_std']:<3.2f} "
                  f"{r['time_seconds']:>8.1f}")
        
        print(f"{'='*80}")
        print(f"\n🏆 Best: {results[0]['model_name']} (ROUGE-2: {results[0]['rouge2']:.2f})")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    results = evaluator.run_all()
    
    print("\n✅ Baseline evaluation complete!")
    print(f"📁 Results saved in: results/baselines/")