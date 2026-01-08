"""
Baseline Model Evaluation on NewsSumm Dataset
=============================================
Author: Kavya Hegde
Date: January 2025

METHODOLOGY ALIGNMENT:
---------------------
This script implements Steps 1-3 and partial Step 5 of the research methodology:

STEP 1: Dataset Selection
   Source domain: Models pretrained on CNN/DailyMail, BBC News, XSum
   Target domain: NewsSumm (Indian English)

STEP 2: Baseline Model Selection
   Primary: LED (Longformer Encoder-Decoder)
   Alternative: Long-T5  
   Comparison: BART, PEGASUS, T5 variants
   All models: Pretrained checkpoints, NO fine-tuning

STEP 3: Multi-Document Summary Generation
   Articles concatenated with separators
   Long-context input to transformer
   Abstractive summary generation

STEP 5: Evaluation (Partial - ROUGE only here)
  ROUGE-1, ROUGE-2, ROUGE-L
  (BERTScore and QAGS added in separate evaluation scripts)
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime

from config import (
    BASELINE_MODELS,
    DATA_PATHS,
    RESULT_PATHS,
    EVAL_CONFIG,
    MULTI_DOC_CONFIG,
    PROJECT_INFO,
    DEVICE
)


class NewsSummBaselineEvaluator:
    """
    Evaluates baseline models following research methodology Steps 1-3
    """
    
    def __init__(self):
        self.output_dir = RESULT_PATHS['baselines']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rouge_scorer = rouge_scorer.RougeScorer(
            EVAL_CONFIG['rouge_metrics'],
            use_stemmer=EVAL_CONFIG['use_stemmer']
        )
        
        self.device = DEVICE
        
        print(f"\n{'='*80}")
        print(f"BASELINE EVALUATION - Following Research Methodology")
        print(f"{'='*80}")
        print(f" {PROJECT_INFO['author']} | {PROJECT_INFO['institution']}")
        print(f"  Device: {'GPU' if self.device == 0 else 'CPU'}")
        print(f" Dataset: NewsSumm (Indian English)")
        print(f"{'='*80}\n")
    
    def load_test_data(self):
        """Load test dataset"""
        test_file = DATA_PATHS['test_full']
        df = pd.read_csv(test_file)
        
        max_samples = EVAL_CONFIG['test_samples']
        if len(df) > max_samples:
            df = df.head(max_samples)
        
        print(f" Test file: {test_file.name}")
        print(f" Samples: {len(df):,}\n")
        
        return df
    
    def prepare_input_multidoc(self, articles_list, model_id):
        """
        STEP 3 IMPLEMENTATION: Multi-Document Concatenation
        
        Methodology: "Multiple related articles will be grouped together 
        at the event level, concatenated to form a long-context input"
        
        Args:
            articles_list: List of related articles (or single article)
            model_id: Model identifier
        
        Returns:
            Concatenated input string
        """
        
        # Handle single article (backward compatible)
        if isinstance(articles_list, str):
            articles_list = [articles_list]
        
        # Limit number of articles
        max_articles = MULTI_DOC_CONFIG['max_articles_per_event']
        if len(articles_list) > max_articles:
            articles_list = articles_list[:max_articles]
        
        # Concatenate with separators
        separator = MULTI_DOC_CONFIG['doc_separator']
        concatenated = separator.join([str(art) for art in articles_list if pd.notna(art)])
        
        # T5 models need prefix
        if 't5' in model_id.lower():
            concatenated = f"summarize: {concatenated}"
        
        # Truncate if needed
        # Use longer limit for LED/Long-T5
        if 'led' in model_id.lower() or 'long-t5' in model_id.lower():
            max_len = EVAL_CONFIG['max_article_length_led']
        else:
            max_len = EVAL_CONFIG['max_article_length']
        
        words = concatenated.split()
        if len(words) > max_len:
            concatenated = ' '.join(words[:max_len])
        
        return concatenated
    
    def prepare_input(self, article, model_id):
        """Single-document wrapper"""
        return self.prepare_input_multidoc([article], model_id)
    
    def evaluate_single_model(self, model_name, model_id, test_df):
        """
        Evaluate one baseline model
        
        METHODOLOGY ALIGNMENT:
        - Step 2: Use pretrained checkpoint (no fine-tuning)
        - Step 3: Generate abstractive summaries
        - Step 5: Calculate ROUGE scores
        """
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"Model ID: {model_id}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Load pretrained model (Step 2: No fine-tuning)
        try:
            print(f" Loading pretrained checkpoint...")
            
            if 'led' in model_id.lower() or 'long-t5' in model_id.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device
                )
            else:
                summarizer = pipeline(
                    "summarization",
                    model=model_id,
                    device=self.device
                )
            
            print(f" Model loaded\n")
            
        except Exception as e:
            print(f" Failed: {e}")
            return None
        
        # Evaluate
        results = []
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        failed = []
        
        print(f" Generating summaries ({len(test_df):,} samples)...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=model_name):
            try:
                # Step 3: Prepare (multi-doc) input
                article = self.prepare_input(row['article'], model_id)
                reference = str(row['summary'])
                
                if not article or not reference:
                    failed.append(idx)
                    continue
                
                # Step 3: Generate abstractive summary
                output = summarizer(
                    article,
                    max_length=EVAL_CONFIG['max_summary_length'],
                    min_length=EVAL_CONFIG['min_summary_length'],
                    num_beams=EVAL_CONFIG['num_beams'],
                    do_sample=EVAL_CONFIG['do_sample'],
                    early_stopping=EVAL_CONFIG['early_stopping']
                )
                
                generated = output[0]['summary_text']
                
                # Step 5: Calculate ROUGE
                scores = self.rouge_scorer.score(reference, generated)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
                
                results.append({
                    'sample_id': int(idx),
                    'article_preview': article[:200] + '...',
                    'reference_summary': reference,
                    'generated_summary': generated,
                    'rouge1': round(scores['rouge1'].fmeasure, 4),
                    'rouge2': round(scores['rouge2'].fmeasure, 4),
                    'rougeL': round(scores['rougeL'].fmeasure, 4),
                })
                
            except Exception as e:
                failed.append(idx)
                if len(failed) <= 5:
                    print(f"\n Error on {idx}: {str(e)[:100]}")
                continue
        
        # Calculate aggregates
        elapsed = time.time() - start_time
        
        aggregate = {
            'model_name': model_name,
            'model_id': model_id,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(test_df),
            'successful_samples': len(results),
            'failed_samples': len(failed),
            'success_rate': round(len(results) / len(test_df) * 100, 2),
            
            'rouge1_mean': round(np.mean(rouge1_scores) * 100, 2),
            'rouge2_mean': round(np.mean(rouge2_scores) * 100, 2),
            'rougeL_mean': round(np.mean(rougeL_scores) * 100, 2),
            'rouge1_std': round(np.std(rouge1_scores) * 100, 2),
            'rouge2_std': round(np.std(rouge2_scores) * 100, 2),
            'rougeL_std': round(np.std(rougeL_scores) * 100, 2),
            
            'evaluation_time_seconds': round(elapsed, 2),
            'time_per_sample': round(elapsed / len(results), 2) if results else 0,
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {model_name}")
        print(f"{'='*80}")
        print(f"Success: {aggregate['successful_samples']}/{aggregate['total_samples']} ({aggregate['success_rate']:.1f}%)")
        print(f"\nROUGE Scores:")
        print(f"  ROUGE-1: {aggregate['rouge1_mean']:.2f} ± {aggregate['rouge1_std']:.2f}")
        print(f"  ROUGE-2: {aggregate['rouge2_mean']:.2f} ± {aggregate['rouge2_std']:.2f}")
        print(f"  ROUGE-L: {aggregate['rougeL_mean']:.2f} ± {aggregate['rougeL_std']:.2f}")
        print(f"\nTiming: {elapsed/3600:.2f} hours ({aggregate['time_per_sample']:.2f}s per sample)")
        print(f"{'='*80}\n")
        
        # Save
        safe_name = model_name.lower().replace('/', '_').replace('-', '_')
        model_dir = self.output_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / 'aggregate_scores.json', 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        with open(model_dir / 'detailed_predictions.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f" Saved to: {model_dir}\n")
        
        return aggregate
    
    def run_all_baselines(self):
        """Evaluate all baseline models"""
        
        print(f"\n{'='*80}")
        print(f"STARTING BASELINE EVALUATION")
        print(f"{'='*80}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models: {len(BASELINE_MODELS)}")
        print(f"Samples: {EVAL_CONFIG['test_samples']:,}")
        print(f"{'='*80}\n")
        
        test_df = self.load_test_data()
        
        all_results = []
        failed_models = []
        
        for i, (model_name, model_id) in enumerate(BASELINE_MODELS.items(), 1):
            print(f"\nProgress: {i}/{len(BASELINE_MODELS)}")
            
            try:
                result = self.evaluate_single_model(model_name, model_id, test_df)
                if result:
                    all_results.append(result)
                else:
                    failed_models.append(model_name)
            except Exception as e:
                print(f"\n FAILED: {model_name}")
                print(f"Error: {str(e)[:200]}")
                failed_models.append(model_name)
        
        # Save summary
        summary = {
            'experiment_metadata': {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'author': PROJECT_INFO['author'],
                'institution': PROJECT_INFO['institution'],
                'num_models_evaluated': len(all_results),
                'num_models_failed': len(failed_models),
                'test_samples': EVAL_CONFIG['test_samples'],
            },
            'results': all_results,
            'failed_models': failed_models,
        }
        
        with open(self.output_dir / 'all_baselines_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.print_comparison(all_results)
        
        if failed_models:
            print(f"\n Failed: {', '.join(failed_models)}")
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison table"""
        
        print(f"\n{'='*90}")
        print(f"BASELINE COMPARISON - INDIAN ENGLISH NEWS")
        print(f"{'='*90}")
        print(f"{'Model':<30} {'R-1':<14} {'R-2':<14} {'R-L':<14} {'Time(h)':<10}")
        print(f"{'-'*90}")
        
        sorted_results = sorted(results, key=lambda x: x['rouge2_mean'], reverse=True)
        
        for r in sorted_results:
            time_h = r['evaluation_time_seconds'] / 3600
            print(f"{r['model_name']:<30} "
                  f"{r['rouge1_mean']:>5.2f}±{r['rouge1_std']:<5.2f} "
                  f"{r['rouge2_mean']:>5.2f}±{r['rouge2_std']:<5.2f} "
                  f"{r['rougeL_mean']:>5.2f}±{r['rougeL_std']:<5.2f} "
                  f"{time_h:>8.2f}")
        
        print(f"{'='*90}")
        
        best = sorted_results[0]
        print(f"\n BEST BASELINE: {best['model_name']}")
        print(f"   ROUGE-2: {best['rouge2_mean']:.2f}%")
        print(f"   → Will be used as base for Step 4 (Factuality Enhancement)")
        print(f"{'='*90}\n")


def main():
    evaluator = NewsSummBaselineEvaluator()
    results = evaluator.run_all_baselines()
    
    print(f"\n BASELINE EVALUATION COMPLETE!")
    print(f" Results: {RESULT_PATHS['baselines']}")
    print(f" Models evaluated: {len(results)}")
    print(f"\n NEXT: Apply Step 4 (Factuality-Aware Module)")


if __name__ == "__main__":
    main()
