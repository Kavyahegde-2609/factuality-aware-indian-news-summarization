"""
Resume-Capable Baseline Evaluation with LED Optimization
========================================================
Can resume from saved checkpoints without losing progress
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LEDTokenizer, LEDForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime
import torch

from config import (
    BASELINE_MODELS,
    DATA_PATHS,
    RESULT_PATHS,
    EVAL_CONFIG,
    MULTI_DOC_CONFIG,
    PROJECT_INFO
)


class ResumableEvaluator:
    """Evaluator that can save progress and resume from checkpoints"""
    
    def __init__(self):
        self.output_dir = RESULT_PATHS['baselines']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rouge_scorer = rouge_scorer.RougeScorer(
            EVAL_CONFIG['rouge_metrics'],
            use_stemmer=EVAL_CONFIG['use_stemmer']
        )
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f" Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(f" Using CPU")
        
        print(f"\n{'='*80}")
        print(f"RESUMABLE BASELINE EVALUATION - OPTIMIZED")
        print(f"{'='*80}\n")
    
    def get_checkpoint_path(self, model_name):
        """Get checkpoint file path for a model"""
        safe_name = model_name.lower().replace('/', '_').replace('-', '_')
        model_dir = self.output_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / 'checkpoint.json'
    
    def load_checkpoint(self, model_name):
        """Load checkpoint if exists"""
        checkpoint_path = self.get_checkpoint_path(model_name)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            print(f" Found checkpoint: {checkpoint['completed_samples']} samples completed")
            return checkpoint
        return None
    
    def save_checkpoint(self, model_name, results, completed_idx):
        """Save progress checkpoint"""
        checkpoint_path = self.get_checkpoint_path(model_name)
        checkpoint = {
            'model_name': model_name,
            'completed_samples': completed_idx + 1,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def get_optimal_padding_length(self, length, attention_window=512):
        """Calculate optimal padding for LED"""
        if length <= attention_window:
            return attention_window
        return ((length // attention_window) + 1) * attention_window
    
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
        """Multi-document concatenation"""
        if isinstance(articles_list, str):
            articles_list = [articles_list]
        
        max_articles = MULTI_DOC_CONFIG['max_articles_per_event']
        if len(articles_list) > max_articles:
            articles_list = articles_list[:max_articles]
        
        separator = MULTI_DOC_CONFIG['doc_separator']
        concatenated = separator.join([str(art) for art in articles_list if pd.notna(art)])
        
        if 't5' in model_id.lower():
            concatenated = f"summarize: {concatenated}"
        
        return concatenated
    
    def load_model_optimized(self, model_id):
        """Load model with LED optimization"""
        is_led = 'led' in model_id.lower()
        
        print(f" Loading: {model_id}")
        
        if is_led:
           tokenizer = LEDTokenizer.from_pretrained(model_id)
           model = LEDForConditionalGeneration.from_pretrained(model_id)
    
    # Keep original attention window for compatibility
           attention_window = model.config.attention_window[0] if isinstance(model.config.attention_window, list) else model.config.attention_window
           print(f"   LED loaded with attention window: {attention_window}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        model = model.to(self.device)
        model.eval()
        
        print(f" Model loaded on {self.device}\n")
        
        return model, tokenizer, is_led
    
    def generate_summary_optimized(self, model, tokenizer, article, is_long_context=False):
        """Generate summary with optimization"""
        max_input_length = 16384 if is_long_context else 1024
        
        inputs = tokenizer(
            article,
            max_length=max_input_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        actual_length = inputs['input_ids'].shape[1]
        
        if is_long_context and 'led' in tokenizer.__class__.__name__.lower():
            optimal_length = self.get_optimal_padding_length(actual_length, attention_window=512)
            
            if actual_length < optimal_length:
                pad_length = optimal_length - actual_length
                inputs['input_ids'] = torch.nn.functional.pad(
                    inputs['input_ids'], (0, pad_length), value=tokenizer.pad_token_id
                )
                inputs['attention_mask'] = torch.nn.functional.pad(
                    inputs['attention_mask'], (0, pad_length), value=0
                )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=EVAL_CONFIG['max_summary_length'],
                min_length=EVAL_CONFIG['min_summary_length'],
                num_beams=EVAL_CONFIG['num_beams'],
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=EVAL_CONFIG['early_stopping']
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary, actual_length, inputs['input_ids'].shape[1]
    
    def evaluate_single_model(self, model_name, model_id, test_df, resume=True):
        """Evaluate one model with resume capability"""
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"Model ID: {model_id}")
        print(f"{'='*80}")
        
        checkpoint = None
        start_idx = 0
        results = []
        
        if resume:
            checkpoint = self.load_checkpoint(model_name)
            if checkpoint:
                results = checkpoint['results']
                start_idx = checkpoint['completed_samples']
                print(f" Resuming from sample {start_idx}")
                
                if start_idx >= len(test_df):
                    print(f" Already completed!")
                    return self.load_final_results(model_name)
        
        start_time = time.time()
        
        try:
            model, tokenizer, is_long_context = self.load_model_optimized(model_id)
        except Exception as e:
            print(f" Failed to load model: {e}")
            return None
        
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        failed = []
        
        total_input_length = sum(r.get('input_length', 0) for r in results)
        total_padded_length = sum(r.get('padded_length', 0) for r in results)
        
        print(f"Generating summaries ({start_idx}/{len(test_df)} → {len(test_df)})...")
        
        for idx in tqdm(range(start_idx, len(test_df)), desc=model_name, initial=start_idx, total=len(test_df)):
            try:
                row = test_df.iloc[idx]
                article = self.prepare_input_multidoc([row['article']], model_id)
                reference = str(row['summary'])
                
                if not article or not reference:
                    failed.append(idx)
                    continue
                
                generated, input_len, padded_len = self.generate_summary_optimized(
                    model, tokenizer, article, is_long_context
                )
                
                total_input_length += input_len
                total_padded_length += padded_len
                
                scores = self.rouge_scorer.score(reference, generated)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
                
                result = {
                    'sample_id': int(idx),
                    'article_preview': article[:200] + '...',
                    'reference_summary': reference,
                    'generated_summary': generated,
                    'input_length': input_len,
                    'padded_length': padded_len,
                    'rouge1': round(scores['rouge1'].fmeasure, 4),
                    'rouge2': round(scores['rouge2'].fmeasure, 4),
                    'rougeL': round(scores['rougeL'].fmeasure, 4),
                }
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    self.save_checkpoint(model_name, results, idx)
                
            except Exception as e:
                failed.append(idx)
                if len(failed) <= 5:
                    print(f"\n Error on {idx}: {str(e)[:100]}")
                continue
        
        elapsed = time.time() - start_time
        
        if checkpoint:
            for r in checkpoint['results']:
                rouge1_scores.append(r['rouge1'])
                rouge2_scores.append(r['rouge2'])
                rougeL_scores.append(r['rougeL'])
        
        avg_input_len = total_input_length / len(results) if results else 0
        avg_padded_len = total_padded_length / len(results) if results else 0
        padding_overhead = ((avg_padded_len - avg_input_len) / avg_input_len * 100) if avg_input_len > 0 else 0
        
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
            'time_per_sample': round(elapsed / (len(results) - start_idx), 2) if (len(results) - start_idx) > 0 else 0,
            'avg_input_length': round(avg_input_len, 0),
            'avg_padded_length': round(avg_padded_len, 0),
            'padding_overhead_percent': round(padding_overhead, 1),
        }
        
        print(f"\n{'='*80}")
        print(f"RESULTS: {model_name}")
        print(f"{'='*80}")
        print(f"Success: {aggregate['successful_samples']}/{aggregate['total_samples']} ({aggregate['success_rate']:.1f}%)")
        print(f"\nROUGE Scores:")
        print(f"  ROUGE-1: {aggregate['rouge1_mean']:.2f} ± {aggregate['rouge1_std']:.2f}")
        print(f"  ROUGE-2: {aggregate['rouge2_mean']:.2f} ± {aggregate['rouge2_std']:.2f}")
        print(f"  ROUGE-L: {aggregate['rougeL_mean']:.2f} ± {aggregate['rougeL_std']:.2f}")
        print(f"\nEfficiency:")
        print(f"  Time: {elapsed/3600:.2f} hours ({aggregate['time_per_sample']:.2f}s per sample)")
        print(f"  Padding overhead: {aggregate['padding_overhead_percent']:.1f}%")
        print(f"{'='*80}\n")
        
        safe_name = model_name.lower().replace('/', '_').replace('-', '_')
        model_dir = self.output_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / 'aggregate_scores.json', 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        with open(model_dir / 'detailed_predictions.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        checkpoint_path = self.get_checkpoint_path(model_name)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        print(f" Saved to: {model_dir}\n")
        
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return aggregate
    
    def load_final_results(self, model_name):
        """Load final results for completed model"""
        safe_name = model_name.lower().replace('/', '_').replace('-', '_')
        model_dir = self.output_dir / safe_name
        
        with open(model_dir / 'aggregate_scores.json', 'r') as f:
            return json.load(f)
    
    def run_all_baselines(self, resume=True):
        """Evaluate all baselines with resume support"""
        
        print(f"\n{'='*80}")
        print(f"BASELINE EVALUATION (RESUME MODE: {resume})")
        print(f"{'='*80}\n")
        
        test_df = self.load_test_data()
        
        all_results = []
        failed_models = []
        
        for i, (model_name, model_id) in enumerate(BASELINE_MODELS.items(), 1):
            print(f"\nProgress: {i}/{len(BASELINE_MODELS)}")
            
            try:
                result = self.evaluate_single_model(model_name, model_id, test_df, resume=resume)
                if result:
                    all_results.append(result)
                else:
                    failed_models.append(model_name)
            except Exception as e:
                print(f"\n FAILED: {model_name}")
                print(f"Error: {str(e)[:200]}")
                failed_models.append(model_name)
        
        summary = {
            'experiment_metadata': {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'author': PROJECT_INFO['author'],
                'num_models_evaluated': len(all_results),
                'num_models_failed': len(failed_models),
                'test_samples': EVAL_CONFIG['test_samples'],
            },
            'results': all_results,
            'failed_models': failed_models,
        }
        
        with open(self.output_dir / 'all_baselines_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return all_results


def main():
    evaluator = ResumableEvaluator()
    
    print("\n" + "="*80)
    print("RESUME OPTIONS:")
    print("="*80)
    print("1. Resume from checkpoints (keep existing progress)")
    print("2. Start fresh (delete all checkpoints)")
    print("="*80)
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    resume = choice == "1"
    
    if not resume:
        confirm = input("⚠ This will delete all progress. Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return
    
    results = evaluator.run_all_baselines(resume=resume)
    
    print(f"\n BASELINE EVALUATION COMPLETE!")
    print(f"✓ Models evaluated: {len(results)}")


if __name__ == "__main__":
    main()
