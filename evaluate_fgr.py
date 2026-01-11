"""
FGR Evaluation Script
Run this to evaluate FGR and compare with baselines
"""

from factuality_guided_reranking import evaluate_fgr
import json
from pathlib import Path
import pandas as pd


def compare_with_baselines():
    """Compare FGR with baseline models"""
    
    print("\n" + "="*80)
    print("COMPARING FGR WITH BASELINES")
    print("="*80)
    
    # Load baseline results
    baselines_dir = Path('results/baselines')
    baseline_names = [
        'bart_base', 'bart_large_cnn', 'distilbart',
        't5_large', 't5_base', 'pegasus_cnn',
        'longt5_base'
    ]
    
    all_results = []
    
    for name in baseline_names:
        result_file = baselines_dir / name / 'aggregate_scores.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
                print(f" Loaded: {result['model_name']}")
    
    # Load FGR results
    fgr_file = Path('results/proposed_model/fgr_aggregate_scores.json')
    if fgr_file.exists():
        with open(fgr_file, 'r') as f:
            fgr_result = json.load(f)
            all_results.append(fgr_result)
            print(f" Loaded: {fgr_result['model_name']}")
    
    # Sort by ROUGE-2
    all_results.sort(key=lambda x: x.get('rouge2_mean', 0), reverse=True)
    
    # Print comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\n{'Rank':<6}{'Model':<30}{'ROUGE-2':<10}{'Fact':<8}")
    print("-" * 80)
    
    for rank, result in enumerate(all_results, 1):
        model = result['model_name'][:29]
        r2 = result.get('rouge2_mean', 0)
        fact = result.get('factuality_mean', 0)
        
        print(f"{rank:<6}{model:<30}{r2:<10.2f}{fact:<8.3f}")

    
    print("="*80 + "\n")
    
    # Save comparison
    comparison = {
        'comparison_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': all_results
    }
    
    with open('results/comparison_table.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(" Comparison saved to: results/comparison_table.json\n")


def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*80)
    print("FGR COMPREHENSIVE EVALUATION")
    print("="*80)
    print("\nThis will:")
    print("1. Evaluate FGR on 200 NewsSumm samples")
    print("2. Compare with baseline models")
    print("\nEstimated time:")
    print("  - CPU: 4-6 hours")
    print("  - GPU: 1-2 hours")
    print("="*80)
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Aborted.")
        return
    
    # Step 1: Evaluate FGR
    print("\n" + "="*80)
    print("STEP 1: EVALUATING FGR")
    print("="*80)
    aggregate, results = evaluate_fgr(num_samples=200)
    
    # Step 2: Compare
    print("\n" + "="*80)
    print("STEP 2: COMPARING WITH BASELINES")
    print("="*80)
    compare_with_baselines()
    
    # Summary
    print("\n" + "="*80)
    print(" EVALUATION COMPLETE!")
    print("="*80)
    print("\nResults:")
    print(f"  - results/proposed_model/fgr_aggregate_scores.json")
    print(f"  - results/comparison_table.json")
    print("\nNext: Write paper and submit to Springer!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()