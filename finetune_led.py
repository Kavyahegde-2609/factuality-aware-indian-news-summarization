"""
Fine-tune LED on NewsSumm
This creates your NOVEL MODEL: LED-NewsSumm
"""

from transformers import (
    LEDTokenizer, 
    LEDForConditionalGeneration, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import json

class NewsSummDataset(Dataset):
    """Custom dataset for NewsSumm"""
    
    def __init__(self, csv_path, tokenizer, max_input=4096, max_target=150):
        # Using 4096 instead of 16384 for faster training
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        
        print(f"  Loaded {len(self.data):,} samples from {csv_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Prepare input
        article = str(row['article'])
        summary = str(row['summary'])
        
        # Tokenize
        inputs = self.tokenizer(
            article,
            max_length=self.max_input,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            summary,
            max_length=self.max_target,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding with -100)
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }


def fine_tune_led():
    """
    Main fine-tuning function
    YOUR NOVEL CONTRIBUTION
    """
    
    print("\n" + "="*80)
    print("FINE-TUNING LED ON NEWSSUMM")
    print("="*80)
    print("\nThis creates: LED-NewsSumm (your novel model)")
    print("="*80)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("\n WARNING: GPU not available!")
        print("Please run this on Kaggle with GPU P100 enabled!")
        print("Training on CPU will take DAYS!")
        return
    
    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print("\n Loading LED-ArXiv (base model)...")
    tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384-arxiv')
    model = LEDForConditionalGeneration.from_pretrained('allenai/led-large-16384-arxiv')
    
    # Load datasets
    print("\n Loading NewsSumm data...")
    train_dataset = NewsSummDataset('data/processed/train_10k.csv', tokenizer)
    val_dataset = NewsSummDataset('data/processed/val_1k.csv', tokenizer)
    
    # Training arguments
    output_dir = Path('models/led_newssumm_finetuned')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        # Training config
        num_train_epochs=3,
        per_device_train_batch_size=1,  # LED is very large
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch = 8
        
        # Learning
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=500,
        
        # Evaluation & saving
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,  # Only keep 2 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        
        # Optimization
        fp16=True,  # Mixed precision for faster training
        dataloader_num_workers=2,
        
        # Logging
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        report_to='none',  # Don't report to wandb/tensorboard
        
        # Other
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors='pt'
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print("\nEstimated time:")
    print("  - 10,000 samples, 3 epochs, batch=8")
    print("  - ~6-8 hours on P100 GPU")
    print("\nProgress:")
    print("="*80 + "\n")
    
    try:
        trainer.train()
        
        # Save final model
        final_dir = Path('models/led_newssumm_final')
        final_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n Saving fine-tuned model...")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Save training info
        training_info = {
            'model': 'LED-NewsSumm',
            'base_model': 'allenai/led-large-16384-arxiv',
            'training_data': 'NewsSumm (10,000 samples)',
            'epochs': 3,
            'status': 'complete'
        }
        
        with open(final_dir / 'training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print("\n" + "="*80)
        print(" FINE-TUNING COMPLETE!")
        print("="*80)
        print(f"\nModel saved to: {final_dir}")
        print("\nThis is your NOVEL MODEL: LED-NewsSumm")
        print("First LED fine-tuned specifically for Indian English news!")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False


if __name__ == "__main__":
    success = fine_tune_led()
    
    if success:
        print("\n SUCCESS! Your novel model is ready!")
        print("Next: Run evaluation to compare with baselines")
    else:
        print("\n Training failed. Check errors above.")