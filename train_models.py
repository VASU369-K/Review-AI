import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 scores for evaluation.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class SentimentDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for Sentiment Analysis review tokens and labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_and_evaluate(model_name, dataset_dir, output_dir, epochs=1, batch_size=8, sample_limit=5000):
    """
    Fine-tunes a transformer model on the train/val splits and evaluates on the test split.
    """
    print(f"\n--- Model Pipeline: {model_name} ---")
    
    # Load dataset splits
    train_file = os.path.join(dataset_dir, "train_split.csv")
    val_file = os.path.join(dataset_dir, "val_split.csv")
    test_file = os.path.join(dataset_dir, "test_split.csv")
    
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Split files not found in {dataset_dir}. Run preprocess_dataset.py first.")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # To avoid hours of CPU training, default is a subset sample limit
    if sample_limit and sample_limit < len(train_df):
        print(f"Limiting dataset sizes for demonstration training: Train={sample_limit}, Val={int(sample_limit*0.2)}, Test={int(sample_limit*0.2)}")
        train_df = train_df.sample(sample_limit, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(int(sample_limit * 0.2), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(int(sample_limit * 0.2), random_state=42).reset_index(drop=True)

    # Resolve HF checkpoint path mapped to clean model name
    checkpoint_map = {
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-small"
    }
    
    model_checkpoint = checkpoint_map.get(model_name.lower(), model_name)
    print(f"Loading tokenizer and model from checkpoint: {model_checkpoint}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    # Tokenize data
    print("Tokenizing scripts and text reviews...")
    
    train_encodings = tokenizer(list(train_df['text'].astype(str)), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_df['text'].astype(str)), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(test_df['text'].astype(str)), truncation=True, padding=True, max_length=128)

    train_dataset = SentimentDataset(train_encodings, list(train_df['label'].astype(int)))
    val_dataset = SentimentDataset(val_encodings, list(val_df['label'].astype(int)))
    test_dataset = SentimentDataset(test_encodings, list(test_df['label'].astype(int)))

    # Set up training arguments
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("Configuring PyTorch sequence trainer...")
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(model_output_dir, 'logs'),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # Use CPU if CUDA not available
        no_cuda=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training (fine-tuning)...")
    trainer.train()

    print("Evaluating model on validation data...")
    val_metrics = trainer.evaluate()
    print(f"Validation metrics: {val_metrics}")

    print("Evaluating model on test data...")
    test_metrics = trainer.evaluate(test_dataset)
    print(f"Test metrics: {test_metrics}")

    # Save tokenizer and model
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved successfully to: {model_output_dir}")
    
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune HF models for Sentiment Analysis")
    parser.add_argument("--model", type=str, default="distilbert", choices=["distilbert", "roberta", "deberta"], help="Model type to train")
    parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Directory where processed datasets are located")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save fine-tuned model checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--sample_limit", type=int, default=1000, help="Limit number of train items (to speed up CPU test run)")
    
    args = parser.parse_args()
    train_and_evaluate(args.model, args.dataset_dir, args.output_dir, args.epochs, args.batch_size, args.sample_limit)
