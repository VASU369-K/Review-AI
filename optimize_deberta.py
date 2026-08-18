"""
DeBERTa Hyperparameter Optimization Script for Review-AI.

Tries multiple hyperparameter configurations, selects best by validation F1,
then evaluates final model on the held-out test set.
"""

import os
import json
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Define hyperparameter search space (3 configs)
CONFIGS = [
    {
        "name": "config_A",
        "learning_rate": 1e-5,
        "epochs": 2,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "batch_size": 8,
        "max_length": 128,
    },
    {
        "name": "config_B",
        "learning_rate": 2e-5,
        "epochs": 2,
        "warmup_ratio": 0.05,
        "weight_decay": 0.02,
        "batch_size": 16,
        "max_length": 128,
    },
    {
        "name": "config_C",
        "learning_rate": 5e-6,
        "epochs": 3,
        "warmup_ratio": 0.15,
        "weight_decay": 0.01,
        "batch_size": 8,
        "max_length": 128,
    },
]


def run_optimization(dataset_dir="Dataset", output_dir="models", sample_limit=1000, seed=42):
    print("\n" + "=" * 60)
    print("  DeBERTa HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    model_checkpoint = "microsoft/deberta-base"

    # Load dataset
    train_df = pd.read_csv(os.path.join(dataset_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(dataset_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(dataset_dir, "test_split.csv"))

    if sample_limit and sample_limit < len(train_df):
        print(f"Limiting: Train={sample_limit}, Val={int(sample_limit * 0.2)}, Test={int(sample_limit * 0.2)}")
        train_df = train_df.sample(sample_limit, random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(min(int(sample_limit * 0.2), len(val_df)), random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(min(int(sample_limit * 0.2), len(test_df)), random_state=seed).reset_index(drop=True)

    train_samples = len(train_df)
    eval_samples = len(test_df)

    # Phase 1: Evaluate each config on validation set
    optimization_log = {
        "model": "deberta",
        "base_checkpoint": model_checkpoint,
        "sample_limit": sample_limit,
        "seed": seed,
        "configs_evaluated": [],
        "best_config": None,
        "final_test_metrics": None,
    }

    best_val_f1 = -1.0
    best_config = None
    best_val_metrics = None

    for cfg in CONFIGS:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {cfg['name']}")
        print(f"  LR={cfg['learning_rate']}, Epochs={cfg['epochs']}, Warmup={cfg['warmup_ratio']}, WD={cfg['weight_decay']}, BS={cfg['batch_size']}")
        print(f"{'='*50}")

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=2,
            id2label={0: "Negative", 1: "Positive"},
            label2id={"Negative": 0, "Positive": 1}
        )

        max_len = cfg["max_length"]
        train_enc = tokenizer(list(train_df['text'].astype(str)), truncation=True, padding=True, max_length=max_len)
        val_enc = tokenizer(list(val_df['text'].astype(str)), truncation=True, padding=True, max_length=max_len)

        train_dataset = SentimentDataset(train_enc, list(train_df['label'].astype(int)))
        val_dataset = SentimentDataset(val_enc, list(val_df['label'].astype(int)))

        tmp_dir = os.path.join(output_dir, f"deberta_opt_{cfg['name']}")
        os.makedirs(tmp_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed,
            use_cpu=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate on validation
        val_metrics = trainer.evaluate(val_dataset)
        val_f1 = val_metrics.get("eval_f1", 0.0)

        config_result = {
            "config_name": cfg["name"],
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "warmup_ratio": cfg["warmup_ratio"],
            "weight_decay": cfg["weight_decay"],
            "batch_size": cfg["batch_size"],
            "val_f1": round(val_f1, 4),
            "val_accuracy": round(val_metrics.get("eval_accuracy", 0.0), 4),
            "val_loss": round(val_metrics.get("eval_loss", 0.0), 4),
        }
        optimization_log["configs_evaluated"].append(config_result)

        print(f"  >> Val F1: {val_f1*100:.2f}%, Val Acc: {val_metrics.get('eval_accuracy', 0)*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_val_metrics = val_metrics
            # Save this as the best model so far
            model_output_dir = os.path.join(output_dir, "deberta")
            os.makedirs(model_output_dir, exist_ok=True)
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            print(f"  >> NEW BEST! Saved to {model_output_dir}")

        # Cleanup temp dir
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Phase 2: Evaluate best config on test set
    print(f"\n{'='*60}")
    print(f"  BEST CONFIG: {best_config['name']}")
    print(f"  Val F1: {best_val_f1*100:.2f}%")
    print(f"  Now evaluating on HELD-OUT TEST SET...")
    print(f"{'='*60}")

    model_output_dir = os.path.join(output_dir, "deberta")
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_output_dir)

    test_enc = tokenizer(list(test_df['text'].astype(str)), truncation=True, padding=True, max_length=best_config["max_length"])
    test_dataset = SentimentDataset(test_enc, list(test_df['label'].astype(int)))

    test_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_eval_batch_size=best_config["batch_size"],
        use_cpu=not torch.cuda.is_available(),
    )

    test_trainer = Trainer(
        model=model,
        args=test_args,
        compute_metrics=compute_metrics,
    )

    test_metrics = test_trainer.evaluate(test_dataset)
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy: {test_metrics['eval_accuracy']*100:.2f}%")
    print(f"  F1:       {test_metrics['eval_f1']*100:.2f}%")
    print(f"  Precision:{test_metrics['eval_precision']*100:.2f}%")
    print(f"  Recall:   {test_metrics['eval_recall']*100:.2f}%")

    optimization_log["best_config"] = best_config["name"]
    optimization_log["final_test_metrics"] = {
        "accuracy": round(test_metrics["eval_accuracy"], 4),
        "f1": round(test_metrics["eval_f1"], 4),
        "precision": round(test_metrics["eval_precision"], 4),
        "recall": round(test_metrics["eval_recall"], 4),
        "loss": round(test_metrics["eval_loss"], 4),
    }

    # Save optimization log
    log_path = os.path.join(output_dir, "deberta_optimization_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(optimization_log, f, indent=2, ensure_ascii=False)
    print(f"\n[\u2713] Optimization log saved to {log_path}")

    # Update models/metrics.json
    metrics_file = os.path.join(output_dir, "metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            all_metrics = json.load(f)

    training_config = {
        "learning_rate": best_config["learning_rate"],
        "batch_size": best_config["batch_size"],
        "warmup_ratio": best_config["warmup_ratio"],
        "weight_decay": best_config["weight_decay"],
        "max_length": best_config["max_length"],
        "epochs": best_config["epochs"],
        "seed": seed,
        "optimization_method": "grid_search_3_configs",
    }

    all_metrics["deberta"] = {
        "accuracy": round(test_metrics["eval_accuracy"], 4),
        "f1": round(test_metrics["eval_f1"], 4),
        "precision": round(test_metrics["eval_precision"], 4),
        "recall": round(test_metrics["eval_recall"], 4),
        "validation_score": round(best_val_f1, 4),
        "validation_accuracy": round(best_val_metrics.get("eval_accuracy", 0), 4),
        "test_samples": eval_samples,
        "train_samples": train_samples,
        "base_checkpoint": model_checkpoint,
        "training_epochs": best_config["epochs"],
        "eval_loss": round(test_metrics["eval_loss"], 4),
        "trained_at": datetime.datetime.now().isoformat(),
        "seed": seed,
        "training_config": training_config,
        "optimization_log": f"models/deberta_optimization_log.json",
    }

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"[\u2713] metrics.json updated with optimized DeBERTa results")

    # Save metadata
    metadata = {
        "model_name": "deberta",
        "base_checkpoint": model_checkpoint,
        "dataset_source": "Amazon Reviews (Kaggle bittlingmayer/amazonreviews)",
        "label_mapping": {"0": "Negative", "1": "Positive"},
        "training_seed": seed,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "training_epochs": best_config["epochs"],
        "training_config": training_config,
        "optimization": optimization_log,
        "trained_at": datetime.datetime.now().isoformat(),
    }
    metadata_path = os.path.join(model_output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[\u2713] DeBERTa metadata saved to {metadata_path}")

    print(f"\n{'='*60}")
    print(f"  DeBERTa OPTIMIZATION COMPLETE")
    print(f"  Best config: {best_config['name']}")
    print(f"  Test F1: {test_metrics['eval_f1']*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize DeBERTa hyperparameters")
    parser.add_argument("--dataset_dir", type=str, default="Dataset")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--sample_limit", type=int, default=1000, help="Limit training samples (CPU-friendly)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_optimization(args.dataset_dir, args.output_dir, args.sample_limit, args.seed)
