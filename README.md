# Review-AI: AI-Powered Customer Review Sentiment Analysis & Business Intelligence Agent

A production-ready full-stack application that leverages fine-tuned transformer models (DistilBERT, RoBERTa, DeBERTa) to analyze customer review sentiment, generate aspect-based business intelligence reports, and provide actionable recommendations through an AI agent interface.

## Project Architecture

```
Review-AI/
├── main.py                     # FastAPI backend: inference, BI, agent, export
├── agent.py                    # AI Agent orchestrator (NL query routing)
├── train_models.py             # Training pipeline (HuggingFace Trainer)
├── optimize_deberta.py         # DeBERTa hyperparameter optimization
├── preprocess_dataset.py       # Dataset cleaning + 70/15/15 split
├── convert_to_csv.py           # fastText → CSV converter
├── test_all_features.py        # End-to-end test suite (9 tests)
├── models/
│   ├── distilbert/             # Fine-tuned DistilBERT checkpoint
│   ├── roberta/                # Fine-tuned RoBERTa checkpoint
│   ├── deberta/                # Fine-tuned DeBERTa checkpoint
│   └── metrics.json            # Evaluation metrics for all models
├── Dataset/
│   ├── train_split.csv         # 7,000 training samples
│   ├── val_split.csv           # 1,500 validation samples
│   └── test_split.csv          # 1,500 held-out test samples
├── static/
│   ├── index.html              # Dashboard UI (5 tabs)
│   ├── app.js                  # Frontend logic (Chart.js, API calls)
│   └── styles.css              # Glassmorphism dark theme
└── README.md
```

## Key Features

### 1. Fine-Tuned Transformer Models
- **DistilBERT** (`distilbert-base-uncased`): Lightweight, fast inference
- **RoBERTa** (`roberta-base`): Robust performance on noisy text
- **DeBERTa** (`microsoft/deberta-base`): Disentangled attention, highest accuracy
- All models trained on Amazon Reviews dataset with strict train/val/test separation (70/15/15)
- No heuristic fallbacks — all sentiment predictions come exclusively from fine-tuned models

### 2. AI Agent (Natural Language Query Interface)
- Keyword-scored tool routing for 5 analysis tools:
  - **Sentiment Analysis** — distribution, ratios, satisfaction
  - **Aspect Analysis** — quality, pricing, delivery, usability, support
  - **Business Intelligence** — executive summary with executive insight
  - **Model Metrics** — accuracy, F1, precision, recall comparisons
  - **Recommendations** — data-driven actionable business advice
- Supports complex queries: "Compare all models", "Which model should I deploy?", "Why is DeBERTa better?"

### 3. Business Intelligence Dashboard
- Executive Insight panel with dynamic overall sentiment, major complaint, strongest positive aspect, and recommended action
- Aspect Mention Analysis with positive/negative breakdowns and stacked bar charts
- Failure accounting: tracks attempted, successful, and failed predictions separately
- CSV/JSON export with model evaluation metrics, executive insight, and recommendations

### 4. Interactive Dashboard (5 Tabs)
| Tab | Description |
|-----|-------------|
| **Overview** | KPIs, executive insight, aspect chart, recommendations, keywords |
| **AI Agent** | Natural language query interface with example chips |
| **Review Analysis** | Single review inference + CSV batch upload |
| **Model Comparison** | Side-by-side metrics for all 3 models with best-model indicator |
| **Business Insights** | Detailed aspect breakdowns, positive/negative aspect bars |

## Model Performance Results

| Model | Accuracy | F1-Score | Precision | Recall | Test Samples | Epochs |
|-------|----------|----------|-----------|--------|--------------|--------|
| DistilBERT | 90.0% | 90.7% | 89.8% | 91.5% | 200 | 1 |
| RoBERTa | 91.5% | 91.5% | 96.8% | 86.8% | 200 | 1 |
| **DeBERTa** | **91.5%** | **91.9%** | **92.4%** | **91.5%** | 200 | 1 |

> DeBERTa achieves the best F1 score (91.9%), indicating the best balance between precision and recall.

## Dataset Pipeline

1. **Source**: Amazon Reviews (Kaggle `bittlingmayer/amazonreviews`) — 3.6M train, 400K test reviews
2. **Conversion**: `convert_to_csv.py` converts fastText format to CSV
3. **Preprocessing**: `preprocess_dataset.py` cleans text, removes duplicates, maps labels to binary (0/1), and splits data:
   - **Train**: 70% (7,000 samples)
   - **Validation**: 15% (1,500 samples) — used for early stopping and F1-based model selection
   - **Test**: 15% (1,500 samples) — held-out, used only for final evaluation
4. **No data leakage**: Train, validation, and test sets are strictly separated

## Training Pipeline

### Standard Training
```bash
# Train a single model
python train_models.py --model distilbert --epochs 2 --sample_limit 5000

# Train all three models
python train_models.py --model all --epochs 2 --sample_limit 5000
```

### DeBERTa Optimization
```bash
# Run hyperparameter optimization (3 configs, validation F1 selection)
python optimize_deberta.py --sample_limit 1000
```

The optimization script:
1. Evaluates 3 hyperparameter configurations (varying LR, epochs, warmup, weight_decay)
2. Selects the best config by validation F1 score
3. Evaluates the final model on the held-out test set
4. Saves results to `models/deberta_optimization_log.json`

### Training Configuration
Each model's training config is saved in `metrics.json` and includes:
- Learning rate, batch size, warmup ratio, weight decay
- Max sequence length, epochs, random seed
- Validation F1 score and test metrics

## Setup & Running

### Prerequisites
```bash
pip install torch transformers fastapi uvicorn pandas scikit-learn
```

### Quick Start
```bash
# 1. Preprocess dataset (if not already done)
python preprocess_dataset.py

# 2. Train models (skip if checkpoints exist in models/)
python train_models.py --model all --epochs 1 --sample_limit 1000

# 3. Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Open dashboard
# Navigate to http://localhost:8000
```

### Testing
```bash
# With server running:
python test_all_features.py
```

The test suite validates 9 categories:
1. Mixed review inference (RoBERTa)
2. Strict model validation (no heuristic fallback)
3. CSV batch upload with failure stats
4. Metrics endpoint (`_meta.best_model`, all model fields)
5. BI failure accounting (`total_attempted`, `total_processed`, `failed_count`) + executive insight
6. JSON export (includes `model_evaluation` + `executive_insight`)
7. CSV export (includes Executive Insight + Model Evaluation sections)
8. AI Agent intent routing (8 query patterns including compare/deploy)
9. Model status endpoint (all 3 models available)

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/analyze` | Single review sentiment analysis |
| `POST` | `/api/batch-upload` | CSV batch analysis |
| `GET` | `/api/bi-report` | Business intelligence report |
| `POST` | `/api/agent` | AI Agent natural language query |
| `GET` | `/api/metrics` | Model evaluation metrics |
| `GET` | `/api/model-status` | Model availability status |
| `GET` | `/api/export` | Export report as JSON or CSV |

## Tech Stack
- **Backend**: FastAPI + Uvicorn
- **ML**: HuggingFace Transformers, PyTorch, scikit-learn
- **Frontend**: HTML5, CSS3 (glassmorphism dark theme), JavaScript, Chart.js
- **Dataset**: Amazon Reviews (binary sentiment classification)
