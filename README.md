# Review-AI — Customer Review Sentiment Analysis & BI Agent 🚀

An AI-powered **Customer Review Sentiment Analysis and Business Intelligence Agent**, built with **fine-tuned transformer models** (DistilBERT, RoBERTa, DeBERTa) and a professional **interactive dashboard**.

> **No heuristic fallbacks** — all inference runs through fine-tuned Hugging Face models trained on Amazon Reviews data.

---

## 📋 Project Statement

Design and develop an **AI Agent for Customer Review Sentiment Analysis and Business Intelligence Generation** that uses fine-tuned transformer models to classify customer reviews, detect product aspects, generate data-driven business recommendations, and answer natural-language business questions through an AI agent orchestrator.

## 🔍 Problem

Businesses receive thousands of customer reviews but lack an automated, intelligent system to:
- Classify overall sentiment (positive/negative) at scale
- Identify specific product aspects driving satisfaction or complaints
- Generate actionable business recommendations from review data
- Compare multiple ML models to choose the best-performing one
- Allow non-technical stakeholders to ask business questions in natural language

## 🎯 Objectives

1. Fine-tune three transformer models (DistilBERT, RoBERTa, DeBERTa) on real Amazon Reviews data
2. Build a production-ready FastAPI backend for sentiment inference and BI generation
3. Create an AI Agent orchestrator that routes natural-language questions to analysis tools
4. Develop an interactive dashboard with KPI cards, charts, and data-driven recommendations
5. Ensure end-to-end reproducibility with fixed seeds, clear label mappings, and saved metrics

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **Fine-tuned Models** | DistilBERT, RoBERTa, DeBERTa-v3 trained on Amazon Reviews |
| 🤖 **AI Agent** | Ask natural-language business questions; the agent routes to the correct analysis tool |
| 📊 **BI Dashboard** | KPI cards, aspect analysis, stacked charts, keyword clouds, AI recommendations |
| 📋 **Bulk CSV Upload** | Drag-and-drop CSV analysis for up to 500 reviews |
| 📥 **Export Reports** | Download BI reports as JSON or CSV |
| ⚡ **Real Metrics** | Accuracy, F1, Precision, Recall from actual evaluation runs (metrics.json) |
| 💎 **Aspect Analysis** | Automatic detection of Quality, Pricing, Delivery, and Usability themes |
| 🎯 **Actionable Recommendations** | Data-driven business suggestions based on real sentiment patterns |
| 🏆 **Model Comparison** | Side-by-side comparison of all three models with Best Model indicator |
| 🔒 **No Fake Predictions** | Missing models return clear errors instead of fake predictions |

---

## 🏗️ Architecture

```
Review-AI/
├── main.py                  # FastAPI backend (routes, model loading, BI generation)
├── agent.py                 # AI Agent orchestrator (NLQ → tool → answer)
├── train_models.py          # Training pipeline (fine-tune + metrics saving)
├── preprocess_dataset.py    # Data cleaning, deduplication, and train/val/test splitting
├── convert_to_csv.py        # Amazon fastText → CSV converter
├── models/                  # Fine-tuned model checkpoints + metrics.json
│   ├── distilbert/          # Fine-tuned DistilBERT checkpoint
│   ├── roberta/             # Fine-tuned RoBERTa checkpoint
│   ├── deberta/             # Fine-tuned DeBERTa-v3 checkpoint
│   └── metrics.json         # Evaluation metrics for all models
├── Dataset/                 # Preprocessed CSV splits (train/val/test)
│   ├── train.csv            # Raw Amazon Reviews training data
│   ├── test.csv             # Raw Amazon Reviews test data
│   ├── train_split.csv      # 70% training split
│   ├── val_split.csv        # 15% validation split
│   └── test_split.csv       # 15% test/evaluation split
├── static/
│   ├── index.html           # Dashboard UI (5 tabs)
│   ├── app.js               # Frontend logic
│   └── styles.css           # Glassmorphism dark theme
└── README.md
```

---

## 📦 Dataset

**Amazon Reviews** sentiment dataset from Kaggle:
- **Source**: [bittlingmayer/amazonreviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **Format**: fastText format (`__label__1` = Negative, `__label__2` = Positive)
- **Size**: ~4 million reviews (train: 3.6M, test: 400K)
- **Label Mapping**: `1 → 0 (Negative)`, `2 → 1 (Positive)`

### Preprocessing Pipeline

1. **convert_to_csv.py** — Converts fastText `.txt` files to CSV format
2. **preprocess_dataset.py** — Cleans text (HTML removal, whitespace normalization), removes duplicates, maps labels, and splits into 70/15/15 (train/val/test) with fixed random seed (42)

---

## 🧠 Models

| Model | Base Checkpoint | Parameters | Use Case |
|---|---|---|---|
| **DistilBERT** | `distilbert-base-uncased` | 66M | Fast inference, resource-constrained environments |
| **RoBERTa** | `roberta-base` | 125M | High accuracy, rich BI insights |
| **DeBERTa** | `microsoft/deberta-base` | 139M | Highest accuracy, disentangled attention |

All models are fine-tuned for binary sentiment classification (Positive/Negative) with:
- `id2label`: `{0: "Negative", 1: "Positive"}`
- `label2id`: `{"Negative": 0, "Positive": 1}`
- Sequence length: 128 tokens
- Optimizer: AdamW with weight decay 0.01 and 100 warmup steps

---

## 🏋️ Training Pipeline

The training script (`train_models.py`) uses Hugging Face `Trainer` API:

1. Load preprocessed train/val/test splits
2. Tokenize with model-specific tokenizer (max_length=128)
3. Fine-tune with `TrainingArguments` (eval at each epoch, save best model by accuracy)
4. Evaluate on held-out test split
5. Save model checkpoint, tokenizer, metrics, and metadata

### Saved Artifacts Per Model

- `model.safetensors` — Model weights
- `config.json` — Model configuration with label mapping
- `tokenizer.json` — Tokenizer
- `metadata.json` — Training metadata (seed, samples, epochs, timestamp, metrics)

### Global Metrics File

`models/metrics.json` contains evaluation results for all trained models with: accuracy, F1, precision, recall, test sample count, training epochs, seed, and timestamp.

---

## 🤖 AI Agent

The AI Agent (`agent.py`) is a keyword-scored orchestrator that:

1. **Receives** a natural-language business question
2. **Classifies** the question into one of 5 tool categories using weighted keyword matching
3. **Routes** to the appropriate analysis tool
4. **Generates** a structured answer with supporting data and recommendations

### Agent Tools

| Tool | Description | Example Question |
|---|---|---|
| **Sentiment Analysis** | Analyzes sentiment distribution | "What percentage of customers are unhappy?" |
| **Aspect Analysis** | Identifies top praise/complaint areas | "What are customers most unhappy about?" |
| **Business Intelligence** | Comprehensive BI summary | "Give me an overall business summary." |
| **Model Metrics** | Model performance comparison | "Which model performs best?" |
| **Recommendations** | Actionable business suggestions | "What should we do to improve?" |

---

## 📊 Business Intelligence

The BI engine analyzes reviews from the test split and generates:

- **Sentiment distribution** — Positive/Negative counts and percentages
- **Aspect analysis** — Quality & Durability, Pricing & Value, Customer Support & Delivery, Usability & Design
- **Keyword frequency** — Top positive and negative keywords
- **Satisfaction level** — Excellent / Good / Fair / Poor (based on positive ratio)
- **Data-driven recommendations** — Triggered when negative sentiment for an aspect exceeds 40%

---

## 📈 Evaluation Metrics

All metrics come from actual model evaluation on the held-out test split. No values are hard-coded.

The Model Comparison dashboard shows for each model:
- Accuracy, F1-Score, Precision, Recall
- Test sample count
- Base checkpoint name
- Training timestamp
- 🏆 Best Model indicator (by highest F1 score)

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install fastapi uvicorn transformers torch pandas scikit-learn tiktoken
```

### Download Dataset

Download the Amazon Reviews dataset from [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) and place `train.ft.txt` and `test.ft.txt` in the `Dataset/` folder.

---

## ▶️ Running the Project

### Step 1: Convert Dataset (if needed)

```bash
python convert_to_csv.py
```

### Step 2: Preprocess Dataset

```bash
python preprocess_dataset.py
```

### Step 3: Train Models

```bash
# Train all three models (1000 samples each for quick demo)
python train_models.py --model all --sample_limit 1000

# Train a specific model with more samples
python train_models.py --model roberta --sample_limit 5000 --epochs 2
```

### Step 4: Start the Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Single review sentiment analysis |
| `POST` | `/api/batch-analyze` | Batch review analysis (JSON body) |
| `POST` | `/api/batch-upload` | CSV file upload for bulk analysis |
| `GET`  | `/api/bi-report` | Generate Business Intelligence report |
| `GET`  | `/api/metrics` | Get evaluation metrics for all models |
| `GET`  | `/api/model-status` | Check which models are trained |
| `POST` | `/api/agent` | AI Agent (natural-language queries) |
| `GET`  | `/api/export` | Export BI report as JSON or CSV |

### Request/Response Examples

**Single Review Analysis:**
```json
POST /api/analyze
{
  "text": "This product is amazing! Great quality and fast delivery.",
  "model": "roberta"
}

Response:
{
  "text": "This product is amazing!...",
  "model": "roberta",
  "sentiment": "POSITIVE",
  "score": 0.987,
  "confidence": 98.7,
  "aspects": [
    {"aspect": "Quality & Durability", "sentiment": "POSITIVE", "matched_keywords": ["quality"]},
    {"aspect": "Customer Support & Delivery", "sentiment": "POSITIVE", "matched_keywords": ["delivery"]}
  ]
}
```

**AI Agent Query:**
```json
POST /api/agent
{
  "question": "What are customers most unhappy about?",
  "model": "distilbert"
}

Response:
{
  "task": "Aspect Analysis",
  "tool_used": "Aspect Analysis Tool",
  "answer": "\"Quality & Durability\" has the highest negative sentiment at 52.3%...",
  "supporting_data": { ... },
  "recommendations": [ ... ]
}
```

---

## 💬 Example Queries for AI Agent

| Query | Expected Tool |
|---|---|
| "What percentage of customers are unhappy?" | Sentiment Analysis |
| "What are customers most unhappy about?" | Aspect Analysis |
| "What is the biggest customer complaint?" | Aspect Analysis |
| "Which model performs best?" | Model Metrics |
| "Give me business recommendations." | Recommendations |
| "Summarize the overall customer feedback." | Business Intelligence |

---

## 📊 Results

Results are dynamically generated from actual model evaluation. See `models/metrics.json` for the latest metrics after training.

Example (with `--sample_limit 1000`):

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| DistilBERT | 90.0% | 90.7% | 89.8% | 91.5% |
| RoBERTa | 91.5% | 91.5% | 96.8% | 86.8% |
| DeBERTa | *See metrics.json* | — | — | — |

> These are sample results with limited training data. Results improve with `--sample_limit 5000+` and `--epochs 2-3`.

---

## 🔮 Future Scope

1. **Multi-class Sentiment** — Extend to 3-class (Positive/Neutral/Negative) or 5-star rating prediction
2. **Attention-based Explainability** — Highlight which words contributed most to the sentiment decision
3. **Real-time Data Ingestion** — Connect to live review sources (Amazon API, social media)
4. **Generative AI Summaries** — Use an LLM to generate natural-language BI narratives
5. **User Authentication** — Add login and role-based access for enterprise deployment
6. **PDF Report Export** — Generate formatted PDF reports with charts
7. **Fine-tuning with More Data** — Train with 50K+ samples and 3+ epochs for production accuracy

---

## 📋 Technologies

- **Backend**: FastAPI, Uvicorn
- **ML**: Hugging Face Transformers, PyTorch, scikit-learn
- **Frontend**: Vanilla HTML/CSS/JS, Chart.js, Font Awesome
- **Design**: Inter + Outfit fonts, glassmorphism dark theme

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
