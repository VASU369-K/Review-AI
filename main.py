import os
import io
import csv
import json
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from agent import process_agent_question

app = FastAPI(title="Sentiment AI Agent & Business Intelligence Generator")

# Create directories if not exist
os.makedirs("static", exist_ok=True)
os.makedirs("Dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Label mapping: 0=Negative, 1=Positive (consistent with training)
LABEL_MAP = {0: "Negative", 1: "Positive"}

# ----------------- Model Inference Layer -----------------

class SentimentModelWrapper:
    """
    Manages loading and running transformers pipelines.
    Only loads fine-tuned models from models/ directory.
    No fake fallback predictions.
    """
    def __init__(self):
        self.pipelines = {}
        self.model_labels = {}  # per-model label mapping from config

    def get_model_path(self, model_name: str) -> str:
        return os.path.join("models", model_name)

    def is_model_available(self, model_name: str) -> bool:
        model_path = self.get_model_path(model_name)
        # Check for key model files
        return (
            os.path.isdir(model_path)
            and (
                os.path.exists(os.path.join(model_path, "model.safetensors"))
                or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
                or os.path.exists(os.path.join(model_path, "config.json"))
            )
        )

    def load_pipeline(self, model_name: str):
        if model_name in self.pipelines:
            return self.pipelines[model_name]

        model_path = self.get_model_path(model_name)

        if not self.is_model_available(model_name):
            raise FileNotFoundError(
                f"Fine-tuned model '{model_name}' not found at '{model_path}'. "
                f"Please train it first: python train_models.py --model {model_name}"
            )

        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            print(f"Loading fine-tuned model from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            # Read label mapping from model config
            config = model.config
            if hasattr(config, 'id2label') and config.id2label:
                self.model_labels[model_name] = config.id2label
            else:
                self.model_labels[model_name] = LABEL_MAP

            pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
            self.pipelines[model_name] = pipe
            print(f"[✓] Model '{model_name}' loaded successfully.")
            return pipe
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}': {str(e)}. "
                f"Ensure the model is properly trained and saved."
            )

    def predict(self, text: str, model_name: str) -> Dict:
        """
        Runs prediction using the fine-tuned model.
        No fallback — returns an error if the model is unavailable.
        """
        pipe = self.load_pipeline(model_name)
        try:
            res = pipe(text[:512])[0]
            label = res["label"]
            score = res["score"]

            # Standardize label to POSITIVE/NEGATIVE using model config
            id2label = self.model_labels.get(model_name, LABEL_MAP)
            lbl_lower = label.lower()

            if "pos" in lbl_lower or "label_1" in lbl_lower:
                sentiment = "POSITIVE"
            elif "neg" in lbl_lower or "label_0" in lbl_lower:
                sentiment = "NEGATIVE"
            elif label in id2label.values():
                sentiment = label.upper()
            else:
                # Try to parse LABEL_N format
                try:
                    label_id = int(label.split("_")[-1])
                    mapped = id2label.get(label_id, id2label.get(str(label_id), "UNKNOWN"))
                    sentiment = mapped.upper()
                except (ValueError, KeyError):
                    sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"

            return {"sentiment": sentiment, "score": float(score)}
        except Exception as e:
            raise RuntimeError(f"Prediction failed for model '{model_name}': {str(e)}")


model_agent = SentimentModelWrapper()

# ----------------- API Request Pydantic Schemas -----------------

class AnalyzeRequest(BaseModel):
    text: str
    model: str = "distilbert"

class BatchAnalyzeRequest(BaseModel):
    reviews: List[str]
    model: str = "distilbert"

class AgentRequest(BaseModel):
    question: str
    model: str = "distilbert"

# ----------------- Aspects Definition -----------------

ASPECTS = {
    "Quality & Durability": {
        "keywords": ["quality", "material", "durable", "broke", "died", "plastic", "sturdy", "cheap", "ripped", "apart", "lasted", "defect"],
        "desc": "Reviews mentioning physical quality, longevity, defects, and assembly."
    },
    "Pricing & Value": {
        "keywords": ["price", "value", "cost", "expensive", "cheap", "worth", "money", "deal", "affordable", "overpriced", "bargain"],
        "desc": "Reviews mentioning item cost, pricing value, and cost-to-performance ratio."
    },
    "Customer Support & Delivery": {
        "keywords": ["support", "service", "shipping", "delivery", "arrived", "contacted", "return", "refund", "customer service", "seller", "package"],
        "desc": "Reviews mentioning shipping issues, responses, returns, or seller support."
    },
    "Usability & Design": {
        "keywords": ["easy", "difficult", "design", "manual", "setup", "install", "fit", "instructions", "comfortable", "size", "use", "heavy"],
        "desc": "Reviews mentioning user experience, setups, sizing, and convenience."
    }
}

# ---- Helper: detect aspects for a single review ----
def detect_aspects(text: str, sentiment: str) -> List[Dict]:
    """Detect which aspects appear in a review text."""
    text_lower = text.lower()
    results = []
    for aspect_name, aspect_info in ASPECTS.items():
        matched_keywords = [kw for kw in aspect_info["keywords"] if kw in text_lower]
        if matched_keywords:
            results.append({
                "aspect": aspect_name,
                "sentiment": sentiment,
                "matched_keywords": matched_keywords
            })
    return results


# ---- Cached BI data for agent ----
_cached_bi = {}


# ----------------- FastAPI Routes -----------------

@app.post("/api/analyze")
def analyze_review(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text review content cannot be empty.")

    try:
        result = model_agent.predict(req.text, req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Detect aspects
    aspects = detect_aspects(req.text, result["sentiment"])

    return {
        "text": req.text,
        "model": req.model,
        "sentiment": result["sentiment"],
        "score": result["score"],
        "confidence": round(result["score"] * 100, 1),
        "aspects": aspects
    }


@app.post("/api/batch-analyze")
def batch_analyze(req: BatchAnalyzeRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")

    try:
        _ = model_agent.load_pipeline(req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    positive_count = 0
    negative_count = 0
    aspect_summary = {k: {"pos": 0, "neg": 0, "total": 0} for k in ASPECTS.keys()}

    for text in req.reviews:
        if not text.strip():
            continue
        try:
            res = model_agent.predict(text, req.model)
            is_pos = res["sentiment"] == "POSITIVE"
            if is_pos:
                positive_count += 1
            else:
                negative_count += 1

            aspects = detect_aspects(text, res["sentiment"])

            # Update aspect summary
            text_lower = text.lower()
            for aspect_name, aspect_info in ASPECTS.items():
                if any(kw in text_lower for kw in aspect_info["keywords"]):
                    aspect_summary[aspect_name]["total"] += 1
                    if is_pos:
                        aspect_summary[aspect_name]["pos"] += 1
                    else:
                        aspect_summary[aspect_name]["neg"] += 1

            results.append({
                "text": text,
                "sentiment": res["sentiment"],
                "score": res["score"],
                "aspects": aspects
            })
        except Exception as e:
            results.append({
                "text": text,
                "sentiment": "ERROR",
                "score": 0.0,
                "error": str(e)
            })

    total = len(results)
    aspect_reports = []
    for name, data in aspect_summary.items():
        t = data["total"]
        if t > 0:
            aspect_reports.append({
                "aspect": name,
                "total_mentions": t,
                "positive_pct": round((data["pos"] / t) * 100, 1),
                "negative_pct": round((data["neg"] / t) * 100, 1),
            })

    return {
        "results": results,
        "model": req.model,
        "summary": {
            "total_processed": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": round((positive_count / total) * 100, 1) if total > 0 else 0,
            "negative_ratio": round((negative_count / total) * 100, 1) if total > 0 else 0,
        },
        "aspect_summary": aspect_reports,
    }


@app.post("/api/batch-upload")
async def batch_upload(file: UploadFile = File(...), model: str = "distilbert"):
    """Upload a CSV file for bulk analysis. CSV must have a 'text' or 'review' column."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        _ = model_agent.load_pipeline(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    content = await file.read()
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        text_content = content.decode('utf-8', errors='replace')

    reader = csv.DictReader(io.StringIO(text_content))
    fieldnames = reader.fieldnames or []

    # Find the text column
    text_col = None
    for col in fieldnames:
        if col.lower() in ['text', 'review', 'review_text', 'content', 'comment']:
            text_col = col
            break
    if not text_col:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have a 'text' or 'review' column. Found columns: {fieldnames}"
        )

    reviews = []
    for row in reader:
        t = row.get(text_col, "").strip()
        if t:
            reviews.append(t)
        if len(reviews) >= 500:
            break

    if not reviews:
        raise HTTPException(status_code=400, detail="No valid review text found in the uploaded CSV.")

    # Process via batch analyze
    batch_req = BatchAnalyzeRequest(reviews=reviews, model=model)
    return batch_analyze(batch_req)


@app.get("/api/bi-report")
def generate_bi_report(model: str = "distilbert", limit: int = 400):
    """
    Examines the preprocessed test_split.csv,
    evaluates reviews up to `limit`, categorizes them into aspects,
    creates sentiment frequencies, and generates business intelligence suggestions.
    """
    global _cached_bi

    # Verify the model is available first
    try:
        _ = model_agent.load_pipeline(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Prefer test_split.csv, fall back to val_split.csv
    split_file = "Dataset/test_split.csv"
    if not os.path.exists(split_file):
        split_file = "Dataset/val_split.csv"
    if not os.path.exists(split_file):
        split_file = "Dataset/train_split.csv"

    if not os.path.exists(split_file):
        raise HTTPException(status_code=404, detail="Processed split CSV files not found. Please run preprocessing first.")

    reviews_batch = []
    with open(split_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                reviews_batch.append((int(row[0]), row[1]))
                if len(reviews_batch) >= limit:
                    break

    if not reviews_batch:
        raise HTTPException(status_code=400, detail="Split CSV file is empty.")

    total_count = len(reviews_batch)
    positive_model_count = 0
    sentiment_history = []
    aspect_counts = {k: {"pos": 0, "neg": 0, "total": 0, "examples": []} for k in ASPECTS.keys()}

    word_freq_pos = {}
    word_freq_neg = {}
    stopwords = {"the", "a", "and", "is", "of", "to", "this", "it", "i", "was", "for", "with", "in", "but", "on", "that", "my", "you", "not", "have", "had", "as", "at"}

    # Run predictions and aspect matching
    for i, (label, text) in enumerate(reviews_batch):
        try:
            pred = model_agent.predict(text, model)
        except Exception:
            continue

        sentiment = pred["sentiment"]
        score = pred["score"]

        is_pos = (sentiment == "POSITIVE")
        if is_pos:
            positive_model_count += 1

        sentiment_history.append({
            "index": i + 1,
            "sentiment": sentiment,
            "score": score
        })

        # Aspect matching
        text_lower = text.lower()
        for aspect_name, aspect_info in ASPECTS.items():
            keywords = aspect_info["keywords"]
            if any(w in text_lower for w in keywords):
                aspect_counts[aspect_name]["total"] += 1
                if is_pos:
                    aspect_counts[aspect_name]["pos"] += 1
                else:
                    aspect_counts[aspect_name]["neg"] += 1

                if len(aspect_counts[aspect_name]["examples"]) < 3:
                    aspect_counts[aspect_name]["examples"].append({
                        "text": text[:150] + "..." if len(text) > 150 else text,
                        "sentiment": sentiment
                    })

        # Keyword mapping
        words = re.findall(r'[a-zA-Z]{3,}', text_lower)
        for w in words:
            if w in stopwords:
                continue
            if is_pos:
                word_freq_pos[w] = word_freq_pos.get(w, 0) + 1
            else:
                word_freq_neg[w] = word_freq_neg.get(w, 0) + 1

    # Format word lists
    top_pos_keywords = sorted(word_freq_pos.items(), key=lambda x: x[1], reverse=True)[:10]
    top_neg_keywords = sorted(word_freq_neg.items(), key=lambda x: x[1], reverse=True)[:10]

    # Overall ratios
    pos_ratio = positive_model_count / total_count if total_count > 0 else 0
    neg_ratio = 1.0 - pos_ratio

    # Compile Aspect Analytics & Recommendations
    aspect_reports = []
    agent_recommendations = []

    # Sort aspects by negative sentiment (worst first)
    sorted_aspects = sorted(aspect_counts.items(), key=lambda x: (x[1]["neg"] / x[1]["total"] if x[1]["total"] > 0 else 0), reverse=True)

    for aspect_name, data in sorted_aspects:
        total_aspect = data["total"]
        if total_aspect > 0:
            pos_pct = round((data["pos"] / total_aspect) * 100, 1)
            neg_pct = round((data["neg"] / total_aspect) * 100, 1)
        else:
            pos_pct, neg_pct = 0.0, 0.0

        aspect_reports.append({
            "aspect": aspect_name,
            "total_mentions": total_aspect,
            "positive_count": data["pos"],
            "negative_count": data["neg"],
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "description": ASPECTS[aspect_name]["desc"],
            "examples": data["examples"]
        })

        # Data-driven recommendations
        if total_aspect > 0 and neg_pct > 40:
            if "quality" in aspect_name.lower() or "durability" in aspect_name.lower():
                agent_recommendations.append(
                    f"⚠️ **High Defect Rate Warning:** Quality aspect has {neg_pct}% negative sentiment across {data['neg']} reviews. "
                    "Investigate early failures, material issues, and manufacturing defects."
                )
            elif "pric" in aspect_name.lower() or "value" in aspect_name.lower():
                agent_recommendations.append(
                    f"💲 **Pricing Strategy Review:** Value sentiment is {neg_pct}% negative across {data['neg']} reviews. "
                    "Customers feel features don't justify the cost. Consider promotional offers or bundle pricing."
                )
            elif "support" in aspect_name.lower() or "delivery" in aspect_name.lower():
                agent_recommendations.append(
                    f"📦 **Shipping & Support Alert:** Delivery sentiment is {neg_pct}% negative across {data['neg']} reviews. "
                    "Review courier SLAs, packaging quality, and support response times."
                )
            elif "usability" in aspect_name.lower() or "design" in aspect_name.lower():
                agent_recommendations.append(
                    f"⚙️ **Usability Improvement Needed:** Design usability is {neg_pct}% negative across {data['neg']} reviews. "
                    "Simplify setup instructions and create video tutorials."
                )

    if not agent_recommendations:
        agent_recommendations.append(
            "✅ **Healthy Performance:** Sentiments across all aspects are positive. Maintain current quality assurance protocols."
        )

    # Most frequent complaint
    most_frequent_complaint = aspect_reports[0]["aspect"] if aspect_reports and aspect_reports[0]["negative_pct"] > 0 else "None"

    # Overall satisfaction
    if pos_ratio >= 0.8:
        satisfaction = "Excellent"
    elif pos_ratio >= 0.6:
        satisfaction = "Good"
    elif pos_ratio >= 0.4:
        satisfaction = "Fair"
    else:
        satisfaction = "Poor"

    result = {
        "summary": {
            "total_processed": total_count,
            "positive_count": positive_model_count,
            "negative_count": total_count - positive_model_count,
            "positive_ratio": round(pos_ratio * 100, 1),
            "negative_ratio": round(neg_ratio * 100, 1),
            "overall_satisfaction": satisfaction,
            "most_frequent_complaint": most_frequent_complaint,
        },
        "aspect_analysis": aspect_reports,
        "keywords": {
            "positive": [{"word": k, "count": c} for k, c in top_pos_keywords],
            "negative": [{"word": k, "count": c} for k, c in top_neg_keywords]
        },
        "sentiment_timeline": sentiment_history[:50],
        "agent_recommendations": agent_recommendations,
        "model_used": model
    }

    # Cache for agent use
    _cached_bi[model] = result
    return result


@app.get("/api/metrics")
def get_model_metrics():
    """
    Returns actual evaluation metrics from models/metrics.json.
    """
    metrics_file = os.path.join("models", "metrics.json")

    if not os.path.exists(metrics_file):
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Train models first: python train_models.py --model all"
        )

    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # Find best model by F1 score
    best_model = None
    best_f1 = 0.0
    for name, m in metrics.items():
        f1 = m.get("f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    # Add availability status and best model flag
    enriched = {}
    for name in ["distilbert", "roberta", "deberta"]:
        if name in metrics:
            entry = dict(metrics[name])
            entry["available"] = model_agent.is_model_available(name)
            entry["is_best"] = (name == best_model)
            # Ensure trained_at and training_epochs are exposed
            if "trained_at" not in entry:
                entry["trained_at"] = None
            if "training_epochs" not in entry:
                entry["training_epochs"] = None
            enriched[name] = entry
        else:
            enriched[name] = {
                "accuracy": 0, "f1": 0, "precision": 0, "recall": 0,
                "available": model_agent.is_model_available(name),
                "is_best": False,
                "trained": False,
                "trained_at": None,
                "training_epochs": None,
            }

    enriched["_meta"] = {
        "best_model": best_model,
        "note": "Results from latest recorded evaluation run",
    }

    return enriched


@app.get("/api/model-status")
def get_model_status():
    """Check which models are trained and available."""
    status = {}
    for name in ["distilbert", "roberta", "deberta"]:
        available = model_agent.is_model_available(name)
        metadata_path = os.path.join("models", name, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        status[name] = {
            "available": available,
            "path": model_agent.get_model_path(name),
            "metadata": metadata
        }
    return status


@app.post("/api/agent")
def agent_endpoint(req: AgentRequest):
    """AI Agent: receives a natural-language question and returns a structured answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Get BI data (from cache or generate fresh)
    bi_data = _cached_bi.get(req.model)
    if not bi_data:
        # Try to generate BI data
        try:
            bi_data = generate_bi_report(model=req.model, limit=200)
        except Exception:
            bi_data = None

    # Get metrics data
    metrics_data = None
    metrics_file = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

    # Run agent
    result = process_agent_question(req.question, bi_data=bi_data, metrics_data=metrics_data)
    result["model_used"] = req.model
    result["question"] = req.question
    return result


@app.get("/api/export")
def export_report(format: str = "json", model: str = "distilbert"):
    """Export BI report as JSON or CSV."""
    bi_data = _cached_bi.get(model)
    if not bi_data:
        try:
            bi_data = generate_bi_report(model=model, limit=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

    if format.lower() == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        # Summary
        writer.writerow(["SENTIMENT ANALYSIS REPORT"])
        writer.writerow([])
        writer.writerow(["Summary"])
        summary = bi_data.get("summary", {})
        writer.writerow(["Total Reviews", summary.get("total_processed", 0)])
        writer.writerow(["Positive Count", summary.get("positive_count", 0)])
        writer.writerow(["Negative Count", summary.get("negative_count", 0)])
        writer.writerow(["Positive %", summary.get("positive_ratio", 0)])
        writer.writerow(["Negative %", summary.get("negative_ratio", 0)])
        writer.writerow(["Overall Satisfaction", summary.get("overall_satisfaction", "N/A")])
        writer.writerow(["Most Frequent Complaint", summary.get("most_frequent_complaint", "N/A")])
        writer.writerow(["Model Used", bi_data.get("model_used", "N/A")])
        writer.writerow([])

        # Aspect Analysis
        writer.writerow(["Aspect Analysis"])
        writer.writerow(["Aspect", "Total Mentions", "Positive %", "Negative %", "Positive Count", "Negative Count"])
        for a in bi_data.get("aspect_analysis", []):
            writer.writerow([
                a["aspect"], a["total_mentions"],
                a["positive_pct"], a["negative_pct"],
                a.get("positive_count", ""), a.get("negative_count", "")
            ])
        writer.writerow([])

        # Recommendations
        writer.writerow(["Recommendations"])
        for r in bi_data.get("agent_recommendations", []):
            writer.writerow([r])

        content = output.getvalue()
        return StreamingResponse(
            io.BytesIO(content.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bi_report.csv"}
        )
    else:
        # JSON export
        content = json.dumps(bi_data, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode('utf-8')),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=bi_report.json"}
        )


# ----------------- Serve HTML Files -----------------

@app.get("/")
def read_root():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/{catchall:path}")
def serve_static(catchall: str):
    file_path = os.path.join("static", catchall)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join("static", "index.html"))
