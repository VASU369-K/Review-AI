"""
AI Agent Orchestrator for Review-AI.

Receives natural-language business questions and routes them to
the appropriate analysis tool (sentiment, aspect, BI, metrics, recommendations).
"""

import re

# --- Tool Definitions ---

TOOLS = {
    "sentiment_analysis": {
        "name": "Sentiment Analysis Tool",
        "description": "Analyzes the sentiment of a review or set of reviews.",
        "triggers": [
            "sentiment", "positive", "negative", "happy", "unhappy", "satisfied",
            "dissatisfied", "feeling", "opinion", "think", "percentage",
            "how many", "ratio", "rate", "proportion", "customers feel",
            "customer satisfaction", "satisfaction rate"
        ]
    },
    "aspect_analysis": {
        "name": "Aspect Analysis Tool",
        "description": "Identifies what specific product aspects customers praise or complain about.",
        "triggers": [
            "aspect", "complaint", "complain", "problem", "issue", "quality",
            "durability", "price", "pricing", "delivery", "shipping", "support",
            "design", "usability", "about", "most unhappy", "biggest",
            "worst", "best feature", "top complaint", "pain point",
            "what are customers", "category", "mention"
        ]
    },
    "business_intelligence": {
        "name": "Business Intelligence Tool",
        "description": "Generates a comprehensive BI report from reviewed data.",
        "triggers": [
            "report", "summary", "summarize", "overview", "overall",
            "business", "intelligence", "insight", "kpi", "dashboard",
            "feedback", "customer feedback", "stats", "statistics",
            "distribution", "breakdown"
        ]
    },
    "model_metrics": {
        "name": "Model Metrics Tool",
        "description": "Returns evaluation metrics and comparisons for trained ML models.",
        "triggers": [
            "model", "accuracy", "f1", "precision", "recall", "performance",
            "which model", "best model", "compare", "comparison", "benchmark",
            "metric", "performs best", "evaluate"
        ]
    },
    "recommendation": {
        "name": "Recommendation Tool",
        "description": "Generates actionable business recommendations based on analysis.",
        "triggers": [
            "recommend", "suggestion", "action", "improve", "fix",
            "should we", "what should", "advise", "advice", "strategy",
            "next step", "how to improve", "optimize"
        ]
    }
}


def classify_question(question: str) -> str:
    """
    Classifies a natural-language question into one of the tool categories.
    Uses keyword matching with scoring.
    """
    question_lower = question.lower().strip()
    scores = {}

    for tool_key, tool_info in TOOLS.items():
        score = 0
        for trigger in tool_info["triggers"]:
            if trigger in question_lower:
                # Multi-word triggers get higher weight
                weight = len(trigger.split())
                score += weight
        scores[tool_key] = score

    # Return the best-scoring tool, default to business_intelligence
    best_tool = max(scores, key=scores.get)
    if scores[best_tool] == 0:
        return "business_intelligence"
    return best_tool


def build_sentiment_answer(bi_data: dict, question: str) -> dict:
    """Builds an answer focused on sentiment distribution."""
    summary = bi_data.get("summary", {})
    total = summary.get("total_processed", 0)
    pos_ratio = summary.get("positive_ratio", 0)
    neg_ratio = summary.get("negative_ratio", 0)
    pos_count = summary.get("positive_count", 0)
    neg_count = summary.get("negative_count", 0)

    question_lower = question.lower()
    if any(w in question_lower for w in ["unhappy", "negative", "dissatisfied", "bad"]):
        answer = (
            f"{neg_ratio}% of customers express negative sentiment. "
            f"Out of {total} reviews analyzed, {neg_count} are negative."
        )
    elif any(w in question_lower for w in ["happy", "positive", "satisfied", "good"]):
        answer = (
            f"{pos_ratio}% of customers express positive sentiment. "
            f"Out of {total} reviews analyzed, {pos_count} are positive."
        )
    else:
        answer = (
            f"Overall sentiment distribution: {pos_ratio}% positive, {neg_ratio}% negative. "
            f"Total reviews analyzed: {total}."
        )

    return {
        "task": "Sentiment Analysis",
        "tool_used": TOOLS["sentiment_analysis"]["name"],
        "answer": answer,
        "supporting_data": {
            "total_reviews": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_pct": pos_ratio,
            "negative_pct": neg_ratio,
        },
        "recommendations": _get_sentiment_recommendations(pos_ratio, neg_ratio)
    }


def build_aspect_answer(bi_data: dict, question: str) -> dict:
    """Builds an answer focused on aspect analysis."""
    aspects = bi_data.get("aspect_analysis", [])
    if not aspects:
        return {
            "task": "Aspect Analysis",
            "tool_used": TOOLS["aspect_analysis"]["name"],
            "answer": "No aspect data available. Please generate a BI report first.",
            "supporting_data": {},
            "recommendations": []
        }

    # Sort by negative percentage (worst first)
    sorted_aspects = sorted(aspects, key=lambda x: x.get("negative_pct", 0), reverse=True)
    worst = sorted_aspects[0]
    best = sorted_aspects[-1]

    question_lower = question.lower()
    if any(w in question_lower for w in ["worst", "complaint", "problem", "unhappy", "negative", "issue"]):
        answer = (
            f"\"{worst['aspect']}\" has the highest negative sentiment at {worst['negative_pct']}%. "
            f"It was mentioned in {worst['total_mentions']} reviews."
        )
    elif any(w in question_lower for w in ["best", "praise", "positive", "happy", "like"]):
        answer = (
            f"\"{best['aspect']}\" has the highest positive sentiment at {best['positive_pct']}%. "
            f"It was mentioned in {best['total_mentions']} reviews."
        )
    else:
        lines = []
        for a in sorted_aspects:
            lines.append(f"• {a['aspect']}: {a['positive_pct']}% positive, {a['negative_pct']}% negative ({a['total_mentions']} mentions)")
        answer = "Aspect breakdown (sorted by highest negative sentiment):\n" + "\n".join(lines)

    return {
        "task": "Aspect Analysis",
        "tool_used": TOOLS["aspect_analysis"]["name"],
        "answer": answer,
        "supporting_data": {
            "aspects": sorted_aspects,
            "worst_aspect": worst["aspect"],
            "best_aspect": best["aspect"],
        },
        "recommendations": _get_aspect_recommendations(sorted_aspects)
    }


def build_bi_answer(bi_data: dict, question: str) -> dict:
    """Builds a comprehensive BI summary answer."""
    summary = bi_data.get("summary", {})
    aspects = bi_data.get("aspect_analysis", [])
    recs = bi_data.get("agent_recommendations", [])
    model_used = bi_data.get("model_used", "unknown")

    sorted_aspects = sorted(aspects, key=lambda x: x.get("negative_pct", 0), reverse=True)
    worst_aspect = sorted_aspects[0]["aspect"] if sorted_aspects else "N/A"

    satisfaction = summary.get('overall_satisfaction', 'N/A')
    answer = (
        f"Business Intelligence Summary:\n"
        f"• Total reviews analyzed: {summary.get('total_processed', 0)}\n"
        f"• Positive sentiment: {summary.get('positive_ratio', 0)}% ({summary.get('positive_count', 0)} reviews)\n"
        f"• Negative sentiment: {summary.get('negative_ratio', 0)}% ({summary.get('negative_count', 0)} reviews)\n"
        f"• Overall satisfaction: {satisfaction}\n"
        f"• Top customer complaint: {worst_aspect}\n"
        f"• Model used: {model_used}"
    )

    return {
        "task": "Business Intelligence",
        "tool_used": TOOLS["business_intelligence"]["name"],
        "answer": answer,
        "supporting_data": {
            "summary": summary,
            "top_complaint": worst_aspect,
            "model_used": model_used,
            "aspect_count": len(aspects),
        },
        "recommendations": [r for r in recs]
    }


def build_metrics_answer(metrics_data: dict, question: str) -> dict:
    """Builds an answer about model performance metrics."""
    if not metrics_data:
        return {
            "task": "Model Metrics",
            "tool_used": TOOLS["model_metrics"]["name"],
            "answer": "No model metrics available. Please train the models first.",
            "supporting_data": {},
            "recommendations": ["Train all three models using: python train_models.py --model all"]
        }

    # Find best model
    best_model = None
    best_f1 = 0.0
    lines = []
    for name, m in metrics_data.items():
        acc = m.get("accuracy", 0)
        f1 = m.get("f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = name
        lines.append(
            f"• {name.upper()}: Accuracy={acc*100:.1f}%, F1={f1*100:.1f}%, "
            f"Precision={m.get('precision',0)*100:.1f}%, Recall={m.get('recall',0)*100:.1f}%"
        )

    question_lower = question.lower()
    if any(w in question_lower for w in ["best", "which", "top", "highest"]):
        answer = (
            f"The best performing model is **{best_model.upper()}** with an F1 score of {best_f1*100:.1f}%.\n\n"
            f"All model metrics:\n" + "\n".join(lines)
        )
    else:
        answer = "Model Performance Comparison:\n" + "\n".join(lines)

    return {
        "task": "Model Metrics",
        "tool_used": TOOLS["model_metrics"]["name"],
        "answer": answer,
        "supporting_data": metrics_data,
        "recommendations": [
            f"Use {best_model.upper()} for highest accuracy in production." if best_model else "Train models first."
        ]
    }


def build_recommendation_answer(bi_data: dict, question: str) -> dict:
    """Builds actionable recommendations."""
    recs = bi_data.get("agent_recommendations", [])
    aspects = bi_data.get("aspect_analysis", [])
    sorted_aspects = sorted(aspects, key=lambda x: x.get("negative_pct", 0), reverse=True)

    if not recs:
        recs = _get_aspect_recommendations(sorted_aspects)

    if not recs:
        recs = ["All aspects show healthy sentiment. Maintain current quality standards."]

    answer = "AI-Generated Business Recommendations:\n" + "\n".join(f"• {r}" for r in recs)

    return {
        "task": "Business Recommendations",
        "tool_used": TOOLS["recommendation"]["name"],
        "answer": answer,
        "supporting_data": {
            "aspects_analyzed": len(aspects),
            "negative_aspects": [a["aspect"] for a in sorted_aspects if a.get("negative_pct", 0) > 40],
        },
        "recommendations": recs
    }


def _get_sentiment_recommendations(pos_ratio, neg_ratio):
    recs = []
    if neg_ratio > 50:
        recs.append("Critical: More than half of customers are unhappy. Immediate product/service review needed.")
    elif neg_ratio > 30:
        recs.append("Significant negative sentiment detected. Investigate top complaint areas and prioritize fixes.")
    else:
        recs.append("Overall sentiment is positive. Continue monitoring for emerging issues.")
    return recs


def _get_aspect_recommendations(sorted_aspects):
    recs = []
    for a in sorted_aspects:
        neg_pct = a.get("negative_pct", 0)
        name = a.get("aspect", "Unknown")
        if neg_pct > 40:
            if "quality" in name.lower() or "durability" in name.lower():
                recs.append(
                    f"Quality & Durability has {neg_pct}% negative sentiment. "
                    "Investigate defect reports and improve QA processes."
                )
            elif "pric" in name.lower() or "value" in name.lower():
                recs.append(
                    f"Pricing & Value has {neg_pct}% negative sentiment. "
                    "Consider competitive pricing analysis or value-add bundles."
                )
            elif "support" in name.lower() or "delivery" in name.lower():
                recs.append(
                    f"Customer Support & Delivery has {neg_pct}% negative sentiment. "
                    "Review shipping SLAs and bolster customer service capacity."
                )
            elif "usability" in name.lower() or "design" in name.lower():
                recs.append(
                    f"Usability & Design has {neg_pct}% negative sentiment. "
                    "Simplify setup instructions and improve product documentation."
                )
    return recs


# --- Main dispatcher ---

def process_agent_question(question: str, bi_data: dict = None, metrics_data: dict = None) -> dict:
    """
    Main entry point for the AI Agent.
    Classifies the question, selects the tool, and returns a structured answer.
    """
    tool = classify_question(question)

    if tool == "sentiment_analysis" and bi_data:
        return build_sentiment_answer(bi_data, question)
    elif tool == "aspect_analysis" and bi_data:
        return build_aspect_answer(bi_data, question)
    elif tool == "business_intelligence" and bi_data:
        return build_bi_answer(bi_data, question)
    elif tool == "model_metrics":
        return build_metrics_answer(metrics_data or {}, question)
    elif tool == "recommendation" and bi_data:
        return build_recommendation_answer(bi_data, question)
    else:
        # Default: try BI if data available, otherwise metrics
        if bi_data:
            return build_bi_answer(bi_data, question)
        elif metrics_data:
            return build_metrics_answer(metrics_data, question)
        else:
            return {
                "task": "General",
                "tool_used": "None",
                "answer": "I cannot answer this question yet — no analysis data is available. Please ensure at least one model is trained and load the Dashboard tab first to generate BI data.",
                "supporting_data": {},
                "recommendations": [
                    "Train models with: python train_models.py --model all",
                    "Start the server: python -m uvicorn main:app --reload",
                    "Load the Dashboard tab to generate BI analysis data."
                ]
            }
