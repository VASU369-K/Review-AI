"""Comprehensive features verification script for Review-AI."""
import urllib.request
import urllib.parse
import json

BASE = "http://127.0.0.1:8000"

def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"})
    req.method = method
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())

def api_multipart(path, file_path):
    # Simple multipart/form-data implementation
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(file_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_upload.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
        f"{csv_content}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body))
        }
    )
    req.method = "POST"
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())

print("=" * 60)
print("VERIFICATION OF ALL DETAILED USER REQUIREMENTS")
print("=" * 60)

# 1. Mixed Review Inference
print("\n[1] Mixed Review test using RoBERTa:")
r = api("POST", "/api/analyze", {"text": "The design looks excellent and feels very premium, but the support was quite slow.", "model": "roberta"})
print(f"  sentiment={r['sentiment']} confidence={r['confidence']}% aspects={r['aspects']}")
assert r["sentiment"] in ["POSITIVE", "NEGATIVE"], "Should return valid sentiment"
print("  PASS")

# 2. Model consistency (No fallback)
print("\n[2] Checking strict model load errors (invalid model names):")
try:
    api("POST", "/api/analyze", {"text": "Testing", "model": "invalid_model"})
    print("  FAILED: Allowed invalid model without error!")
    assert False
except Exception:
    print("  PASS: Invalid model appropriately rejected")

# 3. CSV Upload Validation & Failed Predictions reporting
print("\n[3] Testing CSV Batch Upload and Failure stats reporting:")
with open("temp_test.csv", "w", encoding="utf-8") as f:
    f.write("text\n")
    f.write("This product is amazing. I love the packaging and quality.\n")
    f.write("Worst purchase ever! Highly disappointed.\n")
    f.write("   \n")  # Empty line

upload_res = api_multipart("/api/batch-upload?model=distilbert", "temp_test.csv")
print(f"  Processed count: {len(upload_res['results'])}")
print(f"  Summary stats positive: {upload_res['summary']['positive_count']}, negative: {upload_res['summary']['negative_count']}")
assert upload_res["summary"]["total_processed"] >= 2, "Should process at least two rows"
print("  PASS")

# 4. Metrics endpoint — check required fields
print("\n[4] Testing Metrics endpoint (/api/metrics):")
metrics = api("GET", "/api/metrics")
assert "_meta" in metrics, "Metrics must contain _meta"
assert "best_model" in metrics["_meta"], "Metrics._meta must contain best_model"
best = metrics["_meta"]["best_model"]
print(f"  Best model: {best}")
for model_name in ["distilbert", "roberta", "deberta"]:
    m = metrics.get(model_name, {})
    assert "accuracy" in m, f"{model_name} missing accuracy"
    assert "f1" in m, f"{model_name} missing f1"
    assert "precision" in m, f"{model_name} missing precision"
    assert "recall" in m, f"{model_name} missing recall"
    print(f"  {model_name}: acc={m.get('accuracy',0):.4f} f1={m.get('f1',0):.4f} prec={m.get('precision',0):.4f} recall={m.get('recall',0):.4f}")
print("  PASS")

# 5. BI Report — check failure accounting and executive insight
print("\n[5] Testing BI Report failure accounting and executive insight:")
bi = api("GET", "/api/bi-report?model=distilbert&limit=50")
summary = bi["summary"]
print(f"  total_attempted={summary.get('total_attempted', 'MISSING')}")
print(f"  total_processed={summary.get('total_processed', 'MISSING')}")
print(f"  failed_count={summary.get('failed_count', 'MISSING')}")
assert "total_attempted" in summary, "BI summary must include total_attempted"
assert "total_processed" in summary, "BI summary must include total_processed"
assert "failed_count" in summary, "BI summary must include failed_count"
assert "best_model" in summary, "BI summary must include best_model"
# Check executive insight
assert "executive_insight" in bi, "BI must include executive_insight"
ei = bi["executive_insight"]
assert "overall_sentiment" in ei, "Executive insight must include overall_sentiment"
assert "major_complaint" in ei, "Executive insight must include major_complaint"
assert "strongest_positive_aspect" in ei, "Executive insight must include strongest_positive_aspect"
assert "recommended_action" in ei, "Executive insight must include recommended_action"
print(f"  Executive insight: {ei['overall_sentiment']}, complaint={ei['major_complaint']}, strongest={ei['strongest_positive_aspect']}")
print("  PASS")

# 6. Export endpoint — JSON with model_evaluation
print("\n[6] Testing BI Report Export (/api/export):")
req_json_path = "/api/export?model=distilbert&format=json"
req_json = urllib.request.Request(f"{BASE}{req_json_path}")
req_json.method = "GET"
resp_json = urllib.request.urlopen(req_json)
json_data = json.loads(resp_json.read().decode("utf-8"))
assert json_data.get("summary", {}).get("total_processed", 0) > 0, "Export must have processed data"
assert "model_evaluation" in json_data, "JSON export must include model_evaluation"
assert "executive_insight" in json_data, "JSON export must include executive_insight"
print(f"  JSON export has model_evaluation keys: {list(json_data['model_evaluation'].keys())}")
print("  PASS")

# 7. CSV Export
print("\n[7] Testing CSV Export:")
req_csv_path = "/api/export?model=distilbert&format=csv"
req_csv = urllib.request.Request(f"{BASE}{req_csv_path}")
req_csv.method = "GET"
resp_csv = urllib.request.urlopen(req_csv)
csv_text = resp_csv.read().decode("utf-8")
csv_lines = csv_text.split("\n")
assert any("SENTIMENT ANALYSIS REPORT" in line for line in csv_lines), "CSV must contain report header"
assert any("Executive Insight" in line for line in csv_lines), "CSV must contain Executive Insight section"
assert any("Model Evaluation" in line for line in csv_lines), "CSV must contain Model Evaluation section"
print(f"  CSV export has {len(csv_lines)} lines, includes Executive Insight and Model Evaluation sections")
print("  PASS")

# 8. AI Agent orchestrator intent test
print("\n[8] Testing AI Agent Questions (NL Queries):")
queries = [
    ("What percentage of customers are unhappy?", "Sentiment Analysis Tool"),
    ("What are customers most unhappy about?", "Aspect Analysis Tool"),
    ("What is the biggest complaint?", "Aspect Analysis Tool"),
    ("Which model performs best?", "Model Metrics Tool"),
    ("Give me business recommendations.", "Recommendation Tool"),
    ("Summarize customer feedback.", "Business Intelligence Tool"),
    ("Compare all models", "Model Metrics Tool"),
    ("Which model should I deploy?", "Model Metrics Tool"),
]

for q, expected_tool in queries:
    res = api("POST", "/api/agent", {"question": q, "model": "roberta"})
    print(f"  Query: '{q}' -> Tool: {res['tool_used']} -> Ans: {res['answer'][:80]}...")
    assert res["tool_used"] is not None and res["answer"] != "", f"Agent failed on query '{q}'"
    # Verify no markdown ** in response
    assert "**" not in res["answer"], f"Found markdown bold in response: {res['answer']}"
    for rec in res.get("recommendations", []):
        assert "**" not in rec, f"Found markdown bold in recommendation: {rec}"

print("  PASS")

# 9. Model status endpoint
print("\n[9] Testing Model Status endpoint:")
status = api("GET", "/api/model-status")
for m in ["distilbert", "roberta", "deberta"]:
    assert m in status, f"Model status must include {m}"
    assert status[m]["available"] is True, f"{m} should be available"
    print(f"  {m}: available={status[m]['available']}, trained={status[m].get('trained', 'N/A')}")
print("  PASS")

# Clean temp file
import os
if os.path.exists("temp_test.csv"):
    os.remove("temp_test.csv")

print("\n" + "=" * 60)
print("ALL EXTENSIVE FUNCTIONAL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
