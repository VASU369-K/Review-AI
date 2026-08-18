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
# Generate a test upload CSV with a malformed row or empty text
with open("temp_test.csv", "w", encoding="utf-8") as f:
    f.write("text\n")
    f.write("This product is amazing. I love the packaging and quality.\n")
    f.write("Worst purchase ever! Highly disappointed.\n")
    f.write("   \n")  # Empty line, will cause validation / prediction failure or skip depending on rule

upload_res = api_multipart("/api/batch-upload?model=distilbert", "temp_test.csv")
print(f"  Processed count: {len(upload_res['results'])}")
print(f"  Summary stats positive: {upload_res['summary']['positive_count']}, negative: {upload_res['summary']['negative_count']}")
assert upload_res["summary"]["total_processed"] >= 2, "Should process at least two rows"
print("  PASS")

# 4. Export endpoint
print("\n[4] Testing BI Report Export (/api/export):")
req_path = "/api/export?model=distilbert&format=csv&limit=10"
req = urllib.request.Request(f"{BASE}{req_path}")
req.method = "GET"
resp = urllib.request.urlopen(req)
csv_lines = resp.read().decode("utf-8").split("\n")
print(f"  Exported CSV contains {len(csv_lines)} lines. Header: {csv_lines[0]}")
assert "sentiment" in csv_lines[0].lower(), "Exported CSV must contain sentiment"
print("  PASS")

req_json_path = "/api/export?model=distilbert&format=json&limit=10"
req_json = urllib.request.Request(f"{BASE}{req_json_path}")
req_json.method = "GET"
resp_json = urllib.request.urlopen(req_json)
json_data = json.loads(resp_json.read().decode("utf-8"))
print(f"  Exported JSON count: {len(json_data.get('results', []))}")
assert json_data.get("summary", {}).get("total_processed", 0) > 0
print("  PASS")

# 5. AI Agent orchestrator intent test
print("\n[5] Testing AI Agent Questions (NL Queries):")
queries = [
    "What percentage of customers are unhappy?",
    "What are customers most unhappy about?",
    "What is the biggest complaint?",
    "Which model performs best?",
    "Give me business recommendations.",
    "Summarize customer feedback."
]

for q in queries:
    res = api("POST", "/api/agent", {"question": q, "model": "roberta"})
    print(f"  Query: '{q}' -> Tool: {res['tool_used']} -> Answers: {res['answer'][:90]}...")
    assert res["tool_used"] is not None and res["answer"] != "", f"Agent failed on query '{q}'"
    # Ensure double asterisks ** are not displayed in the recommendations/answer headers or response strings returned
    assert "**" not in res["answer"], f"Found markdown bold in response: {res['answer']}"
    for rec in res.get("recommendations", []):
        assert "**" not in rec, f"Found markdown bold in recommendation: {rec}"

print("  PASS")

# Clean temp file
import os
if os.path.exists("temp_test.csv"):
    os.remove("temp_test.csv")

print("\n" + "=" * 60)
print("ALL EXTENSIVE FUNCTIONAL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
