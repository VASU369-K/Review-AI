"""Quick API verification script for Review-AI."""
import urllib.request
import json

BASE = "http://127.0.0.1:8000"

def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"})
    req.method = method
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())

print("=" * 60)
print("TEST 1: Model Status")
r = api("GET", "/api/model-status")
for m in ["distilbert", "roberta", "deberta"]:
    avail = r[m]["available"]
    print(f"  {m}: available={avail}")
    assert avail, f"{m} should be available"
print("  PASS\n")

print("TEST 2: Metrics")
r = api("GET", "/api/metrics")
best = r["_meta"]["best_model"]
print(f"  Best model: {best}")
for m in ["distilbert", "roberta", "deberta"]:
    acc = r[m].get("accuracy", 0)
    f1 = r[m].get("f1", 0)
    trained_at = r[m].get("trained_at", "N/A")
    print(f"  {m}: acc={acc:.1%} f1={f1:.1%} trained_at={trained_at}")
    assert acc > 0.8, f"{m} accuracy too low"
print("  PASS\n")

print("TEST 3: Positive Review (DistilBERT)")
r = api("POST", "/api/analyze", {"text": "Amazing product, works perfectly! High quality.", "model": "distilbert"})
print(f"  sentiment={r['sentiment']} confidence={r['confidence']:.1f}%")
assert r["sentiment"] == "POSITIVE", f"Expected POSITIVE, got {r['sentiment']}"
print("  PASS\n")

print("TEST 4: Negative Review (RoBERTa)")
r = api("POST", "/api/analyze", {"text": "Terrible quality, broke instantly. Waste of money.", "model": "roberta"})
print(f"  sentiment={r['sentiment']} confidence={r['confidence']:.1f}%")
assert r["sentiment"] == "NEGATIVE", f"Expected NEGATIVE, got {r['sentiment']}"
print("  PASS\n")

print("TEST 5: DeBERTa Inference")
r = api("POST", "/api/analyze", {"text": "Great value for money, fast shipping.", "model": "deberta"})
print(f"  sentiment={r['sentiment']} confidence={r['confidence']:.1f}% aspects={len(r.get('aspects', []))}")
assert r["sentiment"] in ["POSITIVE", "NEGATIVE"], "DeBERTa should return a valid sentiment"
print("  PASS\n")

print("TEST 6: BI Report")
r = api("GET", "/api/bi-report?model=distilbert&limit=50")
summary = r.get("summary", {})
print(f"  total_processed={summary.get('total_processed')} positive_ratio={summary.get('positive_ratio')}%")
print(f"  aspects={len(r.get('aspect_analysis', []))}")
print(f"  recommendations={len(r.get('agent_recommendations', []))}")
assert summary.get("total_processed", 0) > 0, "BI report should process reviews"
print("  PASS\n")

print("TEST 7: AI Agent")
r = api("POST", "/api/agent", {"question": "What are customers most unhappy about?", "model": "distilbert"})
print(f"  task={r['task']} tool_used={r['tool_used']}")
print(f"  answer preview: {r['answer'][:120]}...")
assert r["task"] != "General", "Agent should classify the question"
print("  PASS\n")

print("=" * 60)
print("ALL 7 TESTS PASSED!")
print("=" * 60)
