import urllib.request
import json

base_url = "http://127.0.0.1:8000"

def test_inference(text, model):
    req_data = json.dumps({"text": text, "model": model}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/analyze",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode())

def test_metrics():
    with urllib.request.urlopen(f"{base_url}/api/metrics") as res:
        return json.loads(res.read().decode())

def run_tests():
    print("--- LIVE TEST: REVIEW INFERENCE ---")
    sample_text = "This product has excellent quality. It works perfectly and the value for money is amazing!"
    print(f"Input Review: \"{sample_text}\"\n")
    
    for model in ["distilbert", "roberta", "deberta"]:
        try:
            res = test_inference(sample_text, model)
            print(f"[{model.upper()}] Inference:")
            print(f"  - Sentiment: {res['sentiment']}")
            print(f"  - Confidence: {res['score']*100:.1f}%")
        except Exception as e:
            print(f"[{model.upper()}] Inference Failed: {e}")

    print("\n--- LIVE TEST: ACCURACY & PERFORMANCE METRICS ---")
    try:
        metrics = test_metrics()
        for idx, (model, spec) in enumerate(metrics.items(), 1):
            print(f"{idx}. {model.upper()}:")
            print(f"  - Accuracy: {spec['accuracy']*100:.1f}%")
            print(f"  - F1-Score: {spec['f1']*100:.1f}%")
            print(f"  - Parameter Size: {spec['parameters']}")
            print(f"  - CPU Latency (per batch): {spec['eval_time_sec']}ms")
    except Exception as e:
        print(f"Failed to fetch model metrics: {e}")

if __name__ == "__main__":
    run_tests()
