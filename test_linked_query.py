import requests
import json

def test_query():
    url = "http://localhost:8080/query"
    payload = {
        "query": "Which company manufactures the engines for the rocket used in the Orion Mission?"
    }
    headers = {
        "Content-Type": "application/json"
    }

    print(f"--- Sending Query to {url} ---")
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        print("\n=== ANSWER ===")
        print(data.get("answer"))
        
        print("\n=== SOURCES FOUND ===")
        for i, source in enumerate(data.get("sources", [])):
            fname = source.get("filename", "N/A")
            score = source.get("score", 0)
            print(f"[{i+1}] File: {fname} (Score: {score:.3f})")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_query()
