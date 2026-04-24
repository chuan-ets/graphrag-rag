import requests
import json

url = "http://localhost:8080/query"
query = "Which company manufactures the engines for the rocket used in the Orion Mission, and where are they based?"

def test(method):
    print(f"\n--- TESTING METHOD: {method.upper()} ---")
    payload = {"query": query, "method": method}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
    else:
        print(f"Error: {response.status_code}")

# Thử nghiệm tìm kiếm thông thường (Vector)
test("vector")

# Thử nghiệm tìm kiếm có hỗ trợ Đồ thị (Graph/All)
test("all")
