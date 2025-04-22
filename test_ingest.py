from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ingest_endpoint():
    print("\n🔍 Testing /ingest endpoint...")
    response = client.post("/ingest")
    print(f"📦 Status Code: {response.status_code}")
    print(f"📄 Response JSON: {response.json()}")

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    print("✅ Test Passed: /ingest endpoint works correctly.\n")

if __name__ == "__main__":
    test_ingest_endpoint()
