"""
Equivalent Python script for:

curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{...}'
"""

import requests
import json

URL = "http://localhost:8000/api/v1/jobs/"

HEADERS = {
    "Content-Type": "application/json"
}

DATA = {
    "job_id": "backend-2025",
    "job_title": "Senior Backend Engineer",
    "requirements": [
        "3+ years Python backend development experience",
        "Experience with REST API design using FastAPI or Django",
        "AWS Lambda and S3 hands-on experience",
        "Strong understanding of PostgreSQL and Redis",
        "Experience with Docker and CI/CD pipelines"
    ],
    "nice_to_have": [
        "Kubernetes and container orchestration",
        "Terraform infrastructure as code"
    ]
}


def main():
    try:
        response = requests.post(
            URL,
            headers=HEADERS,
            json=DATA  # requests tự convert sang JSON
        )

        print("=" * 50)
        print("STATUS:", response.status_code)
        print("=" * 50)

        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI server")
        print("👉 Bạn đã chạy server chưa?")
        print("   uvicorn main:app --reload")
    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == "__main__":
    main()