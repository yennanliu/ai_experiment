"""
Quick smoke test — sends all mock emails to the running server and prints results.

Usage:
    uv run python test_mock.py [--url http://localhost:8000]
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--url" else "http://localhost:8000"
DIVIDER = "─" * 70


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def run():
    # Fetch mock emails from the server
    with urllib.request.urlopen(f"{BASE_URL}/mock-emails", timeout=10) as resp:
        mocks = json.loads(resp.read())

    print(f"\n{'AI Email Reply Assistant — Mock Test':^70}")
    print(f"Server: {BASE_URL}  |  Emails: {len(mocks)}\n")

    for i, mock in enumerate(mocks, 1):
        print(DIVIDER)
        print(f"[{i}/{len(mocks)}] {mock['label']}")
        print(DIVIDER)

        try:
            result = post_json(f"{BASE_URL}/generate-draft", {"email": mock["email"]})

            print(f"Type     : {result['email_type']}")
            print(f"\nDraft Reply:\n{result['draft_reply']}")

            if result["checklist"]:
                print(f"\nChecklist:")
                for item in result["checklist"]:
                    prefix = "⚠️ " if item.startswith("Missing") else "🚩 "
                    print(f"  {prefix}{item}")
            else:
                print("\nChecklist: ✅ All clear")

        except urllib.error.URLError as e:
            print(f"  ERROR: {e}")

        print()

    print(DIVIDER)
    print("Done.\n")


if __name__ == "__main__":
    run()
