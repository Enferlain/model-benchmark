import requests
import time
import os
import sys

BASE_URL = "http://localhost:8000"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models")

def wait_for_backend():
    print("Waiting for backend to be ready...")
    retries = 10
    while retries > 0:
        try:
            requests.get(f"{BASE_URL}/api/models", timeout=2)
            print("Backend is ready.")
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            retries -= 1
            print(f"Retrying... ({retries} left)")
    return False

def check_no_active_download():
    resp = requests.get(f"{BASE_URL}/api/models/download/status")
    status = resp.json()
    if status["is_downloading"]:
        print("Warning: A download is already in progress. Waiting for it to finish...")
        while True:
            resp = requests.get(f"{BASE_URL}/api/models/download/status")
            status = resp.json()
            if not status["is_downloading"]:
                break
            time.sleep(1)
    return True

def test_download_success():
    print("\n--- Test Case 1: Successful Download (Small File) ---")

    # Using a small text file as a dummy model
    url = "https://raw.githubusercontent.com/google/google-api-python-client/main/README.md"
    name = "test_download_success"

    payload = {
        "url": url,
        "name": name,
        "source": "Test"
    }

    # Ensure clean state
    target_file = os.path.join(MODELS_DIR, f"{name}.safetensors")
    if os.path.exists(target_file):
        os.remove(target_file)

    print(f"Requesting download from {url} as {name}...")
    resp = requests.post(f"{BASE_URL}/api/models/download", json=payload)

    if resp.status_code != 200:
        print(f"FAILED: Status code {resp.status_code}, {resp.text}")
        return False

    print("Download started. Monitoring status...")

    while True:
        status_resp = requests.get(f"{BASE_URL}/api/models/download/status")
        status = status_resp.json()
        print(f"Status: {status['status']}, Progress: {status['progress']}/{status['total']}")

        if status["status"] == "completed":
            print("Download reported complete.")
            break
        if status["status"] == "error":
            print(f"FAILED: Download reported error: {status['error']}")
            return False

        time.sleep(0.5)

    # Verify file exists
    if os.path.exists(target_file):
        print(f"SUCCESS: File {target_file} exists.")
        # Cleanup
        os.remove(target_file)
        return True
    else:
        print(f"FAILED: File {target_file} does not exist.")
        return False

def test_download_error():
    print("\n--- Test Case 2: Download Error (Invalid URL) ---")

    url = "http://this-domain-does-not-exist-at-all-12345.com/model.safetensors"
    name = "test_download_error"

    payload = {
        "url": url,
        "name": name,
        "source": "Test"
    }

    print(f"Requesting download from invalid URL...")
    resp = requests.post(f"{BASE_URL}/api/models/download", json=payload)

    if resp.status_code != 200:
        # Note: The API returns 200 "started" even if the URL is bad,
        # because it starts in background.
        print(f"Unexpected status code {resp.status_code}")

    print("Monitoring status for expected error...")

    while True:
        status_resp = requests.get(f"{BASE_URL}/api/models/download/status")
        status = status_resp.json()

        if status["status"] == "error":
            print(f"SUCCESS: Error reported as expected: {status['error']}")
            break
        if status["status"] == "completed":
            print("FAILED: Download completed unexpectedly.")
            return False

        time.sleep(0.5)

    return True

def test_concurrency_prevention():
    print("\n--- Test Case 3: Concurrency Prevention ---")

    # Start a download that takes a few seconds
    # Using httpbin delay to simulate a slow download
    url = "https://httpbin.org/delay/3"
    name = "test_concurrency"

    payload = {
        "url": url,
        "name": name,
        "source": "Test"
    }

    print("Starting first download...")
    resp1 = requests.post(f"{BASE_URL}/api/models/download", json=payload)
    if resp1.status_code != 200:
        print(f"FAILED: First request failed: {resp1.status_code}")
        return False

    print("Attempting second download immediately...")
    resp2 = requests.post(f"{BASE_URL}/api/models/download", json=payload)

    if resp2.status_code == 400:
        print("SUCCESS: Second download rejected with 400 (as expected).")
        print(f"Message: {resp2.json()['detail']}")
    else:
        print(f"FAILED: Second download returned {resp2.status_code}, expected 400.")
        return False

    # Wait for the first one to finish to clean up
    print("Waiting for first download to finish...")
    while True:
        status_resp = requests.get(f"{BASE_URL}/api/models/download/status")
        status = status_resp.json()
        if status["status"] in ["completed", "error"]:
            break
        time.sleep(1)

    # Cleanup
    target_file = os.path.join(MODELS_DIR, f"{name}.safetensors")
    if os.path.exists(target_file):
        os.remove(target_file)

    return True

if __name__ == "__main__":
    if not wait_for_backend():
        print("Could not connect to backend. Please ensure it is running.")
        sys.exit(1)

    check_no_active_download()

    results = []
    results.append(test_download_success())
    results.append(test_download_error())
    results.append(test_concurrency_prevention())

    if all(results):
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
