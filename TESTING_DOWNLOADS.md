# Guide to Testing Model Downloads

This guide explains how to test the model downloading functionality of the backend, which supports downloading models from URLs (including Hugging Face and CivitAI).

## Prerequisites

1.  **Backend Environment**: Ensure you have the necessary Python dependencies installed.
    ```bash
    pip install -r backend/requirements.txt
    ```
    (Note: `requests` is required for the test script).

2.  **Running the Backend**: The backend must be running for the tests to work.
    ```bash
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Automated Testing Script

We have provided a Python script `backend/test_model_download.py` that automatically verifies the following scenarios:

1.  **Successful Download**: Downloads a small dummy file and verifies it is saved with the correct extension in `backend/assets/models/`.
2.  **Error Handling**: Attempts to download from an invalid URL and verifies the error status is reported.
3.  **Concurrency Control**: Verifies that only one download can happen at a time (returns 400 if another is in progress).

### Running the Test Script

With the backend running in a separate terminal (or background), run:

```bash
python backend/test_model_download.py
```

If successful, you will see `ALL TESTS PASSED`.

## Manual Testing (cURL / Postman)

You can also manually test the endpoints using cURL.

### 1. Trigger a Download

**Endpoint**: `POST /api/models/download`

```bash
curl -X POST "http://localhost:8000/api/models/download" \
     -H "Content-Type: application/json" \
     -d '{
           "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
           "name": "sd-v1-5",
           "source": "HuggingFace"
         }'
```

*Note: The URL above is large (4GB). For testing, use a smaller file or the dummy URL from the script.*

### 2. Check Status

**Endpoint**: `GET /api/models/download/status`

```bash
curl "http://localhost:8000/api/models/download/status"
```

**Response Example**:
```json
{
  "is_downloading": true,
  "current_file": "sd-v1-5",
  "progress": 10485760,
  "total": 4265380512,
  "status": "downloading",
  "error": null
}
```

### 3. Verify File System

After the status becomes `"completed"`, verify the file exists:

```bash
ls -l backend/assets/models/
```

You should see `sd-v1-5.safetensors`.

## Test Cases (TC)

| TC ID | Description | Pre-conditions | Action | Expected Result |
|-------|-------------|----------------|--------|-----------------|
| **TC1** | Download Valid Model | Backend running | POST /download with valid URL | Status: "started". Backend downloads file. Final status: "completed". File exists in `assets/models`. |
| **TC2** | Invalid URL | Backend running | POST /download with bad URL | Status: "started". Status endpoint shows "error" with details. |
| **TC3** | Concurrent Download | Download in progress | POST /download | Response 400 "Download already in progress". |
| **TC4** | Filename Sanitization | Backend running | POST /download with `name="my/bad/../name"` | File saved as `mybadname.safetensors` (or similar safe name) in correct dir. |
| **TC5** | CivitAI Download | Backend running | POST /download with CivitAI download link | File downloaded correctly. Content-Disposition header used for filename if available. |

## Troubleshooting

*   **Port in use**: If port 8000 is used, kill the process or change the port in the `uvicorn` command and the test script.
*   **Permissions**: Ensure the `backend/assets/models` directory is writable.
