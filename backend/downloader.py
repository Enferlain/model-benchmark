import os
import requests
import re
import traceback
from typing import Optional
import data_loader
from state import download_state, download_state_lock, models_db, ModelResult

def download_model_task(url: str, name: str, source: str, api_token: Optional[str] = None):
    global download_state

    with download_state_lock:
        download_state["current_file"] = name
        download_state["progress"] = 0
        download_state["total"] = 0
        download_state["status"] = "downloading"
        download_state["error"] = None

    try:
        print(f"Starting download: {url}")
        timeout = int(os.environ.get("REQUEST_TIMEOUT", 30))

        # Helper: Transform URL for HuggingFace blob -> resolve
        if "huggingface.co" in url and "/blob/" in url:
            print("Detected HuggingFace /blob/ URL, converting to /resolve/...")
            url = url.replace("/blob/", "/resolve/")

        headers = {
            "User-Agent": "ModelBenchmarkExplorer/1.0"
        }
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        # Helper: Resolve Civitai Model ID to Download URL
        if "civitai.com/models/" in url and "api/download" not in url:
            try:
                # Extract potential ID
                match = re.search(r"civitai\.com/models/(\d+)", url)
                if match:
                    model_id = match.group(1)
                    print(f"Detected Civitai Model ID {model_id}, attempting to resolve download URL via API...")
                    api_url = f"https://civitai.com/api/v1/models/{model_id}"

                    api_resp = requests.get(api_url, headers=headers, timeout=10)
                    if api_resp.ok:
                        data = api_resp.json()
                        if "modelVersions" in data and len(data["modelVersions"]) > 0:
                            # Sort by createdAt descending to ensure we get the latest
                            try:
                                from datetime import datetime
                                versions = sorted(
                                    data["modelVersions"],
                                    key=lambda x: datetime.fromisoformat(x["createdAt"].replace("Z", "+00:00")),
                                    reverse=True
                                )
                            except (ValueError, TypeError):
                                print("Warning: Could not parse createdAt for version sorting. Using default order.")
                                versions = data["modelVersions"]

                            # Use the first (latest) version's download URL
                            download_url = versions[0].get("downloadUrl")
                            if download_url:
                                print(f"Resolved to: {download_url}")
                                url = download_url
                            else:
                                print("No downloadUrl found in API response.")
                        else:
                            print("No modelVersions found in API response.")
                    else:
                        print(f"Civitai API lookup failed: {api_resp.status_code}")
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                print(f"Error resolving Civitai URL: {e}")

        response = requests.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with download_state_lock:
            download_state["total"] = total_size

        # Determine filename
        filename = None
        if "content-disposition" in response.headers:
             cd = response.headers["content-disposition"]
             # Parse Content-Disposition manually to avoid regex greediness
             parts = cd.split(";")
             for part in parts:
                 part = part.strip()
                 if part.lower().startswith("filename="):
                     filename = part[9:].strip().strip('"').strip("'")
                     break

        if not filename:
            filename = url.split("/")[-1]
            if "?" in filename:
                filename = filename.split("?")[0]

        # Basic sanitization for extension check
        if not filename or (not filename.endswith(".safetensors") and not filename.endswith(".ckpt")):
             filename = f"{name or 'model'}.safetensors"

        # Safe filename generation (path traversal prevention)
        filename = os.path.basename(filename) # Strip directory components
        # Whitelist safe characters only
        filename = "".join([c for c in filename if c.isalnum() or c in "._- "])
        # Ensure it has a valid extension (again, after sanitization)
        if not (filename.endswith(".safetensors") or filename.endswith(".ckpt")):
             filename += ".safetensors"

        save_path = data_loader.MODELS_DIR / filename

        # Verify save path is within MODELS_DIR
        try:
             save_path = save_path.resolve()
             models_dir_resolved = data_loader.MODELS_DIR.resolve()
             if not str(save_path).startswith(str(models_dir_resolved)):
                 raise ValueError(f"Invalid path: {save_path}")
        except Exception as e:
             raise ValueError(f"Path verification failed: {e}") from e

        print(f"Saving to: {save_path}")

        downloaded = 0
        last_update = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress every 1MB to reduce lock contention
                    if downloaded - last_update > 1024 * 1024:
                        with download_state_lock:
                            download_state["progress"] = downloaded
                        last_update = downloaded

        # Final update
        with download_state_lock:
            download_state["progress"] = downloaded
            download_state["status"] = "completed"

        print("Download complete.")

        # Add to models_db
        new_model = {
            "id": save_path.stem,
            "name": name or save_path.stem.replace("-", " ").title(),
            "source": source,
            "url": url,
            "path": str(save_path),
            "accuracy": 0.0,
            "diversity": 0.0,
            "rating": 0.0,
            "metrics": {"accuracy": 0.0, "diversity": 0.0}
        }

        # Check if exists
        exists = False
        for m in models_db:
             if m.id == new_model["id"]:
                 exists = True
                 break
        if not exists:
             models_db.append(ModelResult(**new_model))

    except Exception as e:
        print(f"Download error details: {traceback.format_exc()}")
        with download_state_lock:
            download_state["status"] = "error"
            download_state["error"] = str(e)
        print(f"Download error: {e}")
    finally:
        with download_state_lock:
            download_state["is_downloading"] = False
