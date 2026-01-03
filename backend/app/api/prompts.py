from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from PIL import Image
import io
from ..services import prompt_manager as data_loader

router = APIRouter()

MAX_TEXT_LENGTH = 10_000
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB

@router.get("/prompts")
def get_prompts():
    """Get all prompts with metadata for the manager."""
    return data_loader.get_all_prompts_metadata()

@router.post("/prompts/create")
async def create_prompt_multipart(
    text: str = Form(...), image: UploadFile = File(None)
):
    # Validate Text Length
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
        )

    image_bytes = None
    if image:
        # Stream read to check size limit
        content = bytearray()
        size = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        while True:
            chunk = await image.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image size exceeds maximum limit of {MAX_IMAGE_SIZE // (1024*1024)} MB."
                )
            content.extend(chunk)

        image_bytes = bytes(content)

        # Validate Image Type using Pillow
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()  # Verify it's a valid image
                format = img.format
                if format not in ["PNG", "JPEG", "WEBP"]:
                     # Optional: restrict formats if needed, or allow any Pillow supports
                     pass
        except Exception:
             raise HTTPException(status_code=400, detail="Invalid image file.")

    new_id = data_loader.save_new_prompt(
        text, image_bytes, image.filename if image else None
    )
    return {"status": "success", "id": new_id}

@router.put("/prompts/{filename}")
def update_prompt(filename: str, payload: dict = Body(...)):
    # Handle partial updates
    updated = False

    if "enabled" in payload:
        data_loader.toggle_prompt_active(filename, payload["enabled"])
        updated = True

    if "alias" in payload:
        data_loader.set_prompt_alias(filename, payload["alias"])
        updated = True

    if "text" in payload:
        new_text = payload["text"]
        if len(new_text) > MAX_TEXT_LENGTH:
             raise HTTPException(
                status_code=400,
                detail=f"Prompt text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
            )
        try:
            data_loader.update_prompt_text(filename, new_text)
            updated = True
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Prompt not found") from None

    if not updated:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    return {"status": "success"}

@router.delete("/prompts/{filename}")
def delete_prompt(filename: str):
    try:
        data_loader.delete_prompt(filename)
        return {"status": "success"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Prompt not found")

@router.post("/prompts/reorder")
def reorder_prompts(payload: dict = Body(...)):
    order = payload.get("order")
    if not order or not isinstance(order, list):
        raise HTTPException(status_code=400, detail="Order list is required")

    data_loader.save_prompt_order(order)
    return {"status": "success"}

@router.post("/prompts/shuffle")
def shuffle_prompts():
    data_loader.shuffle_prompts_order()
    return {"status": "success"}

@router.post("/prompts/bulk/enable")
def bulk_enable_prompts(payload: dict = Body(...)):
    enabled = payload.get("enabled")
    if enabled is None:
        raise HTTPException(status_code=400, detail="enabled field is required")

    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")

    data_loader.set_all_prompts_enabled(enabled)
    return {"status": "success"}
