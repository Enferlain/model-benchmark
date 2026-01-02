from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from ..services import prompt_manager as data_loader

router = APIRouter()

@router.get("/prompts")
def get_prompts():
    """Get all prompts with metadata for the manager."""
    return data_loader.get_all_prompts_metadata()

@router.post("/prompts/create")
async def create_prompt_multipart(
    text: str = Form(...), image: UploadFile = File(None)
):
    image_bytes = None
    if image:
        image_bytes = await image.read()

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
        try:
            data_loader.update_prompt_text(filename, payload["text"])
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

    data_loader.set_all_prompts_enabled(enabled)
    return {"status": "success"}
