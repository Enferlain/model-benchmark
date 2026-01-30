"""
Image Service - Thumbnail Generation & Caching

Provides on-the-fly thumbnail generation with on-disk caching.
Uses sync functions (run in threadpool by FastAPI) to avoid blocking.
"""

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..services.prompt_manager import ASSETS_DIR

# Thumbnail cache directory
THUMBNAIL_CACHE_DIR = ASSETS_DIR / ".cache" / "thumbnails"


def get_thumbnail(
    relative_path: str,
    width: int = 150,
    height: int = 150,
    quality: int = 80,
) -> tuple[bytes, str]:
    """
    Generate or retrieve a cached thumbnail for an image.

    Args:
        relative_path: Path relative to ASSETS_DIR (e.g., "image_prompts/test.png")
        width: Target width
        height: Target height
        quality: WebP quality (1-100)

    Returns:
        Tuple of (thumbnail_bytes, media_type)

    Raises:
        FileNotFoundError: If source image doesn't exist
        ValueError: If path escapes ASSETS_DIR (security)
    """
    # 1. Security: Resolve and validate path
    source_path = (ASSETS_DIR / relative_path).resolve()

    # Ensure the resolved path is within ASSETS_DIR
    try:
        source_path.relative_to(ASSETS_DIR.resolve())
    except ValueError:
        raise ValueError(f"Path escapes assets directory: {relative_path}") from None

    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {relative_path}")

    if not source_path.is_file():
        raise ValueError(f"Path is not a file: {relative_path}")

    # 2. Generate cache key (hash of path + dimensions)
    cache_key = hashlib.md5(f"{relative_path}:{width}x{height}:{quality}".encode()).hexdigest()
    cache_path = THUMBNAIL_CACHE_DIR / f"{cache_key}.webp"

    # 3. Check cache
    if cache_path.exists():
        return cache_path.read_bytes(), "image/webp"

    # 4. Generate thumbnail
    THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as img:
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Use thumbnail() for efficient resizing (maintains aspect ratio, fits within box)
        img.thumbnail((width, height), Image.Resampling.LANCZOS)

        # Save to buffer and cache
        buffer = BytesIO()
        img.save(buffer, format="WEBP", quality=quality)
        thumb_bytes = buffer.getvalue()

    # Write to cache
    cache_path.write_bytes(thumb_bytes)

    return thumb_bytes, "image/webp"


def clear_thumbnail_cache() -> int:
    """
    Clear all cached thumbnails.

    Returns:
        Number of files deleted
    """
    if not THUMBNAIL_CACHE_DIR.exists():
        return 0

    count = 0
    for f in THUMBNAIL_CACHE_DIR.glob("*.webp"):
        f.unlink()
        count += 1

    return count
