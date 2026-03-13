import aiofiles
import base64
import io
import mimetypes


async def encode_image_to_base64(image_path: str) -> str:
    """Encode an image to base64 asynchronously."""
    async with aiofiles.open(image_path, "rb") as image_file:
        image_data = await image_file.read()
    return base64.b64encode(image_data).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Guess the MIME type for an image path."""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/png"


async def save_image(image, image_path: str):
    """Save an image to a file asynchronously."""
    with io.BytesIO() as buffer:
        image.save(buffer, format=image.format)
        image_data = buffer.getvalue()

    async with aiofiles.open(image_path, "wb") as f:
        await f.write(image_data)
