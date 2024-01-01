import base64
import io
import torch
from diffusers import PixArtAlphaPipeline
from litestar import Litestar, post

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)

@post("/")
def generate_image(data: dict[str, str]) -> bytes:
    prompt = dict["prompt"]
    image = pipe(prompt).images[0]
    buf = io.BytesIO()
    image.save(buf, "JPEG")
    buf.seek(0)
    b = buf.read()
    image_b64 = base64.b64encode(b)
    return image_b64

app = Litestar([generate_image])