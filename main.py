import base64
import io
import torch
from diffusers import PixArtAlphaPipeline
from litestar import Litestar, post
from pydantic import BaseModel

class Prompt(BaseModel):
    prompt: str

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)

@post("/")
def generate_image(data: Prompt) -> bytes:
    try:
        print(data)
        prompt = data.prompt
        print(prompt)
        image = pipe(prompt).images[0]
        # print(image)
        buf = io.BytesIO()
        # print(buf)
        image.save(buf, "JPEG")
        buf.seek(0)
        b = buf.read()
        image_b64 = base64.b64encode(b)
        # print(image_b64)
        return image_b64
    except Exception as e:
        print(e)

app = Litestar([generate_image])