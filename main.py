from time import sleep
import json
import base64
import io
import torch
from diffusers import PixArtAlphaPipeline
from litestar import Litestar, post
from pydantic import BaseModel
import httpx

class Prompt(BaseModel):
    prompt: str

# pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float32)
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float32, use_safetensors=True)
#pipe.text_encoder.to_bettertransformer()
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

@post("/", media_type="image/jpeg")
def generate_image(data: Prompt) -> bytes:
    try:
        print(data)
        prompt = data.prompt
        print(prompt)
        image = pipe(prompt, num_inference_steps=14, width=512, height=512).images[0]
        # print(image)
        buf = io.BytesIO()
        # print(buf)
        image.save(buf, "JPEG")
        buf.seek(0)
        b = buf.read()
        # image_b64 = base64.b64encode(b)
        # print(image_b64)
        # return image_b64
        return b
    except Exception as e:
        print(e)


app = Litestar([generate_image])

if __name__ == "__main__":
    URL = "https://pluckwork.k3s.koski.co"
    while True:
        reserved = httpx.get(URL + "/task")
        try:
            task = reserved.json()
            print(task)
            id = task["id"]
            print(id)
            prompt_dict_str = base64.base64decode(task["input"])
            print(prompt_dict_str)
            prompt_dict = json.loads(prompt_dict_str)
            prompt = Prompt(**prompt_dict)
            print(prompt)
            print("Generating image")
            image = generate_image(prompt)
            print("Writing back")
            httpx.post(URL + "/task", params={"id": id}, body=image)
        except Exception as e:
            print(e)
        
        sleep(10)