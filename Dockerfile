FROM python:3.11

WORKDIR /app
RUN pip3 install -U  torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -U diffusers transformers sentencepiece accelerate litestar uvicorn
COPY . .
EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port 8080
