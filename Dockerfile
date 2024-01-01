FROM python:3.11

WORKDIR /app
RUN pip3 install -U  torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -U diffusers transformers sentencepiece accelerate 
COPY . .
RUN python setup.py
CMD python main.py
