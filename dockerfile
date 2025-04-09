FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update

RUN pip install transformers datasets evaluate rotary_embedding_torch \
    accelerate>=0.26.0 bitsandbytes

WORKDIR /home
COPY proj.py .

CMD ["python", "proj.py"]
