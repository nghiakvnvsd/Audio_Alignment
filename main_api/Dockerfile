FROM python:3.8-slim
WORKDIR /app

EXPOSE 8000
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt install --no-install-recommends -y build-essential python3-dev gcc libpcap-dev ffmpeg libsm6 libxext6 git

RUN pip install --upgrade pip
RUN pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt && rm -rf /root/.cache/pip

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
