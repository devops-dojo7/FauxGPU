FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir pyyaml
COPY engine ./engine
COPY k3s/trainer/train.py ./train.py
CMD ["python", "train.py"]
