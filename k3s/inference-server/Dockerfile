FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pyyaml fastapi "uvicorn[standard]" langfuse
COPY engine ./engine
COPY api/__init__.py ./api/__init__.py
COPY api/langfuse_client.py ./api/langfuse_client.py
COPY k3s/inference-server/server.py ./server.py
EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
