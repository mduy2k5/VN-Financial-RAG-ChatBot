FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY sample_data ./sample_data

EXPOSE 8090

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "financial_rag.api.app:app", "--host", "0.0.0.0", "--port", "8090"]
