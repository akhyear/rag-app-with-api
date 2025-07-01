FROM python:3.13-slim


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000
ENV GROQ_API_KEY=""
ENV HUGGINGFACEHUB_ACCESS_TOKEN=""
ENV HUGGINGFACE_API_TOKEN=""
ENV PINECONE_API_KEY=""
ENV SUPABASE_URL=""
ENV ANON_PUBLIC_KEY=""


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE $PORT


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]