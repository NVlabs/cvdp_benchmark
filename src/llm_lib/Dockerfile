FROM nvidia/cvdp-sim:v1.0.0

WORKDIR /app

# Install build-essential for gcc and g++
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pydantic openai python-dotenv tiktoken pytest
