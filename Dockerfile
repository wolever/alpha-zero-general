FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g. for numpy/pytorch if needed, or gsutil/gcloud if we need them inside??
# actually best to use python libraries for GCS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# We can use uv if preferred, or just pip. The user mentioned instructions to use 'uv', so let's try to stick to that or pip.
# Given it's a simple Dockerfile, regular pip is often easier unless 'uv' is desired for speed.
# Let's use pip for simplicity unless 'uv' is explicitly required by the project structure (e.g. lock file).
# The user has 'uv.lock' in the directory. We should probably use 'uv'.

COPY pyproject.toml uv.lock ./

RUN pip install uv && \
    uv pip install --system --no-cache-dir -r pyproject.toml

# Copy source code
COPY . .

# Expose port (Cloud Run sets PORT env var, defaulting to 8080 usually)
ENV PORT=8189
EXPOSE $PORT

# Run the server
CMD ["python", "server.py"]
