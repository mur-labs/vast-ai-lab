# Custom llama.cpp inference container for Vast.ai spot instances
# Base image: Official CUDA-enabled llama.cpp server
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

# Install dependencies for fast multi-connection downloading and network certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy custom entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default environment configuration
ENV HOST=0.0.0.0
ENV PORT=8000
ENV N_GPU_LAYERS=99
ENV CTX_SIZE=4096

# Expose internal inference port
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
