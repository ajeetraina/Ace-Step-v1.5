# ACE-Step v1.5 - Apple Silicon ARM64 Optimized Dockerfile
# Multi-platform support with ARM64 optimization
FROM --platform=$BUILDPLATFORM python:3.11-slim

# Set build arguments for multi-platform builds
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Building on $BUILDPLATFORM, targeting $TARGETPLATFORM"

# Set environment variables optimized for Apple Silicon/ARM64
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCHAUDIO_USE_TORCHCODEC=0 \
    # Apple Silicon MPS optimizations
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    # ARM64 specific optimizations
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4

# Install system dependencies with ARM64 optimizations
# build-essential is required for triton to compile CUDA kernels and ARM64 native extensions
# ffmpeg and libav* dev packages are required for torchaudio's ffmpeg backend
# Note: torchaudio's ffmpeg backend needs shared libraries, not just the ffmpeg binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        build-essential \
        pkg-config \
        # ARM64 optimized audio libraries
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libsndfile1-dev \
        # ARM64 optimized math libraries
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        # Additional ARM64 optimizations
        libomp-dev \
        && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up a new user named "user" with user ID 1000 (HuggingFace Space requirement)
RUN useradd -m -u 1000 user

# Create /data directory with proper permissions for persistent storage
RUN mkdir -p /data && chown user:user /data && chmod 755 /data

# Set environment variables for user with ARM64 optimizations
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    # ARM64 specific library paths
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first for better Docker layer caching
COPY --chown=user:user requirements.txt .

# Copy the local nano-vllm package
COPY --chown=user:user acestep/third_parts/nano-vllm ./acestep/third_parts/nano-vllm

# Switch to user before installing packages
USER user

# Install dependencies from requirements.txt with ARM64 optimizations
# Use --prefer-binary to prefer wheel packages when available for ARM64
RUN pip install --no-cache-dir --user --prefer-binary --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user --prefer-binary -r requirements.txt

# Install nano-vllm with --no-deps since all dependencies are already installed
RUN pip install --no-deps ./acestep/third_parts/nano-vllm

# Copy the rest of the application
COPY --chown=user:user . .

# Create app.py entry point if missing
RUN if [ ! -f app.py ]; then \
    echo "#!/usr/bin/env python3" > app.py && \
    echo "from acestep.acestep_v15_pipeline import main" >> app.py && \
    echo "if __name__ == '__main__':" >> app.py && \
    echo "    main()" >> app.py && \
    chmod +x app.py; \
    fi

# Expose port
EXPOSE 7860

# Health check for ARM64 compatibility
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application with ARM64 optimized settings
CMD ["python", "app.py", "--server-name", "0.0.0.0", "--port", "7860", "--device", "auto"]