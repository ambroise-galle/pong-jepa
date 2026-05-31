# Use standard high-performance PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies for OpenCV and Gymnasium Atari
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workspace
WORKDIR /workspace

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Import Atari ROMs
RUN AutoROM --accept-license

# Copy repository source code
COPY . .

# Set default training command (can be overridden during run)
CMD ["bash", "run_training.sh", "--dry-run"]
