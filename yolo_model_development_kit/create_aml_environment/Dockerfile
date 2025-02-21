FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04 AS base-image

# Upgrade and install system libraries
RUN apt update -y \
    && apt upgrade -y \
    && apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libgl1 \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY pyproject.toml .

# Install UV package manager
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Create environment and install packages
# NOTE: tensorrt export requires python <= 3.11
RUN uv venv --python 3.11 \
    && uv pip install -r pyproject.toml --extra model_export

ENV PATH="/opt/app/.venv/bin:$PATH"
