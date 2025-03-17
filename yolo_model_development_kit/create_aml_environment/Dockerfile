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

# Install UV package manager
RUN pip install uv

COPY pyproject.toml .

# Install requirements
# Note: model_export requires python <= 3.11
RUN uv pip install --system -r pyproject.toml --extra model_export
