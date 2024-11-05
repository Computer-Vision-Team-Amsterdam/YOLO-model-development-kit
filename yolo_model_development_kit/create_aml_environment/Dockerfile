FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS base-image

# Upgrade and install system libraries
RUN apt update -y \
    && apt upgrade -y \
    && apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY pyproject.toml .

# Install UV package manager

# NOTE: the default way doesn't work because the url is not whitelisted on AML
# # Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh

# This works
RUN python3 -m pip install uv

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.cargo/bin/:$PATH"

# NOTE: this doesn't work because UV downloads python from `https://github.com/indygreg/python-build-standalone` which is not whitelisted on AML
# Create environment and install packages
RUN uv venv --python 3.12 \
    # && echo "source .venv/bin/activate" > ~/.bashrc \
    # && source .venv/bin/activate \
    && uv pip install -r pyproject.toml

ENV PATH="/opt/app/.venv/bin:$PATH"
