# Use an official NVIDIA CUDA runtime image as a parent image
# Make sure the CUDA version matches the PyTorch build (cu118 in requirements.txt)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and git
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.12 \\
    python3.12-venv \\
    python3-pip \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Specify the command to run on container start
# This will likely be python src/main.py with relevant arguments
# Placeholder for now
CMD ["python", "src/main.py"]