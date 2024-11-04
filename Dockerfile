# Use the NVIDIA CUDA image as a base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install necessary dependencies
RUN apt-get update -qq && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install the necessary dependencies for the detectron2 library
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0

# Add TCMalloc to the container for memory management
RUN apt-get install libgoogle-perftools4 libtcmalloc-minimal4 -y

# Create a non-root user and add them to the sudo group
RUN useradd -m -s /bin/bash myuser && echo "myuser:myuser" | chpasswd && adduser myuser sudo

# Download and install Miniconda
ENV CONDA_DIR /opt/miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh
RUN /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
RUN rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Set environment variables for conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda
RUN /opt/miniconda/bin/conda init bash

# Switch to the non-root user
USER myuser

# Set the working directory
WORKDIR /home/myuser/SMPLitex

# Create a conda environment
RUN conda init && conda create -n myenv python=3.10 -y
RUN conda install -n myenv -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -n myenv pytorch3d=0.7.0 -c pytorch3d -y

# Copy the current directory contents into the container at /home/myuser/SMPLitex
COPY . .

# combine the chunked model weights into one zip file and unzip in the simplitex-trained-model directory
RUN cat split.zip.001 split.zip.002 split.zip.003 > SMPLitex-v1.0.zip
RUN mkdir -p simplitex-trained-model && cd /home/myuser/SMPLitex/simplitex-trained-model
RUN unzip SMPLitex-v1.0.zip -d ./simplitex-trained-model

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN python3 -m ensurepip --upgrade && \
    pip install -r requirements.txt

RUN pip install git+https://github.com/facebookresearch/detectron2.git
RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

RUN pip install av
