# Use the NVIDIA CUDA image as a base
FROM ubuntu:22.04 as builder

# Install necessary dependencies
RUN apt-get update -qq && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    git \
    unzip \
    g++ \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*
# Install the necessary dependencies for the detectron2 library (libgl1-mesa-glx, libglib2.0-0)
# Add TCMalloc to the container for memory management (libgoogle-perftools4 libtcmalloc-minimal4)

# Create a non-root user and add them to the sudo group
RUN useradd -m -s /bin/bash myuser && echo "myuser:myuser" | chpasswd && adduser myuser sudo

# Clone the necessary repositories
RUN git clone https://github.com/cxgincsu/SemanticGuidedHumanMatting.git /home/myuser/SemanticGuidedHumanMatting && \
    rm -rf /home/myuser/SemanticGuidedHumanMatting/.git && \
    git clone https://github.com/facebookresearch/detectron2.git /home/myuser/detectron2 && \
    rm -rf /home/myuser/detectron2/.git && \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /home/myuser/stable-diffusion-webui && \
    rm -rf /home/myuser/stable-diffusion-webui/.git

# Change ownership of the directories to myuser
RUN chown -R myuser:myuser /home/myuser/SemanticGuidedHumanMatting /home/myuser/detectron2 /home/myuser/stable-diffusion-webui

# Download and install Miniconda
ENV CONDA_DIR /opt/miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh && \
    /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Set environment variables for conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda
RUN /opt/miniconda/bin/conda init bash
RUN conda update -n base -c defaults conda -y

# Switch to the non-root user
USER myuser

# Set the working directory
WORKDIR /home/myuser/SMPLitex

# Create a conda environment
RUN conda init && conda create -n smplitex python=3.10 -y

# start every shell with the conda environment activated
SHELL ["conda", "run", "-n", "smplitex", "/bin/bash", "-c"]

# Copy the requirements.txt file into the container at /home/myuser/SMPLitex
COPY --chown=myuser:myuser requirements.txt .

RUN conda install -n smplitex \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    pytorch-cuda=12.4 \
    -c pytorch \
    -c nvidia

RUN pip install -r requirements.txt

# Install the detectron2 library and the DensePose project
RUN pip install git+https://github.com/facebookresearch/detectron2.git \
    git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Install av and webuiapi
RUN pip install av webuiapi

# Copy the scripts directory into the container at /home/myuser/SMPLitex/scripts
COPY --chown=myuser:myuser scripts ./scripts

# Move the cloned repositories into the scripts directory
RUN mv /home/myuser/SemanticGuidedHumanMatting ./scripts/SemanticGuidedHumanMatting && \
    mv /home/myuser/detectron2 ./scripts/detectron2 && \
    mv /home/myuser/stable-diffusion-webui ./scripts/stable-diffusion-webui

# Move ./SGHM-RestNet50.pth into the current directory
COPY --chown=myuser:myuser SGHM-ResNet50.pth .

# Move SGHM-ResNet50.pth into the pretrained directory of the SemanticGuidedHumanMatting repository
RUN mkdir -p ./scripts/SemanticGuidedHumanMatting/pretrained && \
    mv ./SGHM-ResNet50.pth ./scripts/SemanticGuidedHumanMatting/pretrained/

# Move split.zip.001, split.zip.002, and split.zip.003 into the current directory
COPY --chown=myuser:myuser split.zip.001 .
COPY --chown=myuser:myuser split.zip.002 .
COPY --chown=myuser:myuser split.zip.003 .

# combine the chunked model weights into one zip file and unzip in the simplitex-trained-model directory
RUN cat split.zip.001 split.zip.002 split.zip.003 > SMPLitex_weights.zip && \
    rm split.zip.001 split.zip.002 split.zip.003 && \
    mkdir -p smplitex-trained-model && \
    unzip SMPLitex_weights.zip -d ./smplitex-trained-model && \
    cd ./smplitex-trained-model && \
    unzip SMPLitex-v1.0.zip && \
    rm SMPLitex-v1.0.zip && \
    cd .. && \
    rm SMPLitex_weights.zip

# Move the SMPLitex-v1.0.ckpt.001, SMPLitex-v1.0.ckpt.002, and SMPLitex-v1.0.ckpt.003 into the current directory
COPY --chown=myuser:myuser SMPLitex-v1.0.ckpt.001 .
COPY --chown=myuser:myuser SMPLitex-v1.0.ckpt.002 .
COPY --chown=myuser:myuser SMPLitex-v1.0.ckpt.003 .

RUN cat SMPLitex-v1.0.ckpt.001 SMPLitex-v1.0.ckpt.002 SMPLitex-v1.0.ckpt.003 > SMPLitex-v1.0.ckpt.zip & \
    rm SMPLitex-v1.0.ckpt.001 SMPLitex-v1.0.ckpt.002 SMPLitex-v1.0.ckpt.003 & \
    unzip SMPLitex-v1.0.ckpt.zip -d scripts/stable-diffusion-webui/models/Stable-diffusion/ & \
    rm SMPLitex-v1.0.ckpt.zip

# # Copy the current directory contents into the container at /home/myuser/SMPLitex
# COPY --chown=myuser:myuser . .

# Copy sample-data directory into the current directory
COPY --chown=myuser:myuser sample-data ./sample-data

# Create a conda environment
RUN conda init && conda create -n pytorch3d python=3.10 -y

# start every shell with the conda environment activated
SHELL ["conda", "run", "-n", "pytorch3d", "/bin/bash", "-c"]

RUN conda install -n pytorch3d \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    pytorch-cuda=12.4 \
    yacs \
    iopath \
    pytorch3d \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    -c iopath \
    -c pytorch3d -y

RUN pip install smplx imageio scipy git+https://github.com/mattloper/chumpy

WORKDIR /home/myuser/SMPLitex/scripts
