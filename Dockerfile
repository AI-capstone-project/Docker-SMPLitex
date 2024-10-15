# Use the NVIDIA CUDA image as a base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash myuser && echo "myuser:myuser" | chpasswd && adduser myuser sudo

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh
RUN /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
RUN rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Set environment variables for conda
ENV PATH /opt/miniconda/bin:$PATH

# Initialize conda
RUN /opt/miniconda/bin/conda init bash

# Switch to the non-root user
USER myuser
WORKDIR /home/myuser/SMPLitex

# Create a conda environment
RUN /bin/bash -c "source /opt/miniconda/bin/activate && conda create -n myenv python=3.10 -y"

COPY . .

# Activate the conda environment (optional)
RUN echo "conda activate myenv" >> ~/.bashrc

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Run a command to verify conda installation
CMD ["bash"]

RUN /bin/bash -c "source /opt/miniconda/bin/activate myenv && \
    python3 -m ensurepip --upgrade && \
    pip install -r requirements.txt && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && \
    conda install pytorch3d=0.7.0 -c pytorch3d -y"

# Download detectron2 from the GitHub repository
WORKDIR /home/myuser/SMPLitex/scripts

# install detectron2 and densepose
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN /bin/bash -c "source /opt/miniconda/bin/activate myenv && pip install git+https://github.com/facebookresearch/detectron2.git"
RUN /bin/bash -c "source /opt/miniconda/bin/activate myenv && pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose"

# Install the necessary dependencies for the detectron2 library
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN RUN /bin/bash -c "source /opt/miniconda/bin/activate myenv && pip3 install av"

# Download the SemanticGuidedHumanMatting repository for computing the silhouette of the subject
RUN git clone https://github.com/cxgincsu/SemanticGuidedHumanMatting.git
WORKDIR /app/scripts/SemanticGuidedHumanMatting
RUN mkdir pretrained

WORKDIR /app/scripts/

# need to manually download the pretrained model from the link below
# https://drive.google.com/drive/folders/15mGzPJQFEchaZHt9vgbmyOy46XxWtEOZ

# Download the stable-diffusion-webui repository for the web interface
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
RUN /bin/bash -c "source /opt/miniconda/bin/activate myenv && pip install webuiapi"

# Add TCMalloc to the container for memory management
RUN apt-get install libgoogle-perftools4 libtcmalloc-minimal4 -y
