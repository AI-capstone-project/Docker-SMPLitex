#!/bin/bash

# Add an ID argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ID>"
    exit 1
fi

# Set the ID
ID=$1

# If a file with the name SMPLitex-v1.0.ckpt is not in 
# /home/myuser/SMPLitex/scripts/stable-diffusion-webui/stable-diffusion-webui/models/Stable-diffusion/ directory, 
# then copy the file from /home/myuser/SMPLitex/scripts/stable-diffusion-webui/models/Stable-diffusion/ to 
# /home/myuser/SMPLitex/scripts/stable-diffusion-webui/stable-diffusion-webui/models/Stable-diffusion/
echo "Copying Stable-diffusion model..."
if [ ! -f /home/myuser/SMPLitex/scripts/stable-diffusion-webui/stable-diffusion-webui/models/Stable-diffusion/SMPLitex-v1.0.ckpt ]; then
    rm /home/myuser/SMPLitex/scripts/stable-diffusion-webui/stable-diffusion-webui/models/Stable-diffusion/*
    cp /home/myuser/SMPLitex/scripts/stable-diffusion-webui/models/Stable-diffusion/SMPLitex-v1.0.ckpt /home/myuser/SMPLitex/scripts/stable-diffusion-webui/stable-diffusion-webui/models/Stable-diffusion/
fi

# Remove any existing dummy data
echo "Removing existing dummy data..."
rm -r /home/myuser/SMPLitex/scripts/dummy_data/densepose_pkl /home/myuser/SMPLitex/scripts/dummy_data/uv-textures-inpainted /home/myuser/SMPLitex/scripts/dummy_data/debug /home/myuser/SMPLitex/scripts/dummy_data/uv-textures /home/myuser/SMPLitex/scripts/dummy_data/uv-textures-masks /home/myuser/SMPLitex/scripts/dummy_data/densepose /home/myuser/SMPLitex/scripts/dummy_data/densepose-masked /home/myuser/SMPLitex/scripts/dummy_data/images-seg

# Remove all files in stableviton-created_images excluding the last modified file
echo "Removing all files in stableviton-created_images excluding the last modified file..."
ls -t /home/myuser/SMPLitex/scripts/dummy_data/stableviton-created_images | tail -n +2 | xargs -I {} rm /home/myuser/SMPLitex/scripts/dummy_data/stableviton-created_images/{}

# Run image_to_densepose.py
echo "Running image_to_densepose.py..."
conda run -n smplitex python image_to_densepose.py --detectron2 ./detectron2 --input_folder ./dummy_data/stableviton-created_images || { echo "Error running image_to_densepose.py. Exiting."; exit 1; }

# Run SemanticGuidedHumanMatting/test_image.py
echo "Running SemanticGuidedHumanMatting/test_image.py..."
conda run -n smplitex python SemanticGuidedHumanMatting/test_image.py --images-dir ./dummy_data/stableviton-created_images --result-dir ./dummy_data/images-seg --pretrained-weight SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth || { echo "Error running SemanticGuidedHumanMatting/test_image.py. Exiting."; exit 1; }

# Run compute_partial_texturemap.py
echo "Running compute_partial_texturemap.py..."
conda run -n smplitex python compute_partial_texturemap.py --input_folder ./dummy_data || { echo "Error running compute_partial_texturemap.py. Exiting."; exit 1; }

# Run inpaint_with_A1111.py
echo "Running inpaint_with_A1111.py..."
conda run -n smplitex python inpaint_with_A1111.py --partial_textures ./dummy_data/uv-textures --masks ./dummy_data/uv-textures-masks --inpainted_textures ./dummy_data/uv-textures-inpainted || { echo "Error running inpaint_with_A1111.py. Exiting."; exit 1; }

echo "Pipeline completed successfully!"