#!/bin/bash

# Run image_to_densepose.py
echo "Running image_to_densepose.py..."
conda run -n smplitex python image_to_densepose.py --detectron2 ./detectron2 --input_folder ./dummy_data/images-stableviton || { echo "Error running image_to_densepose.py. Exiting."; exit 1; }

# Run SemanticGuidedHumanMatting/test_image.py
echo "Running SemanticGuidedHumanMatting/test_image.py..."
conda run -n smplitex python SemanticGuidedHumanMatting/test_image.py --images-dir ./dummy_data/images-stableviton --result-dir ./dummy_data/images-seg --pretrained-weight SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth || { echo "Error running SemanticGuidedHumanMatting/test_image.py. Exiting."; exit 1; }

# Run compute_partial_texturemap.py
echo "Running compute_partial_texturemap.py..."
conda run -n smplitex python compute_partial_texturemap.py --input_folder ./dummy_data || { echo "Error running compute_partial_texturemap.py. Exiting."; exit 1; }

# Run inpaint_with_A1111.py
echo "Running inpaint_with_A1111.py..."
conda run -n smplitex python inpaint_with_A1111.py --partial_textures ./dummy_data/uv-textures --masks ./dummy_data/uv-textures-masks --inpainted_textures ./dummy_data/uv-textures-inpainted || { echo "Error running inpaint_with_A1111.py. Exiting."; exit 1; }

# Run render_results.py
echo "Running render_results.py..."
conda run -n pytorch3d python render_results.py --textures ./dummy_data/uv-textures-inpainted/ || { echo "Error running render_results.py. Exiting."; exit 1; }

echo "Pipeline completed successfully!"