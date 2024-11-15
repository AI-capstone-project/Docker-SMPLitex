# Get arguments from the command line first id, second pose_id
id=$1
pose_id=$2

# Run render_results.py
echo "Running render_results.py..."
conda run -n pytorch3d python render_results.py --textures ./dummy_data/uv-textures-inpainted/ --resultDir ./dummy_data/3d_outputs/ --pose_id $pose_id || { echo "Error running render_results.py. Exiting."; exit 1; }

echo "Pipeline completed successfully!"
