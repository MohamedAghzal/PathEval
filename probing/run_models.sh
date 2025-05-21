#!/bin/bash

models=(
    "google/vit-base-patch16-224"
    "google/siglip-base-patch16-224"
    "openai/clip-vit-large-patch14-336"
    "openai/clip-vit-base-patch32"
    "facebook/dino-vits16"
    "facebook/dinov2-base"
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
)

# Loop through each model and submit a job
for model in "${models[@]}"; do
    # Check if the model name contains "dino" or "siglip"
    if [[ "$model" == *"dinov2"* ]]; then
        script="dino-v2.py"  # Use dino.py if the model name contains "dino"
    elif [[ "$model" == *"dino"* ]]; then
        script="dino.py"  # Use dino.py if the model name contains "dino"
    elif [[ "$model" == *"siglip"* ]]; then
        script="siglip.py"  # Use siglip.py if the model name contains "siglip"
    else
        script="run_probe.py"  # Default script for other models
    fi

    # Submit the job
    sbatch --export=MODEL_NAME="$model" <<EOF
#!/bin/bash
#SBATCH --job-name=probe_${model//\//_}_FALSE   # Job name (slashes replaced with underscores)
#SBATCH --output=logs/${model//\//_}_FALSE.out  # Output log file
#SBATCH --error=logs/${model//\//_}_FALSE.err   # Error log file
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:1          
#SBATCH   --mem-per-cpu=32GB             # Request nGB RAM per core
source ../Open-LLaVA-NeXT/llava-next/bin/activate
module load python gnu10

# Run the appropriate script

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python $script --model_name "\$MODEL_NAME" --unfreeze_encoder False
EOF

    sbatch --export=MODEL_NAME="$model" <<EOF
#!/bin/bash
#SBATCH --job-name=probe_${model//\//_}_TRUE   # Job name (slashes replaced with underscores)
#SBATCH --output=logs/${model//\//_}_TRUE.out  # Output log file
#SBATCH --error=logs/${model//\//_}_TRUE.err   # Error log file
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:1          
#SBATCH   --mem-per-cpu=32GB             # Request nGB RAM per core
source ../Open-LLaVA-NeXT/llava-next/bin/activate
module load python gnu10

# Run the appropriate script
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python $script --model_name "\$MODEL_NAME" --unfreeze_encoder True
EOF
done