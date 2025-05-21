#!/bin/bash
#!/bin/bash
#SBATCH --job-name=llava_3D
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=gpu_job-3D--2?.out
#SBATCH --error=gpu_job-3D--2?.err
#SBATCH   --mem-per-cpu=64GB             # Request nGB RAM per core

source llava-next/bin/activate
# Set cache directories
export HF_HOME="./new_cache_dir"
export TRANSFORMERS_CACHE="./new_cache_dir"
export TORCH_HOME="./"
export CUDA_CACHE_PATH="./"

pip install -e ".[train]"
pip install flash-attn --no-build-isolation

DEEPSPEED_SCRIPT="deepspeed llava/train/train_mem.py"
DEEPSPEED_JSON="./scripts/zero3.json"
MODEL_NAME="liuhaotian/llava-v1.6-vicuna-7b"
DATA_PATH="../path-generation/train-set-added/dataset/training_set_3D.json"
DEV_DATA_PATH="../finetuning/llava_dataset_3D_dev.json"
IMAGE_FOLDER="./"
VISION_TOWER="openai/clip-vit-large-patch14-336"
OUTPUT_DIR="./output_weights/llava_7b_3D"
#export CUDA_VISIBLE_DEVICES=0

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
   --deepspeed $DEEPSPEED_JSON \
    --model_name_or_path $MODEL_NAME \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 50 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-6 \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad