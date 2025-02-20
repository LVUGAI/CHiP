#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
echo "pythonpath="$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0,1,2,3"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path  liuhaotian/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_path ./playground/data/chip.parquet \
    --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./ckpt/llava-v1.6-vicuna-7b-DataRlhfv5733-cmdpo \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 35 \
    --save_total_limit 10 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --max_grad_norm 20.0 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --task DPO \
    --use_image_type diffusion \
    --diffusion_step 500 \
    --dpo_token_weighted False \
    --dpo_token_weight 4.0 \
    --use_cross_modal_loss True \
    --use_tdpo False \
    --tok_beta 0.1
