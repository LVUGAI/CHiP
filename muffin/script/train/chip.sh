export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=$PYTHONPATH:`realpath .`
echo "pythonpath="$PYTHONPATH
sleep 1

export CUDA_VISIBLE_DEVICES="0,1,2,3"
MASTER_ADDR=`hostname`
MASTER_PORT=13251
RDZV_ENDPOINT=$MASTER_ADDR:$MASTER_PORT
RUNNER="torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${RDZV_ENDPOINT}"
echo RUNNER=$RUNNER

$RUNNER ./muffin/train/train_mem_muffin.py \
    --model_name_or_path  Yirany/RLHF-V-SFT \
    --output_dir {chip_ckpt_name} \
    --image_folder not_used \
    --vision_tower beit3_large_patch16_448 \
    --pretrain_mm_mlp_adapter not_used \
    --fully_tune True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --fp16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 35 \
    --save_total_limit 10 \
    --data_source_names  '' \
    --data_source_weights '' \
    --data_dir ./playground/data \
    --ref_name RLHFV \
    --max_steps -1 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --logging_dir ./log/ \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --fsdp "full_shard auto_wrap" \
    --task DPO \
    --report_to wandb \
    --dataloader_num_workers 10 \
    --dpo_use_average False \
    --dpo_token_weighted True \
    --dpo_token_weight 2.0 \
    --use_cross_modal_loss True \
    --use_tok True \
    --tok_beta 0.1 \
    --tok_logits_path playground/data/chip_logits \
    --parquet_path playground/data/chip_logits.parquet \
    --use_image_type diffusion
