export WANDB_PROJECT=regen_debug
export WANDB_MODE=disabled
export WANDB_RUN_NAME=qwen2_5-1.5b-regen-debug
export CUDA_VISIBLE_DEVICES=0

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    regen_train.py configs/config_regen.yaml \
    --model_name_or_path="Qwen2.5-1.5B-Instruct" \
    --data_path="data/train/Trajectories" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --num_train_epochs=1 \
    --max_steps=5 \
    --logging_steps=1 \
    --save_strategy='no' \
    --output_dir=outputs/debug \
    --regen_alpha=1 \
    --regen_clip_ratio_max=10.0 \
    --regen_length_normalize=false \
    --prompt=ours \
    --gradient_checkpointing=false
