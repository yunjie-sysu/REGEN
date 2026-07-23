export WANDB_PROJECT=regen
export WANDB_MODE=offline

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    --mixed_precision=bf16 \
    --num_processes 4 \
    regen_train.py configs/config_regen.yaml \
    --model_name_or_path=/data/vjuicefs_ai_gpt_wl/public_data/11195618/yunjie/base_model/Qwen2.5-1.5B-Instruct\
    --data_path="data/train/Teacher-Online-All-V5" \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=5e-7 \
    --max_length=1024 \
    --num_train_epochs=5 \
    --save_strategy='steps' \
    --save_steps=600 \
    --output_dir=outputs/qwen2_5-1.5b-instruct-regen \
    --regen_alpha=1 \
    --regen_clip_ratio_max=10.0 \
    --regen_length_normalize=false \
    --prompt="ours"
