NUM_PROCESSES=4 DEEPSPEED_CONFIG=configs/deepspeed_zero2_stage2_grpo.json \
scripts/train_stage2_search_grpo.sh \
  --group_size 4 \
  --max_turns 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --auto_adjust_search_cost \
  --lora_adapter_path /data/YM/sft-selected/qwen25_7b_turn_sft_lora_ddp-selected \
  --ref_lora_adapter_path /data/YM/sft-selected/qwen25_7b_turn_sft_lora_ddp-selected
