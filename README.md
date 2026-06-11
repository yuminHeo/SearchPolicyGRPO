# SearchPolicyGRPO

This is a standalone Stage 2 trainer for trajectory-level GRPO over the decision to continue searching or stop and answer. It does not import `trajrl.*`; prompt formatting, parsing, reward computation, rollout, masking, and metric logging are self-contained.

Core behavior:

- Samples `G` trajectories per triple.
- Uses the same iterative context shape as inference: previous search results are compacted into `<result_summary>`, and only the latest retrieval is kept as full `<result>`.
- Computes loss only on model-generated `<think>`, `<search>`, and `<answer>` tokens. Retrieval-provided `<result>` and `<result_summary>` text is only conditioning context and is masked out by construction.
- Applies the requested trajectory reward, group-relative advantage, KL penalty against the Stage 1 SFT reference model, reward clipping, identical-reward group skipping, gradient accumulation, and mixed precision.
- Trains with LoRA by default. Use `--no_lora` only when full fine-tuning is intended.
- Logs accuracy, UNKNOWN rate, false UNKNOWN rate, true to false rate, false to true rate, average search count, forced-final invalid count, false recall, and true recall.

Example:

```bash
scripts/train_stage2_search_grpo.sh \
  --group_size 4 \
  --max_turns 4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_r 16 \
  --lora_alpha 32 \
  --auto_adjust_search_cost
```

To continue from a Stage 1 LoRA adapter and use the same adapter as the frozen KL reference:

```bash
scripts/train_stage2_search_grpo.sh \
  --lora_adapter_path outputs/stage1_sft_lora \
  --ref_lora_adapter_path outputs/stage1_sft_lora
```

For multi-GPU:

```bash
NUM_PROCESSES=4 MIXED_PRECISION=bf16 scripts/train_stage2_search_grpo.sh \
  --group_size 4 \
  --gradient_accumulation_steps 4
```

For DeepSpeed, pass an Accelerate-compatible config:

```bash
NUM_PROCESSES=4 DEEPSPEED_CONFIG=configs/deepspeed_zero2_stage2_grpo.json scripts/train_stage2_search_grpo.sh
```

Defaults assume this directory lives next to TrajRL:

- train data: `../TrajRL/dataset/trex_renlg/train.jsonl`
- eval data: `../TrajRL/dataset/trex_renlg/dev.jsonl`
- retriever: `http://localhost:8090`

Override them with `TRAIN_FILE=...`, `EVAL_FILE=...`, and `SEARCH_URL=...`.
