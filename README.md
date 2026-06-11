# SearchPolicyGRPO

This is a standalone Stage 2 trainer for trajectory-level GRPO over the decision to continue searching or stop and answer. It does not import `trajrl.*`; prompt formatting, parsing, reward computation, rollout, masking, and metric logging are self-contained.

Core behavior:

- Samples `G` trajectories per triple.
- Uses the same iterative context shape as inference: previous search results are compacted into `<result_summary>`, and only the latest retrieval is kept as full `<result>`.
- Computes loss only on model-generated `<think>`, `<search>`, and `<answer>` tokens. Retrieval-provided `<result>` and `<result_summary>` text is only conditioning context and is masked out by construction.
- Applies the requested trajectory reward, group-relative advantage, KL penalty against the Stage 1 SFT reference model, identical-reward group skipping, gradient accumulation, and mixed precision.
- Trains with LoRA by default. Use `--no_lora` only when full fine-tuning is intended.
- Logs accuracy, UNKNOWN rate, false UNKNOWN rate, true to false rate, false to true rate, average search count, forced-final invalid count, false recall, and true recall to JSONL and Weights & Biases.
- Shows train/eval progress with `tqdm`, including ETA and current loss/reward/accuracy/search stats.

## Reward

Reward is computed once at the end of each trajectory:

```text
R = R_correct + R_evidence + R_search
```

- `R_correct`: `+1.0` if the final prediction matches the label, `0.0` if it is wrong, `-0.5` if it is `unknown`.
- `R_evidence`: `+0.5` if the answer is correct and any retrieved document matches `gold_evidence`; otherwise `0.0`. This is applied for both positive and negative examples because the original positive gold document can also serve as contradiction evidence for negative variants.
- `R_search`: search cost after the first search only, `max(0, N_search - 1) * -0.05`.

The prediction is considered `true` or `false` only when it appears as boxed text inside `<answer>...</answer>`, for example `<answer>\boxed{true}</answer>`. Missing answers, unboxed answers, or other labels are treated as `unknown`.

Example:

```bash
scripts/train_stage2_search_grpo.sh \
  --group_size 4 \
  --max_turns 4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_r 16 \
  --lora_alpha 32
```

To continue from a Stage 1 LoRA adapter and use the same adapter as the frozen KL reference:

```bash
scripts/train_stage2_search_grpo.sh \
  --lora_adapter_path outputs/stage1_sft_lora \
  --ref_lora_adapter_path outputs/stage1_sft_lora
```

## Logging

Weights & Biases is enabled by default on the main process:

```bash
scripts/train_stage2_search_grpo.sh \
  --wandb_project SearchPolicyGRPO \
  --wandb_run_name grpo-bge-lora
```

Useful options:

```bash
--wandb_entity ENTITY
--wandb_mode offline
--disable_wandb
```

The terminal also shows `tqdm` progress bars for training and evaluation. The training bar is step-based, so its ETA tracks optimizer steps.

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
- BGE corpus: `/data/YM/ExpCodes/TrajRL/dataset/trex_renlg/corpus.jsonl`
- BGE index dir: `/data/YM/ExpCodes/TrajRL/outputs/indexes`

Override them with `TRAIN_FILE=...`, `EVAL_FILE=...`, and `SEARCH_URL=...`.

## BGE Retriever

`run.sh` starts a standalone SearchPolicyGRPO BGE retriever by default using the existing index files:

```bash
/data/YM/ExpCodes/TrajRL/outputs/indexes/trex_renlg_bge.npy
/data/YM/ExpCodes/TrajRL/outputs/indexes/trex_renlg_bge_ivf4096.faiss
```

Common overrides:

```bash
TRAJRL_DIR=/data/YM/ExpCodes/TrajRL \
INDEX_DIR=/data/YM/ExpCodes/TrajRL/outputs/indexes \
CORPUS=/data/YM/ExpCodes/TrajRL/dataset/trex_renlg/corpus.jsonl \
EMBEDDING_CACHE=/path/to/trex_renlg_bge.npy \
INDEX_CACHE=/path/to/trex_renlg_bge_ivf4096.faiss \
RETRIEVER_PORT=8090 \
AUTO_START_RETRIEVER=1 \
./run.sh
```

If a retriever is already running:

```bash
AUTO_START_RETRIEVER=0 SEARCH_URL=http://localhost:8090 ./run.sh
```

The retriever implementation is local to this project:

```bash
bge_retriever_server.py
scripts/launch_bge_retriever.sh
```

It does not import or execute TrajRL code. The default corpus/index paths point at the already-built files under `/data/YM/ExpCodes/TrajRL`, but those are ordinary data files and can be overridden with environment variables.
