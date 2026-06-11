# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams


FINAL_ANSWER_PREFIX = (
    "\n<think>Maximum search turn budget reached. Given the retrieved context information above "
    "and without prior knowledge, evaluate whether the documents support the triple. Do not search "
    "again. Output exactly \\boxed{true} if the triple is correct according to the documents. "
    "Output exactly \\boxed{false} if the triple is incorrect according to the documents, with no "
    "explanation.</think>\n<answer>"
)

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get('max_num_batched_tokens', 8192))

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        # This rollout uses string stop sequences for tool turns. vLLM 0.6.3
        # requires detokenization for string stops, while the vendored wrapper
        # still returns token_ids for training.
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = True

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            # log_probs = output[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)


class vLLMRolloutWithTool(vLLMRollout):
    """vLLM rollout with iterative <search>/<result> tool execution for customized vLLM."""

    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r"<search>", output_str)
        end_tags = re.findall(r"</search>", output_str)
        if len(start_tags) != len(end_tags):
            return False
        start_positions = [m.start() for m in re.finditer(r"<search>", output_str)]
        end_positions = [m.start() for m in re.finditer(r"</search>", output_str)]
        return all(start < end for start, end in zip(start_positions, end_positions))

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []
        matches = re.finditer(r"<search>((?:(?!</search>).)*)</search>", output_str, re.DOTALL)
        return [match.group(1).strip() for match in matches if match.group(1).strip()][-1:]

    def batch_execute(self, env_list: List[str], tool_calls_list: List[List[str]]):
        del env_list

        def format_search_payload(payload):
            if isinstance(payload, dict):
                if isinstance(payload.get("documents"), list):
                    docs = payload["documents"]
                elif isinstance(payload.get("results"), list):
                    docs = payload["results"]
                else:
                    return f"search_error unexpected_response={payload}"
            elif isinstance(payload, list):
                docs = payload[0] if len(payload) == 2 and isinstance(payload[0], list) else payload
            else:
                return f"search_error unexpected_response={payload}"
            chunks = []
            max_chars = int(self.config.get("max_result_chars", 6000))
            for doc in docs:
                if not isinstance(doc, dict):
                    chunks.append(str(doc))
                    continue
                doc_id = doc.get("id", "")
                title = doc.get("title", "")
                contents = doc.get("contents", "")
                chunks.append(f"[{doc_id}] {title}\n{contents}".strip())
            text = "\n\n".join(chunk for chunk in chunks if chunk)
            return text[:max_chars]

        def exe_tool_call(call):
            search_url = str(self.config.get("search_url", "") or "").rstrip("/")
            top_n = int(self.config.get("top_n", 5))
            if not search_url:
                return "search_error: actor_rollout_ref.rollout.search_url is not set"
            try:
                response = requests.post(f"{search_url}/search", json={"query": call, "top_n": top_n}, timeout=20)
                if response.status_code == 422:
                    response = requests.post(f"{search_url}/search", params={"query": call, "top_n": top_n}, timeout=20)
                if response.status_code >= 400:
                    return f"search_error status={response.status_code} body={response.text[:500]}"
                return format_search_payload(response.json())
            except requests.exceptions.Timeout:
                return "search_error: retrieval timed out"
            except Exception as exc:
                return f"search_error: {exc}"

        all_tasks = []
        task_indices = []
        for env_idx, tool_calls in enumerate(tool_calls_list):
            for call_idx, tool_call in enumerate(tool_calls):
                all_tasks.append(tool_call)
                task_indices.append((env_idx, call_idx))

        all_results = [None] * len(all_tasks)
        max_workers = int(self.config.get("search_max_workers", 32))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(exe_tool_call, call): i for i, call in enumerate(all_tasks)
            }
            for future in as_completed(future_to_index):
                all_results[future_to_index[future]] = future.result()

        results_list = [[None for _ in tool_calls] for tool_calls in tool_calls_list]
        for (env_idx, call_idx), result in zip(task_indices, all_results):
            results_list[env_idx][call_idx] = result
        return results_list

    def _trim_generated_ids(self, output_ids):
        ids = list(output_ids)
        while ids and ids[-1] == self.pad_token_id:
            ids.pop()
        return ids

    def _summarize_result(self, result_text: str) -> str:
        body = re.sub(r"^<result>|</result>\s*$", "", result_text.strip(), flags=re.DOTALL).strip()
        mode = str(self.config.get("result_summary_mode", "truncate"))
        if mode == "none":
            return result_text
        max_chars = int(self.config.get("result_summary_chars", 480))
        body = re.sub(r"\s+", " ", body)
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + " ..."
        return f"<result_summary>{body}</result_summary>\n"

    def _summarize_previous_results(self, parts):
        for pos, (text, mask_value) in enumerate(parts):
            if mask_value == 0 and text.lstrip().startswith("<result>"):
                parts[pos] = (self._summarize_result(text), 0)

    def _parts_to_ids_and_mask(self, init_ids, parts):
        response_ids = []
        response_mask = []
        for text, mask_value in parts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            response_ids.extend(ids)
            response_mask.extend([mask_value] * len(ids))
        return init_ids + response_ids, response_mask

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = ori_input_ids.size(0)
        eos_token_id = prompts.meta_info["eos_token_id"]
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        sample_kwargs = {}
        if not do_sample:
            sample_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            sample_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }

        with self.update_sampling_params(**sample_kwargs):
            rollout_n = self.sampling_params.n if do_sample and not is_validate else 1
            prompt_ids = [_pre_process_inputs(self.pad_token_id, ori_input_ids[i]) for i in range(batch_size)]
            init_inputs = [ids.copy() for ids in prompt_ids for _ in range(rollout_n)]
            curr_inputs = [ids.copy() for ids in prompt_ids for _ in range(rollout_n)]
            result_mask_list = [[] for _ in curr_inputs]
            response_parts = [[] for _ in curr_inputs]

            if "env" in prompts.non_tensor_batch:
                source_envs = list(prompts.non_tensor_batch["env"])
                env_list = [source_envs[i] for i in range(batch_size) for _ in range(rollout_n)]
            else:
                env_list = [""] * len(curr_inputs)

            curr_max_tokens = [self.config.response_length] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            for turn_idx in range(self.config.max_turns):
                if not active_indices:
                    break

                active_inputs = [curr_inputs[i] for i in active_indices]
                max_tokens = min(4096, max(curr_max_tokens[i] for i in active_indices))
                print(
                    f"[vLLMRolloutWithTool] turn active={len(active_inputs)} max_tokens={max_tokens} "
                    f"prompt_tokens_max={max(len(x) for x in active_inputs) if active_inputs else 0}",
                    flush=True,
                )
                with self.update_sampling_params(
                    n=1,
                    max_tokens=max_tokens,
                    stop=["</search>", "</answer>"],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                ):
                    generate_start = time.perf_counter()
                    output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False,
                    )
                    generate_elapsed = time.perf_counter() - generate_start
                print(
                    f"[vLLMRolloutWithTool] turn={turn_idx} generate_sec={generate_elapsed:.2f} "
                    f"active={len(active_inputs)}",
                    flush=True,
                )

                output_tensor = output[0].to(ori_input_ids.device)
                tool_calls_list: List[List[str]] = []
                call_indices: List[int] = []
                new_active_indices = []

                for row_idx, idx in enumerate(active_indices):
                    output_ids = self._trim_generated_ids(output_tensor[row_idx].tolist())
                    output_str = self.tokenizer.decode(output_ids)
                    if "<search>" in output_str and "</search>" not in output_str and "<answer>" not in output_str:
                        close_ids = self.tokenizer.encode("</search>", add_special_tokens=False)
                        output_ids += close_ids
                        output_str += "</search>"

                    response_parts[idx].append((output_str, 1))
                    curr_inputs[idx], result_mask_list[idx] = self._parts_to_ids_and_mask(init_inputs[idx], response_parts[idx])
                    tool_calls = self.extract_tool_calls(output_str)
                    if tool_calls:
                        tool_calls_list.append(tool_calls)
                        call_indices.append(idx)
                        new_active_indices.append(idx)

                if tool_calls_list:
                    active_env_list = [env_list[i] for i in call_indices]
                    tool_call_count = sum(len(calls) for calls in tool_calls_list)
                    tool_start = time.perf_counter()
                    tool_responses_list = self.batch_execute(active_env_list, tool_calls_list)
                    tool_elapsed = time.perf_counter() - tool_start
                    print(
                        f"[vLLMRolloutWithTool] turn={turn_idx} tool_sec={tool_elapsed:.2f} "
                        f"envs={len(active_env_list)} calls={tool_call_count}",
                        flush=True,
                    )
                    for idx, tool_calls, tool_responses in zip(call_indices, tool_calls_list, tool_responses_list):
                        tool_response_str = ""
                        self._summarize_previous_results(response_parts[idx])
                        for _, response in zip(tool_calls, tool_responses):
                            tool_response_str += f"<result>{response}\n</result>\n"
                        response_parts[idx].append((tool_response_str, 0))
                        curr_inputs[idx], result_mask_list[idx] = self._parts_to_ids_and_mask(init_inputs[idx], response_parts[idx])

                length_checked_active_indices = []
                for idx in active_indices:
                    generated_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if generated_len >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] + curr_inputs[idx][
                            len(init_inputs[idx]) : len(init_inputs[idx]) + self.config.response_length
                        ]
                        result_mask_list[idx] = result_mask_list[idx][: self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - generated_len
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices

            forced_indices = [idx for idx in active_indices if "<answer>" not in "".join(text for text, _ in response_parts[idx])]
            if forced_indices:
                for idx in forced_indices:
                    response_parts[idx].append((FINAL_ANSWER_PREFIX, 0))
                    curr_inputs[idx], result_mask_list[idx] = self._parts_to_ids_and_mask(init_inputs[idx], response_parts[idx])
                active_inputs = [curr_inputs[i] for i in forced_indices]
                remaining = [self.config.response_length - (len(curr_inputs[i]) - len(init_inputs[i])) for i in forced_indices]
                max_tokens = max(1, min(64, max(remaining)))
                with self.update_sampling_params(n=1, max_tokens=max_tokens, stop=["</answer>"], temperature=0.0, top_p=1.0, top_k=-1):
                    output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False,
                    )
                output_tensor = output[0].to(ori_input_ids.device)
                for row_idx, idx in enumerate(forced_indices):
                    output_ids = self._trim_generated_ids(output_tensor[row_idx].tolist())
                    output_str = self.tokenizer.decode(output_ids)
                    if "</answer>" not in output_str:
                        output_str += "</answer>"
                    response_parts[idx].append((output_str, 1))
                    curr_inputs[idx], result_mask_list[idx] = self._parts_to_ids_and_mask(init_inputs[idx], response_parts[idx])

            output_ids_list = [
                curr_inputs[i][len(init_inputs[i]) : len(init_inputs[i]) + self.config.response_length]
                for i in range(len(curr_inputs))
            ]

        repeated_prompts = ori_input_ids.repeat_interleave(rollout_n, dim=0)
        repeated_attention_mask = attention_mask.repeat_interleave(rollout_n, dim=0)
        repeated_position_ids = position_ids.repeat_interleave(rollout_n, dim=0)
        out_batch_size = repeated_prompts.size(0)

        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask_tensor = torch.tensor(result_mask[: len(output_ids)], device=ori_input_ids.device)
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask_tensor)

        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)
        seq = torch.cat([repeated_prompts, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(out_batch_size, 1)
        response_position_ids = repeated_position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([repeated_position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((repeated_attention_mask, response_attention_mask), dim=-1)
        loss_mask = result_mask * response_attention_mask

        batch = TensorDict(
            {
                "prompts": repeated_prompts,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            },
            batch_size=out_batch_size,
        )

        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
