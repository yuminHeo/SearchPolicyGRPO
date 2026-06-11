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

import os
import logging
import torch
import numpy as np
from packaging import version
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.protocol import all_gather_data_proto
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version
from vllm.version import __version__ as VLLM_VERSION

from .base import BaseShardingManager
from .patch import patched_ds_v3_load_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _to_full_tensor(tensor):
    if hasattr(tensor, "full_tensor"):
        return tensor.full_tensor()
    return tensor


def _normalize_peft_state_key(name: str) -> str:
    name = name.removeprefix("_fsdp_wrapped_module.")
    name = name.removeprefix("module.")
    name = name.removeprefix("base_model.model.")
    name = name.replace(".base_layer.", ".")
    return name


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def _get_lora_scalings(self):
        wrapped_module = getattr(self.module, "_fsdp_wrapped_module", self.module)
        peft_config = getattr(wrapped_module, "peft_config", None)
        if not peft_config:
            return {}

        scalings = {}
        for adapter_name, adapter_config in peft_config.items():
            rank = adapter_config.r
            alpha = adapter_config.lora_alpha
            if getattr(adapter_config, "use_rslora", False):
                scalings[adapter_name] = alpha / np.sqrt(rank)
            else:
                scalings[adapter_name] = alpha / rank
        return scalings

    def _prepare_peft_lora_state_dict_for_vllm(self, params):
        has_lora = any(".lora_A." in name or ".lora_B." in name for name in params.keys())
        has_peft_prefix = any(name.startswith("base_model.model.") or ".base_layer." in name for name in params.keys())
        if not has_lora and not has_peft_prefix:
            return params

        scalings = self._get_lora_scalings()
        if not scalings:
            scalings = {"default": 1.0}

        normalized_params = {}
        lora_groups = {}

        for name, tensor in params.items():
            if ".lora_A." in name or ".lora_B." in name:
                marker = ".lora_A." if ".lora_A." in name else ".lora_B."
                side = "A" if marker == ".lora_A." else "B"
                module_name, suffix = name.split(marker, 1)
                adapter_name = suffix.rsplit(".weight", 1)[0]
                base_key = f"{_normalize_peft_state_key(module_name)}.weight"
                group = lora_groups.setdefault(base_key, {})
                group[f"{side}:{adapter_name}"] = tensor
                continue

            normalized_name = _normalize_peft_state_key(name)
            if "lora_" in normalized_name:
                continue
            normalized_params[normalized_name] = tensor

        merged_count = 0
        for base_key, group in lora_groups.items():
            if base_key not in normalized_params:
                continue
            adapter_names = {
                key.split(":", 1)[1]
                for key in group.keys()
                if key.startswith("A:")
            }
            merged_weight = _to_full_tensor(normalized_params[base_key])
            for adapter_name in adapter_names:
                a = group.get(f"A:{adapter_name}")
                b = group.get(f"B:{adapter_name}")
                if a is None or b is None:
                    continue
                a = _to_full_tensor(a).to(device=merged_weight.device, dtype=merged_weight.dtype)
                b = _to_full_tensor(b).to(device=merged_weight.device, dtype=merged_weight.dtype)
                scaling = scalings.get(adapter_name, scalings.get("default", 1.0))
                merged_weight = merged_weight + (b @ a) * scaling
                merged_count += 1

            normalized_params[base_key] = merged_weight.detach().cpu()

        if merged_count > 0:
            print(
                f"[FSDPVLLMShardingManager] rank={torch.distributed.get_rank()} "
                f"merged {merged_count} LoRA matrices into vLLM weights",
                flush=True,
            )
        return normalized_params

    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()

        print(f"[FSDPVLLMShardingManager] rank={torch.distributed.get_rank()} entering state_dict()", flush=True)
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        params = self._prepare_peft_lora_state_dict_for_vllm(params)
        print(
            f"[FSDPVLLMShardingManager] rank={torch.distributed.get_rank()} state_dict done "
            f"num_params={len(params)} load_format={load_format}",
            flush=True,
        )

        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            print(f"[FSDPVLLMShardingManager] rank={torch.distributed.get_rank()} sync_model_weights start", flush=True)
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
            print(f"[FSDPVLLMShardingManager] rank={torch.distributed.get_rank()} sync_model_weights done", flush=True)
            del params
        else:
            if version.parse(VLLM_VERSION) >= version.parse("0.8.3"):
                # wake up only weights
                self.inference_engine.wake_up(tags=["weights"])
                # update model params
                self.update_params(params)

                log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
                del params
                torch.cuda.empty_cache()

                # wake up kv
                self.inference_engine.wake_up(tags=["kv_cache"])
            else:
                self.inference_engine.wake_up()
                self.update_params(params)
                log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
                del params

        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params):
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        world_size = torch.distributed.get_world_size()
        if model.config.architectures[0] in ['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM']:
            loaded_params = patched_ds_v3_load_weights(
                model, ((name, param.full_tensor() if world_size != 1 and hasattr(param, 'full_tensor') else param)
                        for name, param in updated_params.items()))
        else:
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if world_size != 1 else param) for name, param in updated_params.items()))
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")
