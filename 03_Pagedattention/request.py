from typing import Any, List, Optional
from dataclasses import dataclass, field

import torch

@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class Request:
    request_id: int = -1
    prefill_token_ids: torch.Tensor = None  # prefill阶段的token_ids，等同于prompt的token_ids
    gen_token_ids: torch.Tensor = None      # 已经生成的tokens
    
    
    block_table: List[int] = field(default_factory=list)
    block_size: int = 0
    seq_len: int = 0
    num_prompt_tokens: int = 0
    num_gen_tokens: int = 0

    #-----------------------
    # 后续用paged_kv_cache替换
    past_key_values: Any = None             # KVCache
    #-----------------------

    max_gen_tokens: Any = None              # 本request允许生成最大的token数
    eos_token_id: Any = None
    next_token: Any = None                  # 推理生成的下一个token
    finished: bool = False                  # 本requset是否已经完成推理
    gen_text: str = ""
    # 采样参数
    sampling_params: 'SamplingParams' = field(default_factory=lambda: SamplingParams())
    


    @property
    def num_total_tokens(self)->int:
        return self.seq_len
    
    @property
    def last_block_filled(self)->int:
        if self.block_size <= 0:
            raise ValueError("block_size must be set before accessing last_block_filled")
        return self.seq_len % self.block_size
