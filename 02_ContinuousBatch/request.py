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
    prefill_token_ids: torch.Tensor = None
    gen_token_ids: torch.Tensor = None
    past_key_values: Any = None
    max_gen_tokens: Any = None
    eos_token_id: Any = None
    next_token: Any = None
    finished: bool = False
    gen_text: str = ""
    # sampling_params: SamplingParams = SamplingParams()
    sampling_params: 'SamplingParams' = field(default_factory=lambda: SamplingParams())
    



