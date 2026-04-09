import time
import torch
import asyncio

from typing import List, AsyncGenerator

from model import Model
from tokenizer import Tokenizer
from scheduler import Scheduler
from request import Request, SamplingParams
from paged_kv_cache import PagedKVCache
from transformers.cache_utils import DynamicCache

from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence
from performance_stats import EngineStats, RequestStats, calculate_kv_cache_size_mb


@dataclass
class GenerationState:
    ''' 生成统计信息，用于跟踪生成过程 这是学习的关键！'''
    return_generation_state: bool = False # 是否返回生成状态统计信息
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    prefill_memory: float = 0.0
    decode_memory: float = 0.0
    total_memory: float = 0.0
    num_tokens_generated: int = 0
    tokens_per_second: float = 0.0    
    prefill_kv_cache_size_mb: float = 0.0 # prefill阶段的KV Cache大小
    kv_cache_size_mb: float = 0.0 # 总KV Cache的大小，单位MB    
    prompt_tokens: int = 0 # prompt中的token数
    time_to_first_token: float = 0.0 # 生成第一个token的时间
    eos_info: bool = False # 是否由于生成了eos而停止


class MiniVLLM:
    def __init__(
        self,
        model_id: str,
        bnb_config=None,
        device_map="cuda",
        enable_stats: bool = True,
        block_size: int = 16,
        total_kv_blocks: int = 4096,
    ):
        self.model = Model(model_id, bnb_config, device_map=device_map)
        self.tokenizer = Tokenizer(model_id)
        self.scheduler = Scheduler()
        self.engine_running = False
        self.engine_task = None

        self.engine_stats = EngineStats()
        self.enable_stats = enable_stats
        self.block_size = block_size
        self.total_kv_blocks = total_kv_blocks
        self.kv_cache = None

    def _get_cache_num_layers(self, cache) -> int:
        """兼容新版 DynamicCache 和旧版 tuple cache 的层数读取。"""
        if hasattr(cache, "layers"):
            return len(cache.layers)
        return len(cache)

    def _get_layer_kv(self, cache, layer_idx: int):
        """兼容新版 DynamicCache 和旧版 tuple cache 的单层 KV 读取。"""
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            return layer.keys, layer.values
        return cache[layer_idx]

    def _ensure_paged_kv_cache(self, cache) -> None:
        if self.kv_cache is not None:
            return

        num_layers = self._get_cache_num_layers(cache)
        sample_k, _ = self._get_layer_kv(cache, 0)
        self.kv_cache = PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=sample_k.size(1),
            head_dim=sample_k.size(3),
            block_size=self.block_size,
            total_blocks=self.total_kv_blocks,
            dtype=sample_k.dtype,
            device=sample_k.device,
        )

    def _store_prefill_in_paged_cache(self, req: Request, cache, batch_idx: int, seq_len: int) -> None:
        self._ensure_paged_kv_cache(cache)

        num_layers = self._get_cache_num_layers(cache)
        needed_blocks = self.kv_cache.get_num_blocks_needed(seq_len)
        block_ids = self.kv_cache.allocate_blocks(needed_blocks)

        for layer_idx in range(num_layers):
            k, v = self._get_layer_kv(cache, layer_idx)
            k_req = k[batch_idx : batch_idx + 1, :, :seq_len, :].contiguous()
            v_req = v[batch_idx : batch_idx + 1, :, :seq_len, :].contiguous()
            self.kv_cache.write_prefill(layer_idx, block_ids, k_req, v_req, seq_len)

        req.block_table = block_ids
        req.block_size = self.block_size
        req.seq_len = seq_len
        req.num_prompt_tokens = seq_len

    def _build_dynamic_cache_from_paged(self, req: Request) -> DynamicCache:
        if self.kv_cache is None:
            raise RuntimeError("Paged KV cache is not initialized")

        rebuilt_cache = DynamicCache()
        for layer_idx in range(self.kv_cache.num_layers):
            k_hist, v_hist = self.kv_cache.gather_sequence(
                layer_idx=layer_idx,
                block_table=req.block_table,
                seq_len=req.seq_len,
            )
            rebuilt_cache.update(k_hist.contiguous(), v_hist.contiguous(), layer_idx)
        return rebuilt_cache

    def _ensure_decode_block(self, req: Request) -> tuple[int, int]:
        if self.kv_cache is None:
            raise RuntimeError("Paged KV cache is not initialized")

        if not req.block_table or req.seq_len % self.block_size == 0:
            req.block_table.extend(self.kv_cache.allocate_blocks(1))

        block_id = req.block_table[-1]
        offset = req.seq_len % self.block_size
        return block_id, offset

    async def generate(
            self, 
            prompt: str, 
            max_new_tokens: int = 100, 
            temperature: float = 1.0, 
            top_p: float = 1.0, 
            return_generation_state: bool = False
        ) -> AsyncGenerator[str, None]:
        """
        异步生成文本
        
        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            return_generation_state: 是否返回统计信息
        
        Yields:
            生成的文本片段 (逐 token 流式输出)
        """
                
        # 首先要把prompt保存到request中，并添加到shceduler管理器中
        req = self._make_request(prompt, max_new_tokens, temperature, top_p)
        self.scheduler.add_request(req)

        # 启动引擎循环 (如果还没启动)
        if self.engine_task is None or self.engine_task.done():
            self.engine_task = asyncio.create_task(self._engine_loop())

        # 接收已经生成的token，并以stream方式返回
        emitted = 0
        while not req.finished or emitted < req.gen_token_ids.size(1):
            # 后台引擎若失败，及时把异常抛到前台，避免静默卡住
            if self.engine_task is not None and self.engine_task.done():
                exc = self.engine_task.exception()
                if exc is not None:
                    raise RuntimeError("MiniVLLM engine loop failed") from exc

            cur = req.gen_token_ids.size(1)
            if cur > emitted:
                token_id = int(req.gen_token_ids[0, emitted].item())
                yield self.tokenizer.decode([token_id])
                emitted += 1
                continue
            await asyncio.sleep(0)
        
    def _make_request(
            self, 
            prompt: str, 
            max_new_tokens: int = 100, 
            temperature: float = 1.0, 
            top_p: float = 1.0
        )->Request:

        """创建请求对象"""
        input_ids = self.tokenizer.encode(prompt)
        req = Request(
            prefill_token_ids = input_ids, 
            max_gen_tokens = max_new_tokens, 
            sampling_params = SamplingParams(temperature, top_p)
        )
        req.eos_token_id = self.tokenizer.eos_token_id
        req.finished = False
        req.gen_token_ids = torch.empty((1, 0), dtype=torch.long)

        req.stats = RequestStats(request_id=req.request_id)
        req.stats.prompt_tokens = input_ids.size(1)
        req.stats.prefill_start_at = time.time()
        
        return req


    async def _engine_loop(self):
        """引擎主循环"""
        if self.engine_running:
            return
        
        self.engine_running = True
        try:
            while self.scheduler.has_pending():
                await self._step()
                await asyncio.sleep(0)
        except Exception as e:
            print(f"[Engine] Error: {type(e).__name__}: {e}")
            raise
        finally:
            self.engine_running = False

    async def _step(self):
        """单步执行: 处理 prefill 和 decode"""

        # 处理 waiting_list 中的请求 (prefill)
        if self.scheduler.waiting_list:
            waiting = self.scheduler.waiting_list[:]           
            prompted_ids =  await self._batch_prefill(waiting)
            for rid in prompted_ids:
                self.scheduler.promote_to_running(rid)
                # ！！！注意，我们在这里的处理会导致一个请求完成了prefill之后，会在同一个step中立刻进行一次decode。
                #  相当于我们在一个step中对一个request forward了两次。这会带来性能和调度公平性上的问题。
                # 不过我们是“玩具”引擎，为了简单直观，我们还是赶紧把他送往decode吧。
        
        # 处理 running_list 中的请求 (decode)
        if self.scheduler.running_list:
            running = self.scheduler.running_list[:]
            finished_ids = await self._batch_decode(running)
            for rid in finished_ids:
                self.scheduler.remove_request(rid)

    async def _batch_prefill(self, request_list: List[Request])->List[int]:
        """
        批量 Prefill 阶段
        
        关键步骤:
        1. 对不同长度的 prompt 做 padding
        2. 生成 attention_mask
        3. Batch forward
        4. 从 batch KV Cache 中切出每个请求的部分 (注意去掉 padding!)
        """

        if not request_list:
            return []
        
        
        start_time = time.time()
        
        device = self.model.model.device
        pad_id = self.tokenizer.pad_token_id 
        
        # 从每个request取出token tensor组成list，同时这些tensor 拷贝到目标设备上，如GPU
        # 当 device 是 GPU 时，.to(device) 通常会触发对应 tensor 的显存分配与数据拷贝，第一次用 CUDA 还可能有上下文初始化开销。
        seqs = [req.prefill_token_ids.squeeze(0).to(device) for req in request_list]

        # 把 list[int] 变成 torch.Tensor，这样后面才能做向量化张量运算（如 unsqueeze、广播比较）。
        # 放到同一 device（如 GPU），避免后续和 GPU tensor 运算时报设备不一致错误。
        lengths = torch.tensor([s.size(0) for s in seqs], device=device)

        # pad_sequence 函数把 list[int] 变成 torch.Tensor，并填充成相同长度，返回一个二维张量
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)

        # input_ids.size = [batch_size, max_len],直接说，参数0，行数，参数1，列数
        max_len = input_ids.size(1)

        # 获得attention_mask 张量，shape = [batch_size, max_len]，与input_ids张量一样，
        # 但是attention_mask张量中，1表示有效，0表示无效
        attention_mask = (
            torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()

        '''
        ----------------------------------------
        GPU 同步 + 记时
        ----------------------------------------
        '''
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_start = time.time()

        outputs = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.time() - forward_start

        # 取出每条请求的最后一个token的 logits
        batch_idx = torch.arange(len(request_list), device=device)
        last_pos = lengths - 1
        next_logits = outputs.logits[batch_idx, last_pos, :]   # 每条样本最后一个真实 token 位置        
        # next_logits shape: [batch_size, vocab_size]

        # 切分 KV Cache 并采样
        promoted = []
        self._ensure_paged_kv_cache(outputs.past_key_values)
        for i, req in enumerate(request_list):
            real_len = int(lengths[i].item())
            req_cache = DynamicCache()
            num_layers = self._get_cache_num_layers(outputs.past_key_values)

            for layer_idx in range(num_layers):
                k, v = self._get_layer_kv(outputs.past_key_values, layer_idx)
                k_req = k[i:i+1, :, :real_len, :].contiguous()
                v_req = v[i:i+1, :, :real_len, :].contiguous()
                req_cache.update(k_req, v_req, layer_idx)

            req.past_key_values = req_cache
            self._store_prefill_in_paged_cache(req, outputs.past_key_values, i, real_len)

            # 采样下一个 token
            tok = int(self._sample(next_logits[i:i+1,:], 
                                   req.sampling_params.temperature, 
                                   req.sampling_params.top_p
                                   ).item())
            req.next_token = tok
            req.gen_token_ids = torch.tensor([[tok]], device=device, dtype=torch.long)
            req.gen_text += self.tokenizer.decode([tok])


            '''
            ----------------------------------------
            GPU 同步 + 记时
            ----------------------------------------
            '''
            req.stats.prefill_end_at = time.time()
            req.stats.first_token_at = time.time()
            req.stats.prefill_kv_cache_mb = calculate_kv_cache_size_mb(req.past_key_values)


            # 检查是否结束
            req.finished = (tok == req.eos_token_id) or (req.max_gen_tokens <= 1)
            if req.finished:
                if req.block_table:
                    self.kv_cache.free_blocks(req.block_table)
                    req.block_table = []
                self.scheduler.remove_request(req.request_id)
            else:
                promoted.append(req.request_id)
            
            '''
            ----------------------------------------
            GPU 同步 + 记时
            ----------------------------------------
            '''
        elapsed = time.time() - start_time
        if self.enable_stats:
            self.engine_stats.record_prefill_batch(
                batch_size=len(request_list),
                elapsed_time=elapsed
            )

            print(f"[Prefill] batch_size={len(request_list)}, "
                  f"total={elapsed:.3f}s, forward={forward_time:.3f}s")
            
        return promoted
        
    async def _batch_decode(self, request_list: List[Request]) -> List[int]:
        if not request_list:
            return []
        
        start_time = time.time()

        device = self.model.model.device
        batch_size = len(request_list)
        kv_gather_time = 0.0
        forward_time = 0.0
        kv_append_time = 0.0

        finished_ids = []

        for req in request_list:
            gather_start = time.time()
            paged_cache = self._build_dynamic_cache_from_paged(req)
            kv_gather_time += time.time() - gather_start

            input_ids = torch.tensor([[req.next_token]], dtype=torch.long, device=device)
            attention_mask = torch.ones((1, req.seq_len + 1), dtype=torch.long, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_start = time.time()
            outputs = self.model.forward(
                input_ids=input_ids,
                past_key_values=paged_cache,
                attention_mask=attention_mask,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time += time.time() - forward_start

            next_token = int(
                self._sample(
                    outputs.logits[:, -1, :],
                    req.sampling_params.temperature,
                    req.sampling_params.top_p,
                ).item()
            )

            append_start = time.time()
            block_id, offset = self._ensure_decode_block(req)
            num_layers = self._get_cache_num_layers(outputs.past_key_values)
            for layer_idx in range(num_layers):
                k_full, v_full = self._get_layer_kv(outputs.past_key_values, layer_idx)
                k_new = k_full[:, :, -1:, :].contiguous()
                v_new = v_full[:, :, -1:, :].contiguous()
                self.kv_cache.append_token(layer_idx, block_id, offset, k_new, v_new)
            kv_append_time += time.time() - append_start

            req.past_key_values = outputs.past_key_values
            req.seq_len += 1
            req.gen_token_ids = torch.cat(
                [req.gen_token_ids, torch.tensor([[next_token]], device=device)],
                dim=-1,
            )
            req.num_gen_tokens = req.gen_token_ids.size(1)
            req.gen_text += self.tokenizer.decode([next_token])

            req.stats.decode_steps += 1
            req.stats.generated_tokens += 1

            gen_len = req.gen_token_ids.size(1)
            req.next_token = next_token
            req.finished = (next_token == req.eos_token_id) or (gen_len >= req.max_gen_tokens)
            if req.finished:
                req.stats.finished_at = time.time()
                req.stats.final_kv_cache_mb = calculate_kv_cache_size_mb(req.past_key_values)
                finished_ids.append(req.request_id)
                if req.block_table:
                    self.kv_cache.free_blocks(req.block_table)
                    req.block_table = []
                if self.enable_stats:
                    self.engine_stats.record_request_completed()
    
        # 记录引擎统计
        elapsed = time.time() - start_time
        if self.enable_stats:
            self.engine_stats.record_decode_batch(
                batch_size=batch_size,
                elapsed_time=elapsed
            )
        
            print(f"[Decode] batch_size={batch_size}, "
                f"total={elapsed:.3f}s, "
                f"gather={kv_gather_time:.3f}s, "
                f"forward={forward_time:.3f}s, "
                f"append={kv_append_time:.3f}s")    
        
        return finished_ids

    def _merge_past_kv(self, request_list):
        # 每个 req.past_key_values: tuple[(k,v)]，其中 k/v shape [1, H, S, D]
        num_layers = self._get_cache_num_layers(request_list[0].past_key_values)
        merged = []

        for l in range(num_layers):
            k_list = [self._get_layer_kv(req.past_key_values, l)[0] for req in request_list]
            v_list = [self._get_layer_kv(req.past_key_values, l)[1] for req in request_list]
            k = torch.cat(k_list, dim=0)  # [N,H,S,D]
            v = torch.cat(v_list, dim=0)  # [N,H,S,D]
            merged.append((k, v))

        return tuple(merged)

    def print_performance_report(self):
        """打印完整的性能报告"""
        if not self.enable_stats:
            print("性能统计已关闭（enable_stats=False），跳过性能报告。")
            return
        self.engine_stats.print_summary()

    def _split_past_kv_back(self, merged_past_kv, request_list):
        for i, req in enumerate(request_list):
            req.past_key_values = tuple(
                (k[i:i+1].contiguous(), v[i:i+1].contiguous())
                for (k, v) in merged_past_kv
            )


    async def _handle_decode(self, request_list: List[Request]):

        finished_ids = []

        for req in request_list:
            if req.finished:
                finished_ids.append(req.request_id)
                continue

            input_ids = torch.tensor([[req.next_token]], dtype=torch.long, device=self.model.model.device)
            outputs = self.model.forward(input_ids=input_ids, past_key_values=req.past_key_values)

            # next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            next_token = int(self._sample(outputs.logits[ :, -1, :], 
                                   req.sampling_params.temperature, 
                                   req.sampling_params.top_p
                                   ).item())
            req.past_key_values = outputs.past_key_values
            req.next_token = next_token
            req.gen_token_ids = torch.cat(
                [req.gen_token_ids, torch.tensor([[next_token]], device=req.gen_token_ids.device)],
                dim=-1,
            )
            req.gen_text += self.tokenizer.decode([next_token])

            gen_len = req.gen_token_ids.size(1)
            req.finished = (next_token == req.eos_token_id) or (gen_len >= req.max_gen_tokens)
            if req.finished:
                finished_ids.append(req.request_id)

        return finished_ids

    def _sample_batch(
        self,
        logits: torch.Tensor,
        sampling_params_list: List[SamplingParams],
    ) -> torch.Tensor:
        """
        批量采样 (为每个请求使用不同的 temperature/top_p)
        
        logits: [B, V]
        return: [B, 1]
        """
        batch_size = logits.size(0)
        result_tokens = []

        for i in range(batch_size):
            params = sampling_params_list[i]
            token = self._sample(
                logits[i : i + 1, :], params.temperature, params.top_p
            )
            result_tokens.append(token)

        return torch.cat(result_tokens, dim=0)  # [B, 1]

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """
        logits: [B, V]
        return: [B, 1]
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D [B, V], got shape={tuple(logits.shape)}")
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        # greedy
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # temperature scaling
        logits = logits / temperature

        # top-p (batch-safe)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)   # [B, V], [B, V]
            sorted_probs = torch.softmax(sorted_logits, dim=-1)                            # [B, V]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)                          # [B, V]

            # remove tokens with cumulative prob > top_p, but keep first token above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # scatter mask back to original vocab order
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)                  # [B, V]
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)                                               # [B, V]
        next_token_id = torch.multinomial(probs, num_samples=1)                            # [B, 1]
        return next_token_id


    def _sample_old(self, logits: torch.Tensor, temperature: float, top_p: float)->torch.Tensor:

            if temperature <= 0.0:
                # 直接取最大值
                return logits.argmax(dim=-1, keepdim=True)
            
            # 应用温度
            logits = logits / temperature

            # Top-p(nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                # 找到累计概率超过top_p的索引
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过top_p的token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # 将这些token的logits设为负无穷
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')

            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            # 从分布中采样
            next_token_id = torch.multinomial(probs, num_samples=1)
            return next_token_id
