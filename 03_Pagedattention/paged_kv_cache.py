from __future__ import annotations

import math
from typing import List

import torch


class PagedKVCache:
    """A minimal paged KV cache manager for the toy engine."""

    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    total_blocks: int
    free_block_ids: List[int]

    key_caches: List[torch.Tensor]
    value_caches: List[torch.Tensor]

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        total_blocks: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if total_blocks <= 0:
            raise ValueError(f"total_blocks must be positive, got {total_blocks}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        self.total_blocks = total_blocks
        # 初始化 free_block_ids 列表，构造时所有block都是空闲的
        self.free_block_ids = list(range(total_blocks))

        # 预先在目标设备上一次分配连续的缓存作为整个paded KV cache的缓存池
        cache_shape = (total_blocks, num_kv_heads, block_size, head_dim)
        self.key_caches = [
            torch.zeros(cache_shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        self.value_caches = [
            torch.zeros(cache_shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if num_blocks == 0:
            return []
        if num_blocks > len(self.free_block_ids):
            raise RuntimeError(
                f"Not enough free KV blocks: requested={num_blocks}, "
                f"available={len(self.free_block_ids)}"
            )

        # 将空闲ids切片，取出空闲blocks ids
        allocated = self.free_block_ids[:num_blocks]
        # 从空闲blocks列表中删除空闲blocks的ids
        del self.free_block_ids[:num_blocks]
        return allocated

    def free_blocks(self, block_ids: List[int]) -> None:
        if not block_ids:
            return

        # 用set相比list可以降低时间复杂度
        seen = set(self.free_block_ids)
        for block_id in block_ids:
            # 传入blockid有效性判断
            if block_id < 0 or block_id >= self.total_blocks:
                raise ValueError(f"Invalid block id: {block_id}")
            # 防止传入的blockid已经在free列表中，保证block id的唯一性
            if block_id in seen:
                raise ValueError(f"Block {block_id} is already free")
            # 将block id逐个加入free_block_ids 和seen set中
            self.free_block_ids.append(block_id)
            seen.add(block_id)

    def write_prefill(
        self,
        layer_idx: int,         # 写入哪一层
        block_ids: List[int],   # 为这个seq分配的block ids
        k: torch.Tensor,        # key tensor，形状[1, num_kv_heads, seq_len, head_dim]
        v: torch.Tensor,        # value tensor，形状[1, num_kv_heads, seq_len, head_dim]
        seq_len: int,           # 写入的token数量 可能小于k/v tensor的seq_len维度长度，因为可能只写入部分token（比如最后一块token数量不足block_size）
    ) -> None:
        self._validate_layer_idx(layer_idx)
        self._validate_kv_shape(k, v)
        self._validate_seq_len(seq_len, k)

        needed_blocks = self.get_num_blocks_needed(seq_len)
        if len(block_ids) != needed_blocks:
            raise ValueError(
                f"block_ids length mismatch: expected={needed_blocks}, got={len(block_ids)}"
            )

        # 获取缓存引用
        # 两个cache的shape都是[total_blocks, num_kv_heads, block_size, head_dim]
        key_cache = self.key_caches[layer_idx]
        value_cache = self.value_caches[layer_idx]

        # 截取有效的K/V序列，形状是[1, num_kv_heads, seq_len, head_dim]
        copied = 0 # 已经复制的token数量
        k_seq = k[0, :, :seq_len, :]
        v_seq = v[0, :, :seq_len, :]

        # 逐块写入缓存
        for block_id in block_ids:
            # 计算本块要写入多少token，不能超过block_size，也不能超过剩余的token数量（可能少于block_size）
            block_tokens = min(self.block_size, seq_len - copied)
            if block_tokens <= 0: # 如果没有token需要写入了，说明已经写满了请求的seq_len，可以退出循环了
                break

            # 将本块的缓存清零（可选，但有助于调试和避免过期数据干扰）
            key_cache[block_id].zero_()
            value_cache[block_id].zero_()

            # 将k_seq和v_seq中对应的token复制到缓存中，注意要根据block_tokens计算每块的token数量
            key_cache[block_id, :, :block_tokens, :].copy_(
                k_seq[:, copied : copied + block_tokens, :]
            )
            value_cache[block_id, :, :block_tokens, :].copy_(
                v_seq[:, copied : copied + block_tokens, :]
            )
            # 增加已拷贝token计数
            copied += block_tokens


    def append_token(
        self,
        layer_idx: int,     # 层索引
        block_id: int,      # 要写入的block id，必须是已经分配给这个请求的block id
        offset: int,        # 写入的token在block中的偏移量，必须是0到block_size-1之间
        k: torch.Tensor,    # key tensor，形状[1, num_kv_heads, 1, head_dim]
        v: torch.Tensor,    # value tensor，形状[1, num_kv_heads, 1, head_dim]
    ) -> None:
        self._validate_layer_idx(layer_idx)
        self._validate_block_id(block_id)
        if offset < 0 or offset >= self.block_size:
            raise ValueError(f"offset out of range: {offset}")

        # 把传入的k/v tensor规范化成[head_num, head_dim]的形状，方便后续复制到缓存中
        k_token = self._normalize_single_token_kv(k)
        v_token = self._normalize_single_token_kv(v)

        # 在目标block的offset位置写入新的token，注意这里不需要清零了，因为每个位置只会写入一次，不存在过期数据干扰的问题
        self.key_caches[layer_idx][block_id, :, offset, :].copy_(k_token)
        self.value_caches[layer_idx][block_id, :, offset, :].copy_(v_token)

    # 从分页缓存中提取（重组）完整的序列KV tensor。他是write_prefill的逆过程
    def gather_sequence(
        self,
        layer_idx: int,             # 层索引
        block_table: List[int],     # 请求的block id列表，按照序列顺序排列，长度必须大于等于get_num_blocks_needed(seq_len)返回的值
        seq_len: int,               # 获取的序列长度
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer_idx(layer_idx)
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")

        needed_blocks = self.get_num_blocks_needed(seq_len)
        if len(block_table) < needed_blocks:
            raise ValueError(
                f"block_table is too short: need {needed_blocks}, got {len(block_table)}"
            )

        # 空序列处理
        if seq_len == 0:
            # 返回形状为[1, num_kv_heads, 0, head_dim]的空tensor，注意这里seq_len是0了，但其他维度保持一致，方便后续拼接和处理
            empty = torch.empty(
                (1, self.num_kv_heads, 0, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty.clone()

        # 逐块提取和拼接，这是核心逻辑
        key_chunks = []         # 存储 key块切片
        value_chunks = []       # 存储 value块切片
        remaining = seq_len     # 还需要提取的token数量，初始值是seq_len，每提取一块就减少对应的token数量，直到提取完所有需要的token或者达到seq_len为止

        for block_id in block_table[:needed_blocks]:
            self._validate_block_id(block_id)

            # 计算本块要取多少个 token
            take_tokens = min(self.block_size, remaining)

            # 从缓存中提取数据，注意shape的变化
            key_chunks.append(
                self.key_caches[layer_idx][block_id : block_id + 1, :, :take_tokens, :]
            )
            value_chunks.append(
                self.value_caches[layer_idx][block_id : block_id + 1, :, :take_tokens, :]
            )
            # 减少剩余计数
            remaining -= take_tokens
        # 沿序列维度拼接所有块的切片，得到完整的序列KV tensor，形状是[1, num_kv_heads, seq_len, head_dim]
        return torch.cat(key_chunks, dim=2), torch.cat(value_chunks, dim=2)

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_num_blocks_needed(self, seq_len: int) -> int:
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")
        return math.ceil(seq_len / self.block_size)

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")

    def _validate_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.total_blocks:
            raise ValueError(f"Invalid block id: {block_id}")

    def _validate_seq_len(self, seq_len: int, k: torch.Tensor) -> None:
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")
        if seq_len > k.size(2):
            raise ValueError(
                f"seq_len exceeds KV tensor length: seq_len={seq_len}, kv_len={k.size(2)}"
            )

    def _validate_kv_shape(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.dim() != 4:
            raise ValueError(f"K/V must be 4D [1, H, S, D], got {tuple(k.shape)}")
        if k.size(0) != 1:
            raise ValueError(f"Only single-request KV is supported, got batch={k.size(0)}")
        if k.size(1) != self.num_kv_heads:
            raise ValueError(
                f"num_kv_heads mismatch: expected={self.num_kv_heads}, got={k.size(1)}"
            )
        if k.size(3) != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: expected={self.head_dim}, got={k.size(3)}"
            )


    # 将单token的k/v tensor规范化成[head_num, head_dim]的形状，方便后续复制到缓存中
    def _normalize_single_token_kv(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：检查输入的k/v tensor的形状，支持[1, H, 1, D]或者[H, 1, D]两种形状，并将其转换成[H, D]的形状
        if x.dim() == 4: # [1, H, 1, D]的情况
            # 确保batch 和 seq len 都是 1
            if x.size(0) != 1 or x.size(2) != 1:
                raise ValueError(
                    f"Expected [1, H, 1, D] for single token KV, got {tuple(x.shape)}"
                )
            x = x[0, :, 0, :]  # 切片成[H, D]的形状
        elif x.dim() == 3: # [H, 1, D]的情况
            if x.size(1) != 1: # 确保 seq len 是 1
                raise ValueError(
                    f"Expected [H, 1, D] for single token KV, got {tuple(x.shape)}"
                )
            x = x[:, 0, :] # 切片成[H, D]的形状
        elif x.dim() != 2:
            raise ValueError(f"Unsupported single token KV shape: {tuple(x.shape)}")
        # 第二步： 验证最终形状与缓存配置一致
        if x.size(0) != self.num_kv_heads or x.size(1) != self.head_dim:
            raise ValueError(
                f"Single token KV shape mismatch: expected=({self.num_kv_heads}, {self.head_dim}), "
                f"got={tuple(x.shape)}"
            )
        return x
