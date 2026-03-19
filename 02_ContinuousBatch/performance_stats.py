"""
MiniVLLM 性能分析系统
记录每个阶段的详细性能指标
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time
import torch

@dataclass
class RequestStats:
    """单个请求的统计信息"""
    request_id: int
    
    # 时间统计
    created_at: float = field(default_factory=time.time)
    prefill_start_at: Optional[float] = None
    prefill_end_at: Optional[float] = None
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    
    # Token 统计
    prompt_tokens: int = 0
    generated_tokens: int = 0
    
    # KV Cache 统计
    prefill_kv_cache_mb: float = 0.0
    final_kv_cache_mb: float = 0.0
    
    # Decode 步数统计
    decode_steps: int = 0
    
    @property
    def prefill_time(self) -> Optional[float]:
        if self.prefill_start_at and self.prefill_end_at:
            return self.prefill_end_at - self.prefill_start_at
        return None
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.created_at and self.first_token_at:
            return self.first_token_at - self.created_at
        return None
    
    @property
    def decode_time(self) -> Optional[float]:
        if self.first_token_at and self.finished_at:
            return self.finished_at - self.first_token_at
        return None
    
    @property
    def total_time(self) -> Optional[float]:
        if self.created_at and self.finished_at:
            return self.finished_at - self.created_at
        return None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.decode_time and self.decode_time > 0:
            return self.generated_tokens / self.decode_time
        return None


@dataclass
class EngineStats:
    """引擎全局统计"""
    
    # 步数统计
    total_steps: int = 0
    prefill_batches: int = 0
    decode_batches: int = 0
    
    # Batch size 统计
    prefill_batch_sizes: List[int] = field(default_factory=list)
    decode_batch_sizes: List[int] = field(default_factory=list)
    
    # 时间统计
    total_prefill_time: float = 0.0
    total_decode_time: float = 0.0
    
    # 每步的时间记录（用于分析）
    prefill_step_times: List[float] = field(default_factory=list)
    decode_step_times: List[float] = field(default_factory=list)
    
    # 请求统计
    completed_requests: int = 0
    
    def record_prefill_batch(self, batch_size: int, elapsed_time: float):
        """记录一次 Prefill Batch"""
        self.prefill_batches += 1
        self.prefill_batch_sizes.append(batch_size)
        self.total_prefill_time += elapsed_time
        self.prefill_step_times.append(elapsed_time)
    
    def record_decode_batch(self, batch_size: int, elapsed_time: float):
        """记录一次 Decode Batch"""
        self.decode_batches += 1
        self.decode_batch_sizes.append(batch_size)
        self.total_decode_time += elapsed_time
        self.decode_step_times.append(elapsed_time)
    
    def record_request_completed(self):
        """记录请求完成"""
        self.completed_requests += 1
    
    @property
    def avg_prefill_batch_size(self) -> float:
        if not self.prefill_batch_sizes:
            return 0.0
        return sum(self.prefill_batch_sizes) / len(self.prefill_batch_sizes)
    
    @property
    def avg_decode_batch_size(self) -> float:
        if not self.decode_batch_sizes:
            return 0.0
        return sum(self.decode_batch_sizes) / len(self.decode_batch_sizes)
    
    @property
    def avg_prefill_time(self) -> float:
        if not self.prefill_step_times:
            return 0.0
        return sum(self.prefill_step_times) / len(self.prefill_step_times)
    
    @property
    def avg_decode_time(self) -> float:
        if not self.decode_step_times:
            return 0.0
        return sum(self.decode_step_times) / len(self.decode_step_times)
    
    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 70)
        print("MiniVLLM 性能统计")
        print("=" * 70)
        
        print("\n【步数统计】")
        print(f"  总步数: {self.total_steps}")
        print(f"  Prefill batches: {self.prefill_batches}")
        print(f"  Decode batches: {self.decode_batches}")
        
        print("\n【Batch Size 统计】")
        print(f"  平均 Prefill batch size: {self.avg_prefill_batch_size:.2f}")
        print(f"  平均 Decode batch size: {self.avg_decode_batch_size:.2f}")
        print(f"  最大 Prefill batch size: {max(self.prefill_batch_sizes) if self.prefill_batch_sizes else 0}")
        print(f"  最大 Decode batch size: {max(self.decode_batch_sizes) if self.decode_batch_sizes else 0}")
        
        print("\n【时间统计】")
        print(f"  总 Prefill 时间: {self.total_prefill_time:.3f}s")
        print(f"  总 Decode 时间: {self.total_decode_time:.3f}s")
        print(f"  平均 Prefill 时间/batch: {self.avg_prefill_time:.3f}s")
        print(f"  平均 Decode 时间/batch: {self.avg_decode_time:.3f}s")
        
        print("\n【吞吐量】")
        print(f"  完成请求数: {self.completed_requests}")
        total_time = self.total_prefill_time + self.total_decode_time
        if total_time > 0:
            print(f"  请求吞吐量: {self.completed_requests / total_time:.2f} req/s")
        
        print("=" * 70 + "\n")


def calculate_kv_cache_size_mb(past_key_values) -> float:
    """计算 KV Cache 大小（MB）"""
    if past_key_values is None:
        return 0.0
    
    total_bytes = 0
    
    # 处理 DynamicCache
    if hasattr(past_key_values, '__len__'):
        for layer_kv in past_key_values:
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                k, v = layer_kv
                if torch.is_tensor(k):
                    total_bytes += k.element_size() * k.numel()
                if torch.is_tensor(v):
                    total_bytes += v.element_size() * v.numel()
    
    return total_bytes / (1024 ** 2)


# 使用示例
if __name__ == "__main__":
    # 模拟统计
    engine_stats = EngineStats()
    
    # 模拟 prefill
    engine_stats.record_prefill_batch(batch_size=10, elapsed_time=0.5)
    
    # 模拟 decode
    for i in range(50):
        batch_size = max(1, 10 - i // 10)  # 逐渐减小
        engine_stats.record_decode_batch(batch_size=batch_size, elapsed_time=0.03)
    
    # 打印统计
    engine_stats.print_summary()
