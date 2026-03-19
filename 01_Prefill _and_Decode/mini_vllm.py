
import time
from model import Model
from tokenizer import Tokenizer


from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class PrefillOutput:
    outputs: Any
    generated_ids: torch.Tensor
    next_token_id: torch.Tensor
    past_key_values: Any


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



# MiniVLLM主类
class MiniVLLM:

    model = None
    tokenizer = None

    def __init__(self, model_id: str, bnb_config=None, device_map="cuda"):
        self.model = Model(model_id, bnb_config, device_map=device_map)
        self.tokenizer = Tokenizer(model_id)

    
    def generate(self, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 1.0, 
                 top_p: float = 1.0, 
                 return_generation_state: bool = False):
        
        generation_state = GenerationState(return_generation_state=return_generation_state)
        # prompt -> tokens
        input_ids = self.tokenizer.encode(prompt)
        input_ids = input_ids.to(self.model.model.device)

        if generation_state.return_generation_state:
            # 计时前先做CUDA同步，确保计时准确
            torch.cuda.synchronize()
            generation_state.prefill_time = time.time()

        # 处理 prefill阶段，获取初始的past_key_values和第一个生成的token id
        # past_key_values 就是KV Cache，会被decode阶段不断更新, 保存在outputs.past_key_values里
        outputs = self.prefill(input_ids, temperature=temperature, top_p=top_p, generation_state=generation_state)

        if generation_state.return_generation_state:
            torch.cuda.synchronize()
            generation_state.prefill_time = time.time() - generation_state.prefill_time
            generation_state.prompt_tokens = input_ids.size(1)
            generation_state.prefill_kv_cache_size_mb = self.kv_cache_size_mb(outputs.past_key_values)
            generation_state.decode_time = time.time()
        
        #处理 decode阶段，逐步生成新token     
        generated_ids = self.decode(outputs.next_token_id, 
                                    outputs.past_key_values, 
                                    max_new_tokens=max_new_tokens - 1, 
                                    eos_token_id=self.model.eos_token_id, 
                                    temperature=temperature, 
                                    top_p=top_p,
                                    generation_state=generation_state)

                
        if generation_state.return_generation_state:
            torch.cuda.synchronize()
            generation_state.decode_time = time.time() - generation_state.decode_time
        
        # 解码生成的token ids为文本, 只输出纯生成部分
        generated_text = self.tokenizer.decode(generated_ids[0])

        # 如果需要返回生成状态统计信息，计算相关数据
        if return_generation_state:
            generation_state.total_time = generation_state.prefill_time + generation_state.decode_time
            generation_state.num_tokens_generated = generated_ids.size(1)
            if generation_state.total_time > 0:
                generation_state.tokens_per_second = generation_state.num_tokens_generated / generation_state.total_time
            
            return generated_text, generation_state

        return generated_text


    #-------------------------------------------------------------
    # prefill阶段处理
    #-------------------------------------------------------------
    def prefill(self, input_ids: torch.Tensor, 
                temperature: float = 1.0, 
                top_p: float = 1.0, 
                generation_state=None)->PrefillOutput:
   
        prefill_output = PrefillOutput(outputs=None, generated_ids=None, next_token_id=None, past_key_values=None)
        
        #性能观测：计算生成第一个token的时间
        if generation_state is not None and generation_state.return_generation_state:
            torch.cuda.synchronize()
            generation_state.time_to_first_token = time.time()

        #1. 一次性将完整的 prompt输入给模型
        prefill_output.outputs = self.model.forward(input_ids)
        # 2. 获取初始 KV Cache（包含了 Prompt 中所有 token 的信息）
        # past_key_values 的形状通常是: [层数, 2 (K和V), Batch, Head数, 序列长, Head维度]
        prefill_output.past_key_values = prefill_output.outputs.past_key_values
		
        # 3. 采样出第一个生成的 token
        # 这里的 -1 是因为在 Prefill 阶段，我们其实计算了所有输入 token 的预测结果，但为了生成第一个新 token，我们只取最后一个位置的预测。
        logits = prefill_output.outputs.logits[:, -1, :]

        prefill_output.next_token_id = self._sample(logits, temperature, top_p)
        
        # 性能观测：记录prefill阶段的耗时
        if generation_state is not None and generation_state.return_generation_state:
            torch.cuda.synchronize()
            generation_state.time_to_first_token = time.time() - generation_state.time_to_first_token

         # 4 新token拼接到生成序列中
        prefill_output.generated_ids = torch.cat([input_ids, prefill_output.next_token_id], dim=-1)

        return prefill_output


    #-------------------------------------------------------------
    # decode阶段处理
    #-------------------------------------------------------------
    def decode(self, next_token_id, past_key_values, 
               max_new_tokens: int = 100, 
               eos_token_id: int = None, 
               temperature: float = 1.0, 
               top_p: float = 1.0,
               generation_state: GenerationState = None)->torch.Tensor:
        
        # 初始化生成的token ids为空
        generated_ids = torch.empty((1, 0), dtype=torch.long).to(self.model.model.device)

        # 先把第一个token拼接进去
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        new_tokens_list = []
        # 自回归循环，逐步生成新 token
        for _ in range(max_new_tokens):
            # ！这里是核心关键：每次只传入【一个】token_id 和 之前的past_key_values
            outputs = self.model.forward(next_token_id, past_key_values=past_key_values)
            
            # 刷新 KV cache：模型会将当前token的kv 追加到历史缓存中
            past_key_values = outputs.past_key_values
            # 获取最后一个位置的 logits，采样下一个 token
            logits = outputs.logits[:, -1, :]
            next_token_id = self._sample(logits, temperature, top_p)
            
            # 拼接到生成序列中
            new_tokens_list.append(next_token_id)

            # 如果生成了结束符，则停止生成
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
        
        if new_tokens_list:
            new_token_tensor = torch.cat(new_tokens_list, dim=-1)
            generated_ids = torch.cat([generated_ids, new_token_tensor], dim=-1)

        if generation_state is not None and generation_state.return_generation_state:
            generation_state.kv_cache_size_mb = self.kv_cache_size_mb(past_key_values)
            if next_token_id.item() == eos_token_id:    
                generation_state.eos_info = True            

        return generated_ids


    def greet(self):
        return "Hello from MiniVLLM!"
    
    #--------------------------------------------------------------
    # 计算KV Cache的大小，单位MB
    # kv cache 大小的计算规则: size_bm = batch_size * head_num * head_dim * seq_len * 2(K&V) * 2(FP16) / (1024*1024)
    #--------------------------------------------------------------
    def kv_cache_size_mb(self, past_key_values):
        total_size_mb = 0.0
        for layer_kv in past_key_values:
            # layer_kv 是一个 tuple，包含 (key, value)
            # 常见时(k,v)
            for kv in layer_kv:
                if torch.is_tensor(kv):
                    total_size_mb += kv.element_size() * kv.numel()
        return total_size_mb / (1024 ** 2)
    

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float)->torch.Tensor:
        ''' 
        从lofits 中采样下一个token id
        temperature： 控制采样的随机性，temperature越高，采样越随机
        top_p: 采用 nucleus sampling，保留累计概率达到top_p的token 
        '''
        
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