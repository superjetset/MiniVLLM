import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


#加载HF模型
class Model:
    def __init__(self, model_id: str, bnb_config= None, device_map="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        self.eos_token_id = self.model.config.eos_token_id
        self.device_map = self.model.device

    # 单次前向传播，可用于prefill和decode
    # 返回值: 模型输出，包含logits和新的past_key_values
    # 参数: input_ids: 输入的token ids
    #       past_key_values: 过去的key value对，用于加速生成，也就是 kv cache
    def forward(self, input_ids, past_key_values=None, attention_mask=None):
        if input_ids.device != self.model.device:
            input_ids = input_ids.to(self.model.device)
        
        if attention_mask is not None and attention_mask.device != self.model.device:
            attention_mask = attention_mask.to(self.model.device)


        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
            # outputs的类型是 transformers.modeling_outputs.CausalLMOutputWithPast，包含logits和past_key_values
            # 详细定义见 transformers.modeling_outputs.CausalLMOutputWithPast
        return outputs


        '''
        查看kv cache的大小和形状
        返回值: dict, 包含num_layers, total_size_mb, key_shape, value_shape
        其中key_shape和value_shape是torch.Size类型
        例子:
        {'num_layers': 28, 
        'total_size_mb': 345.0, 
        'key_shape': torch.Size([1, 16, 1024, 64]), 
        'value_shape': torch.Size([1, 16, 1024, 64])}
        解释:
        num_layers: 模型的层数
        total_size_mb: kv cache的总大小，单位MB
        key_shape: 每一层的key的形状 (batch_size, num_heads, seq_len, head_dim)
        value_shape: 每一层的value的形状 (batch_size, num_heads, seq_len, head_dim)
        计算方法:
        total_elements = sum over layers of (key_shape.numel() + value_shape.numel())
        total_size_mb = total_elements * 2 / (1024 ** 2)  # 假设每个元素2字节 (float16) 
        '''    
    def get_kv_cache(self, past_key_values):

        if past_key_values is None:
            return None
        
        num_layers = len(past_key_values)


        key_shape = past_key_values[0][0].shape  # (batch_size, num_heads, seq_len, head_dim)
        value_shape = past_key_values[0][1].shape  # (batch_size, num_heads, seq_len, head_dim)

        total_elements = sum([key_shape.numel() + value_shape.numel() for _ in range(num_layers)])
        
        #假设FP16，每个元素2字节
        totla_size_mb = total_elements * 2/(1024 ** 2)  # 每个元素2字节（float16）
        
        return {
            "num_layers": num_layers, 
            "total_size_mb": totla_size_mb, 
            "key_shape": key_shape, 
            "value_shape": value_shape
            }


    # 获取显存占用
    def get_memory_footprint(self):
        return self.model.get_memory_footprint()
    
