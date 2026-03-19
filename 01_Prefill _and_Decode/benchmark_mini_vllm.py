"""
MiniVLLM性能分析脚本
用于理解Prefill vs Decode的性能差异
"""
from mini_vllm import MiniVLLM
import time
import torch

def benchmark_generation(model_id: str, prompts: list, max_new_tokens: int = 200):
    """
    测试不同prompt长度对性能的影响
    """
    print("=" * 60)
    print(f"测试模型: {model_id}")
    
    mllm = MiniVLLM(model_id)
    
    # 预热GPU
    print("预热中" + "=" * 60)
    for _ in range(1):
        mllm.generate("预热1", 
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                return_generation_state=False)
        torch.cuda.synchronize()
    print("=" * 60 + "-> 预热完成 <-")

    print(f"生成token数: {max_new_tokens}")
    print("=" * 60)
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[测试 {i+1}/{len(prompts)}]")
        # print(f"Prompt:{prompt}")
        
        # 生成并获取统计信息
        text, stats = mllm.generate(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            return_generation_state=True
        )
        # print(f"------------------------------------------------------")
        # print(f"生成结果: {text}")
        # print(f"------------------------------------------------------")
        print(f"✓ Prompt长度: {len(prompt)} 词")
        print(f"✓ Prompt Token数{stats.prompt_tokens}")
        print(f"✓ 生成第一个Token耗时{stats.time_to_first_token:.3f}秒")
        print(f"✓ Prefill耗时: {stats.prefill_time:.3f}秒")
        print(f"✓ Decode耗时: {stats.decode_time:.3f}秒")
        print(f"✓ 生成token: {stats.num_tokens_generated}")
        print(f"✓ EOS info: {stats.eos_info}")
        print(f"✓ 总耗时: {stats.total_time:.3f}秒")
        print(f"✓ 生成速度: {stats.tokens_per_second:.2f} tokens/s")
        print(f"✓ prefill阶段KV Cache大小：{stats.prefill_kv_cache_size_mb:.2f} MB")
        print(f"✓ KV Cache大小: {stats.kv_cache_size_mb:.2f} MB")
        
        results.append({
            'prompt_length': len(prompt),
            'prompt_tokens': stats.prompt_tokens, 
            'prefill_time': stats.prefill_time,
            'decode_time': stats.decode_time,
            'tokens_per_second': stats.tokens_per_second,
            'kv_cache_size_mb': stats.kv_cache_size_mb
        })
    
    # 分析结果
    print("\n" + "=" * 60)
    print("性能分析总结:")
    print("=" * 60)
    
    print("\n观察1: Prefill vs Decode时间对比")
    for i, r in enumerate(results):
        ratio = r['prefill_time'] / r['decode_time'] if r['decode_time'] > 0 else 0
        print(f"  Prompt_token数 {r['prompt_tokens']} : "
              f"Prefill {r['prefill_time']:.3f}s vs Decode {r['decode_time']:.3f}s "
              f"(比例 {ratio:.2f})")
    
    print("\n观察2: KV Cache随prompt长度增长")
    for i, r in enumerate(results):
        print(f"  Prompt_token数 {r['prompt_tokens']} : "
              f"KV Cache {r['kv_cache_size_mb']:.2f} MB")

    print("\n学习要点:")
    print("  1. Prefill阶段时间随prompt token 数量线性增长")
    print("  2. Decode阶段每个token的时间基本恒定")
    print("  3. KV Cache大小 = (prompt token长度 + 生成长度) × 模型层数 × hidden维度")

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 不同长度的prompt
    prompts = [
        "唉",  # 短prompt                 
        "吃饭",
        "站起来,我们走吧",
        "天大地大",
        "你是老师，请详细介绍一下人工智能的发展历史，包括重要的里程碑事件。",  # 中等prompt
        "你是一个老师，请详细介绍一下人工智能的发展历史，从图灵测试开始，到深度学习的兴起，再到大语言模型的突破，包括每个阶段的重要人物、关键技术和代表性成果。", # 长prompt
        "你好，你是一个诗人，请写一首关于码农写代码的五言绝句,谢谢。要求：表现出码农不仅面临家庭，工作，社会的经济压力和情感压力，以及对抗BOSS的PUA，还要担心被贼惦记，还要努力debug加班加点地工作。。甚至在路上捡到10块钱也要交给警察。"
        "要现实主义风格但又带有一点浪漫主义的豪放，最好能参考白居易的作品，不过白居易好像很少写绝句，还是参考杜甫吧，如果你喜欢李白，也是可以的。直接写诗，不要输出任何废话" # 超长prompt
    ]
    
    benchmark_generation(model_id, prompts, max_new_tokens=500)