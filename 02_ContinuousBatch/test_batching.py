"""
测试 MiniVLLM 的 Continuous Batching 效果
对比单请求 vs 批量请求的性能
"""

import asyncio
import time
from mini_vllm import MiniVLLM


async def test_single_requests(mllm, prompts):
    """串行处理多个请求"""
    print("=" * 60)
    print("测试1: 串行处理 (无 batching)")
    print("=" * 60)
    
    start = time.time()
    
    for i, prompt in enumerate(prompts):
        # print(f"\n[Request {i+1}] Processing...")
        result = ""
        async for token in mllm.generate(prompt, max_new_tokens=500, temperature=0.0):
            result += token
        # print(f"Result: {result[:100]}...")
    
    elapsed = time.time() - start
    print(f"\n串行总耗时: {elapsed:.2f}秒")
    return elapsed


async def test_concurrent_requests(mllm, prompts):
    """并发处理多个请求 (测试 batching 效果)"""
    print("\n" + "=" * 60)
    print("测试2: 并发处理 (有 batching)")
    print("=" * 60)
    
    start = time.time()
    
    async def process_one(i, prompt):
        # print(f"\n[Request {i+1}] Started")
        result = ""
        async for token in mllm.generate(prompt, max_new_tokens=500, temperature=0.0):
            result += token
        # print(f"[Request {i+1}] Result: {result[:100]}...")
        return result
    
    # 并发启动所有请求
    tasks = [process_one(i, p) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"\n并发总耗时: {elapsed:.2f}秒")
    return elapsed


async def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    mllm = MiniVLLM(model_id)
    
    # 准备多个测试 prompt
    prompts = [
        "请用一句话介绍人工智能",
        "请用一句话介绍机器学习",
        "请用一句话介绍深度学习",
        "请用一句话介绍自然语言处理",
        "请用2句话介绍中国的电影发展历史",
        "请用3句话介绍中国的电影发展历史",
        "请用4句话介绍中国的电影发展历史",
        "请用5句话介绍中国的电影发展历史",
        "请用6句话介绍中国的电影发展历史",
        "请用7句话介绍中国的电影发展历史",
        "请用8句话介绍中国的电影发展历史",
        "请用9句话介绍中国的电影发展历史",
        "请用10句话介绍中国的电影发展历史",
        "请用11句话介绍中国的电影发展历史",
        "请用2句话介绍人工智能",
        "请用3句话介绍机器学习",
        "请用4句话介绍深度学习",
        "请用5句话介绍自然语言处理",
        "请用五句话介绍中国的发展历史",
        "请用五句话介绍中国的小说发展历史",
        "请用五句话介绍中国的诗歌发展历史",
        "请用五句话介绍中国的地形发展历史",
        "请用五句话介绍中国的英雄发展历史",
        "请用五句话介绍中国的游戏发展历史",
        "请用五句话介绍中国的电视发展历史",
        "请用五句话介绍中国的广播发展历史",
        "请用五句话介绍中国的音乐发展历史",
        "请用五句话介绍中国的自行车发展历史",
    ]
    
    # 预热
    print("预热中...")
    async for _ in mllm.generate("预热", max_new_tokens=10):
        pass
    
    # 测试1: 串行
    serial_time = await test_single_requests(mllm, prompts)
    
    # 重新创建 mllm (清理状态)
    mllm = MiniVLLM(model_id)
    
    # 测试2: 并发
    concurrent_time = await test_concurrent_requests(mllm, prompts)
    
    # 总结
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"串行耗时:   {serial_time:.2f}秒")
    print(f"并发耗时:   {concurrent_time:.2f}秒")
    print(f"加速比:     {serial_time/concurrent_time:.2f}x")
    print("\n💡 说明:")
    print("- 如果加速比 > 1, 说明 batching 生效了")
    print("- 加速比越接近请求数量, batching 效果越好")
    print("- Prefill batching 的加速效果通常更明显")


    # 总结
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"串行耗时:   {serial_time:.2f}秒")
    print(f"并发耗时:   {concurrent_time:.2f}秒")
    print(f"加速比:     {serial_time/concurrent_time:.2f}x")
    
    # 添加详细性能报告
    print("\n")
    mllm.print_performance_report()


if __name__ == "__main__":
    asyncio.run(main())


    
