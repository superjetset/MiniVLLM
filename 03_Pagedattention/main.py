import asyncio
from mini_vllm import MiniVLLM


async def run_one(mllm: MiniVLLM, prompt: str):
    print(f"\n[Prompt] {prompt}")
    print("[Output] ", end="", flush=True)

    async for chunk in mllm.generate(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
        print(chunk, end="", flush=True)

    print("\n")


async def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    mllm = MiniVLLM(model_id, enable_stats=False)

    # 单请求
    #await run_one(mllm, "请写一个关于机器学习的短文，分三段。")

    # 多请求并发（用于观察 continuous batching）
    await asyncio.gather(
        run_one(mllm, "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n解释一下什么是梯度下降。\n<|im_end|>\n<|im_start|>assistant\n"),
        run_one(mllm, "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n用通俗语言介绍一下Transformer。\n<|im_end|>\n<|im_start|>assistant\n"),
    )


if __name__ == "__main__":
    asyncio.run(main())
