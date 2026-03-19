from mini_vllm import MiniVLLM


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    mllm = MiniVLLM(model_id)

    prompt = "你好，请写一首关于码农写代码的五言绝句,谢谢。表现出码农面临家庭，工作，社会的经济压力和情感压力，还要努力debug，加班加点地工作。要现实主义风格但又带有一点浪漫主义的豪放。回复控制在100字以内。"
    generated_text = mllm.generate(prompt, max_new_tokens=1000, temperature=0.7, top_p=0.9)
    
    print(f"\n模型回答: {generated_text}")


