from pathlib import Path

from transformers import AutoTokenizer

#加载HF分词器
class Tokenizer:
    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.tokenizer = self._load_tokenizer(
            model_id=model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.pad_token_id = self.tokenizer.eos_token_id  # Fix padding token issue
        self.eos_token_id = self.tokenizer.eos_token_id

    def _load_tokenizer(
        self,
        model_id: str,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        if local_files_only:
            load_kwargs["local_files_only"] = True

        try:
            return AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:
            is_local_path = Path(model_id).exists()
            if local_files_only or is_local_path:
                raise RuntimeError(
                    f"Failed to load tokenizer from '{model_id}'. "
                    "Please verify the local path or cached files are complete."
                ) from exc

            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["local_files_only"] = True
            try:
                print(
                    "[Tokenizer] Remote load failed, retrying with local cache only. "
                    "If this also fails, use a local tokenizer path or pre-download the model files."
                )
                return AutoTokenizer.from_pretrained(model_id, **fallback_kwargs)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Failed to load tokenizer '{model_id}' from Hugging Face Hub, and no usable local cache was found. "
                    "Recommended fixes: retry network access, pre-download the model, or pass a local model directory."
                ) from fallback_exc

    # 编码方法
    # 返回值: tensor of token ids
    def encode(self, text: str):
        return self.tokenizer.encode(text, return_tensors='pt')

    # 解码方法
    # 返回值: 解码后的字符串
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
