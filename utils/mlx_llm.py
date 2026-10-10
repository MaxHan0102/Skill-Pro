# utils/mlx_llm.py
"""
MLX 后端 LLM wrapper —— 在 Apple Silicon (arm64) Mac 上等价替代 LocalLLM(vLLM)。
- 生成:   mlx_lm.generate
- logprob: model.forward + log_softmax (参考 mlx_lm/perplexity.py)，供 PPO Gate 使用
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import torch


class MLXLLM:
    def __init__(self, model_path, max_new_tokens=1024, temp=0.6, top_p=0.9):
        print(f"[MLXLLM] Loading model: {model_path}")
        self.model, self.tokenizer = load(
            model_path, tokenizer_config={"trust_remote_code": True}
        )
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temp = temp
        self.top_p = top_p
        self.device = "mlx"
        # 让 SkillEvolution._compute_skill_logprobs 走 compute_logprob_batch 分支
        self.use_vllm = True

    def __call__(self, prompt: str) -> str:
        # mlx-lm 0.31+ 采样参数改用 sampler 对象（不再直接传 temp/top_p）
        sampler = make_sampler(temp=self.temp, top_p=self.top_p)
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text.strip()

    def compute_logprob_batch(self, prompts, targets, max_length=2048):
        """对每个 (prompt, target)，返回 target 部分各 token 的 logprob 之和。返回 torch.Tensor[B]。"""
        results = []
        for prompt, target in zip(prompts, targets):
            # 与 loss.py(HF 版)对齐：prompt 加 special tokens，target 不加
            p_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            t_ids = self.tokenizer.encode(target, add_special_tokens=False)
            combined = (p_ids + t_ids)[:max_length]
            p_len = len(p_ids)
            if p_len >= len(combined):          # target 被截断光
                results.append(0.0)
                continue

            input_ids = mx.array(combined, dtype=mx.int32)[None]      # [1, L]
            logits = self.model(input_ids).astype(mx.float32)         # [1, L, V]
            # logits[i] 预测 combined[i+1]
            log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)    # [1, L-1, V]
            nxt = mx.array(combined[1:], dtype=mx.int32)              # [L-1]
            # 每个位置取“实际下一个 token”的 logprob
            tok_logp = mx.take_along_axis(
                log_probs[0], nxt[:, None], axis=-1
            ).squeeze(-1)                                             # [L-1]
            # target token = combined[p_len:]; 对应 shift 索引 [p_len-1, L-1)
            target_logp = tok_logp[p_len - 1: len(combined) - 1]
            mx.eval(target_logp)
            results.append(float(target_logp.sum()))

        return torch.tensor(results, dtype=torch.float32)
