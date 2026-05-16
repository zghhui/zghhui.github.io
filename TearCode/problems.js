window.PROBLEMS = [
    {
        id: 1,
        title: "多头注意力 (Multi-Head Attention)",
        category: "attention",
        difficulty: "easy",
        description: `## 多头注意力机制

实现标准的 Multi-Head Attention 模块。

### 核心公式
\`\`\`
Attn(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
\`\`\`

### 要求
- 输入: Q, K, V 形状为 (B, L, D)
- 通过线性层投影后分成 num_heads 个头
- 每个头独立做 Scaled Dot-Product Attention
- 支持可选的 mask 参数
- 最后合并多头并通过输出投影层

### 关键步骤
1. 线性变换: W_q, W_k, W_v 投影
2. 分头: (B, L, D) -> (B, H, L, d_k)
3. 注意力: softmax(QK^T / sqrt(d_k)) V
4. 合并: (B, H, L, d_k) -> (B, L, D)
5. 输出投影: W_o`,
        template: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # TODO

    def forward(self, Q, K, V, mask=None):
        # TODO
`,
        reference: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.shape

        Q = self.W_q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)`,
        hints: [
            "d_k = d_model // num_heads，每个头处理的维度",
            "分头用 .view(B, -1, num_heads, d_k).transpose(1, 2)",
            "合并用 .transpose(1, 2).contiguous().view(B, L, -1)",
            "别忘了 scale factor: sqrt(d_k)",
            "mask 应用: scores.masked_fill(mask == 0, float('-inf'))"
        ],
        keyPoints: [
            "Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V",
            "分头: view + transpose",
            "合并: transpose + contiguous + view",
            "mask 用 -inf 填充保证 softmax 后为 0"
        ]
    },
    {
        id: 2,
        title: "Transformer (Encoder + Decoder)",
        category: "transformer",
        difficulty: "medium",
        description: `## Transformer 完整实现

实现包含 Encoder 和 Decoder 的完整 Transformer 模型。

### 架构要点
- **Encoder**: Pre-Norm + Self-Attention + FFN
- **Decoder**: Pre-Norm + Self-Attention + Cross-Attention + FFN
- **位置编码**: 正弦/余弦位置编码
- **FFN**: Linear -> GELU -> Linear

### 要求
1. TransformerEncoderLayer: self-attn + ffn + pre-norm + residual
2. TransformerDecoderLayer: self-attn + cross-attn + ffn + pre-norm + residual
3. Transformer: embedding + 位置编码 + encoder stack + decoder stack + output projection`,
        template: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # TODO

    def forward(self, x, mask=None):
        # TODO


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # TODO

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # TODO


class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6,
                 vocab_size=30000, max_len=512, dropout=0.1):
        # TODO

    def encode(self, src, mask=None):
        # TODO

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        # TODO

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # TODO
`,
        reference: `class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6,
                 vocab_size=30000, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src, mask=None):
        x = self.embed(src) * math.sqrt(self.d_model) + self.pe[:, :src.size(1)]
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(tgt) * math.sqrt(self.d_model) + self.pe[:, :tgt.size(1)]
        for layer in self.dec_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        return self.out_proj(self.decode(tgt, enc_out, src_mask, tgt_mask))`,
        hints: [
            "Pre-Norm: 先 LayerNorm 再送入子层",
            "Encoder 每层: self-attn + ffn，各加残差",
            "Decoder 每层: self-attn + cross-attn + ffn",
            "Cross-Attention: Q 来自 decoder，K/V 来自 encoder output",
            "位置编码: pe[:, 0::2] = sin, pe[:, 1::2] = cos",
            "Embedding 要乘 sqrt(d_model) 再加位置编码"
        ],
        keyPoints: [
            "Pre-Norm vs Post-Norm",
            "Cross-Attention 的 Q/K/V 来源",
            "正弦位置编码的频率公式",
            "Embedding scaling: * sqrt(d_model)"
        ]
    },
    {
        id: 3,
        title: "Flow Matching 训练与采样",
        category: "flow",
        difficulty: "medium",
        description: `## Flow Matching

实现 Flow Matching 的训练步骤和 Euler ODE 采样。

### 核心公式
- 插值路径: \`x_t = (1 - sigma(t)) * x_0 + sigma(t) * eps\`
- 速度场目标: \`v_t = d_sigma(t) * (eps - x_0)\`
- Schedule: 线性 sigma(t)=t，余弦 sigma(t)=sin(pi*t/2)
- Logit-Normal 时间采样: \`t = sigmoid(N(0,1))\`

### 要求
1. 训练函数: 采样时间 t，构造 x_t，预测速度场，计算 MSE loss
2. 采样函数: 从 t=1 到 t=0 的 Euler ODE 积分，支持 CFG`,
        template: `import torch
import torch.nn.functional as F
import math


def flow_matching_train_step(model, optimizer, x_0, cond=None, schedule='linear'):
    # TODO


@torch.no_grad()
def flow_matching_sample(model, shape, num_steps=50, cond=None, cfg_scale=7.5):
    # TODO
`,
        reference: `def flow_matching_train_step(model, optimizer, x_0, cond=None, schedule='linear'):
    B = x_0.shape[0]

    u = torch.randn(B, device=x_0.device)
    t = torch.sigmoid(u).clamp(1e-5, 1 - 1e-5)

    if schedule == 'cosine':
        sigma_t = torch.sin(t * math.pi / 2)
        d_sigma_t = (math.pi / 2) * torch.cos(t * math.pi / 2)
    else:
        sigma_t = t
        d_sigma_t = torch.ones_like(t)

    t_expand = t.view(-1, *([1] * (x_0.dim() - 1)))
    sigma_expand = sigma_t.view(-1, *([1] * (x_0.dim() - 1)))
    d_sigma_expand = d_sigma_t.view(-1, *([1] * (x_0.dim() - 1)))

    eps = torch.randn_like(x_0)
    x_t = (1 - sigma_expand) * x_0 + sigma_expand * eps

    target_v = d_sigma_expand * (eps - x_0)

    pred_v = model(x_t, t, cond)
    loss = F.mse_loss(pred_v, target_v)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def flow_matching_sample(model, shape, num_steps=50, cond=None, cfg_scale=7.5):
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)

    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur = timesteps[i]
        dt = timesteps[i + 1] - t_cur

        t_batch = torch.full((shape[0],), t_cur, device=device)

        v_uncond = model(x, t_batch, cond=None)
        v_cond = model(x, t_batch, cond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        x = x + v * dt

    return x`,
        hints: [
            "Logit-Normal 采样: t = sigmoid(randn)，使中间时间步被更频繁采样",
            "sigma/d_sigma 需要 expand 到与 x_0 相同维度才能广播",
            "CFG: v = v_uncond + scale * (v_cond - v_uncond)",
            "Euler 步: x_new = x + v * dt，注意 dt 是负值（从1到0）",
            "t 要 clamp 到 (1e-5, 1-1e-5) 避免边界问题"
        ],
        keyPoints: [
            "v_t = d_sigma(t) * (eps - x_0)",
            "x_t = (1-sigma)*x_0 + sigma*eps",
            "Logit-Normal 时间采样",
            "CFG 在采样时引导"
        ]
    },
    {
        id: 4,
        title: "KL 散度计算",
        category: "kl",
        difficulty: "easy",
        description: `## KL 散度

实现三种常见的 KL 散度计算。

### 公式
1. **两个高斯分布**: KL(p||q) = 0.5 * [log(var_q/var_p) + (var_p + (mu_p-mu_q)^2)/var_q - 1]
2. **KL 到标准正态**: KL(N(mu,sigma^2)||N(0,I)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
3. **离散分布 KL**: KL(p||q) = sum(p * (log_p - log_q))

### 要求
- 输入用 logvar（log方差）表示，数值更稳定
- 离散分布从 logits 计算
- 输出在最后一维 sum，保留 batch 维`,
        template: `import torch
import torch.nn.functional as F


def kl_gaussian(mu_p, logvar_p, mu_q, logvar_q):
    # TODO


def kl_to_standard_normal(mu, logvar):
    # TODO


def kl_categorical(logits_p, logits_q):
    # TODO
`,
        reference: `def kl_gaussian(mu_p, logvar_p, mu_q, logvar_q):
    return 0.5 * (logvar_q - logvar_p + (logvar_p.exp() + (mu_p - mu_q).pow(2)) / logvar_q.exp() - 1).sum(-1)


def kl_to_standard_normal(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)


def kl_categorical(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)`,
        hints: [
            "logvar.exp() = var，不需要手动取 exp 后再取 log",
            "VAE 的 KL 正则项是 KL(q(z|x) || N(0,I))",
            "离散 KL 用 log_softmax 计算更稳定",
            "sum(-1) 对最后一维求和"
        ],
        keyPoints: [
            "KL = 0.5 * [log(var_q/var_p) + (var_p + (mu_p-mu_q)^2)/var_q - 1]",
            "VAE KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))",
            "用 logvar 而非 var 保证数值稳定"
        ]
    },
    {
        id: 5,
        title: "归一化层 (RMSNorm / LayerNorm / BatchNorm)",
        category: "norm",
        difficulty: "easy",
        description: `## 归一化层

手写三种归一化层：RMSNorm、LayerNorm、BatchNorm1d。

### 对比
| 特性 | BatchNorm | LayerNorm | RMSNorm |
|------|-----------|-----------|---------|
| 归一化维度 | batch 维 | 特征维 | 特征维 |
| 去均值 | 是 | 是 | 否 |
| 训练/推理差异 | 是 | 否 | 否 |
| 适用场景 | CV (CNN) | NLP/Transformer | LLM (LLaMA) |

### 要求
- RMSNorm: 只做缩放，不去均值
- LayerNorm: 去均值 + 缩放
- BatchNorm1d: 训练用 batch 统计量，推理用 running stats`,
        template: `import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        # TODO

    def forward(self, x):
        # TODO


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        # TODO

    def forward(self, x):
        # TODO


class BatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        # TODO

    def forward(self, x):
        # TODO
`,
        reference: `class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class BatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(d))
        self.register_buffer('running_var', torch.ones(d))

    def forward(self, x):
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias`,
        hints: [
            "RMSNorm 没有 bias，只有 weight (gamma)",
            "RMS = sqrt(mean(x^2) + eps)，不去均值",
            "LayerNorm 的 var 用 unbiased=False",
            "BatchNorm 用 register_buffer 存 running stats",
            "running stats 更新用 lerp_（指数移动平均）"
        ],
        keyPoints: [
            "RMSNorm: x * weight / sqrt(mean(x^2)+eps)",
            "LayerNorm: weight * (x-mean) / sqrt(var+eps) + bias",
            "BatchNorm: 训练用 batch 统计量，推理用 running stats",
            "register_buffer vs nn.Parameter"
        ]
    },
    {
        id: 6,
        title: "GRPO 手撕",
        category: "grpo",
        difficulty: "hard",
        description: `## GRPO (Group Relative Policy Optimization)

实现 DeepSeek-R1 风格的 GRPO 训练步骤。

### 核心思想
- 对同一 prompt 采样一组响应，用组内相对排名代替 value 网络
- 优势函数: A_i = (r_i - mean) / std
- PPO 裁剪 + KL 正则

### 公式
- 优势: \`A_i = (r_i - mean(r)) / (std(r) + eps)\`
- 比率: \`ratio = exp(cur_logprob - old_logprob)\`
- 裁剪: \`loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)\`
- KL 惩罚: \`kl = cur_logprob - ref_logprob\``,
        template: `import torch


def grpo_train_step(policy, ref_model, reward_fn, prompts, optimizer,
                    group_size=8, clip_eps=0.2, beta_kl=0.01):
    # TODO
`,
        reference: `def grpo_train_step(policy, ref_model, reward_fn, prompts, optimizer,
                    group_size=8, clip_eps=0.2, beta_kl=0.01):
    total_loss = 0.0

    for prompt in prompts:
        with torch.no_grad():
            responses, old_logprobs = zip(*[
                policy.generate(prompt, return_logprob=True) for _ in range(group_size)
            ])
            rewards = torch.tensor([reward_fn(prompt, r) for r in responses])

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for i in range(group_size):
            cur_logprob = policy.get_logprob(prompt, responses[i])
            ref_logprob = ref_model.get_logprob(prompt, responses[i])

            ratio = torch.exp(cur_logprob - old_logprobs[i])
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[i]

            policy_loss = -torch.min(surr1, surr2)
            kl = cur_logprob - ref_logprob

            total_loss += policy_loss + beta_kl * kl

    total_loss = total_loss / (len(prompts) * group_size)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()`,
        hints: [
            "采样阶段用 torch.no_grad()，不需要梯度",
            "advantages 是组内标准化: (r - mean) / (std + eps)",
            "ratio = exp(new_logp - old_logp)，不是直接除",
            "PPO 裁剪: clamp ratio 到 [1-eps, 1+eps]，取 min(surr1, surr2)",
            "KL 简单版: kl = cur_logprob - ref_logprob"
        ],
        keyPoints: [
            "组内相对优势代替 value 网络",
            "PPO 裁剪防止策略更新过大",
            "KL 惩罚保持与参考模型接近",
            "ratio = exp(log_pi - log_pi_old)"
        ]
    },
    {
        id: 7,
        title: "旋转位置编码 (RoPE)",
        category: "rope",
        difficulty: "medium",
        description: `## 旋转位置编码 (RoPE)

实现 LLaMA/GPT 等模型使用的旋转位置编码。

### 核心公式
- \`q_rot = q * cos(m*theta) + rotate_half(q) * sin(m*theta)\`
- 性质: <R(m)q, R(n)k> 只依赖相对位置 m-n
- 频率: \`theta_i = 1 / (base^(2i/d))\`

### rotate_half 操作
将向量的前半部分和后半部分交换，后半变负:
\`[x1, x2, x3, x4] -> [-x3, -x4, x1, x2]\`

### 要求
1. 预计算 inv_freq 和 cos/sin 缓存
2. 实现 rotate_half
3. 对 q, k 分别应用旋转`,
        template: `import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, d_k, max_len=4096, base=10000.0):
        # TODO

    def forward(self, q, k):
        # TODO
`,
        reference: `class RoPE(nn.Module):
    def __init__(self, d_k, max_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)

        pos = torch.arange(max_len).float()
        angles = torch.outer(pos, inv_freq)
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def forward(self, q, k):
        L = q.shape[2]
        cos = self.cos[:L].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:L].unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k`,
        hints: [
            "inv_freq 的维度是 d_k/2，因为每两个维度一组",
            "angles 需要 cat([angles, angles]) 扩展到 d_k 维度",
            "rotate_half: 把 x 分成前后两半，后半变负放前面",
            "cos/sin 用 register_buffer 缓存，推理时不需要重新计算",
            "unsqueeze(0).unsqueeze(0) 是为了广播 (1,1,L,d_k)"
        ],
        keyPoints: [
            "q_rot = q*cos + rotate_half(q)*sin",
            "inv_freq = 1/(base^(2i/d))",
            "rotate_half: [-x2, x1]",
            "相对位置编码: <R(m)q, R(n)k> 只依赖 m-n"
        ]
    },
    {
        id: 8,
        title: "Flow-GRPO 手撕",
        category: "flow-grpo",
        difficulty: "hard",
        description: `## Flow-GRPO

结合 Flow Matching 和 GRPO 的训练方法。

### 核心思想
- ODE 转 SDE：注入随机性使采样成为随机策略
- SDE: \`dx = [v + sigma^2/(2t) * (x + (1-t)v)] dt + sigma * dw\`
- sigma(t) = a * sqrt(t / (1-t))
- 组内相对优势 + PPO 裁剪

### 训练流程
1. SDE 采样一组样本，记录轨迹
2. Reward 打分，组内标准化优势
3. 逐步计算转移概率的 log prob
4. PPO 裁剪 + KL 惩罚（用速度场 MSE）`,
        template: `import torch
import torch.nn.functional as F
import math


def flow_grpo_train_step(flow_model, ref_model, reward_fn, cond, optimizer,
                         latent_shape, group_size=4, num_steps=10,
                         clip_eps=0.2, beta_kl=0.01, noise_scale=0.3):
    # TODO
`,
        reference: `def flow_grpo_train_step(flow_model, ref_model, reward_fn, cond, optimizer,
                         latent_shape, group_size=4, num_steps=10,
                         clip_eps=0.2, beta_kl=0.01, noise_scale=0.3):
    device = next(flow_model.parameters()).device
    dt = 1.0 / num_steps

    def get_sigma(t_val):
        return noise_scale * math.sqrt(t_val / (1 - t_val + 1e-8))

    group_x0s, group_trajs = [], []
    with torch.no_grad():
        for _ in range(group_size):
            x = torch.randn(latent_shape, device=device)
            traj = []
            for step in range(num_steps):
                t_cur = 1.0 - step * dt
                sigma = get_sigma(t_cur)
                t_batch = torch.full((x.shape[0],), t_cur, device=device)

                v = flow_model(x, t_batch, cond)
                drift = v + (sigma**2) / (2*t_cur) * (x + (1-t_cur)*v)
                noise = torch.randn_like(x)
                x_next = x - drift * dt + sigma * math.sqrt(dt) * noise

                traj.append({'x': x.clone(), 't': t_cur, 'x_next': x_next.clone()})
                x = x_next

            group_x0s.append(x)
            group_trajs.append(traj)

    rewards = torch.tensor([reward_fn(x0, cond) for x0 in group_x0s], device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = 0.0
    for i in range(group_size):
        for step_info in group_trajs[i]:
            x_t, t_cur, x_next = step_info['x'], step_info['t'], step_info['x_next']
            sigma = get_sigma(t_cur)
            t_batch = torch.full((x_t.shape[0],), t_cur, device=device)

            v = flow_model(x_t, t_batch, cond)
            drift = v + (sigma**2) / (2*t_cur) * (x_t + (1-t_cur)*v)
            mean = x_t - drift * dt

            var = sigma**2 * dt
            log_prob = -0.5 * ((x_next - mean)**2 / var).sum(dim=[1,2,3])

            with torch.no_grad():
                v_ref = ref_model(x_t, t_batch, cond)
                drift_ref = v_ref + (sigma**2) / (2*t_cur) * (x_t + (1-t_cur)*v_ref)
                mean_ref = x_t - drift_ref * dt
                log_prob_ref = -0.5 * ((x_next - mean_ref)**2 / var).sum(dim=[1,2,3])

            ratio = torch.exp(log_prob - log_prob_ref)
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[i]
            total_loss += -torch.min(surr1, surr2).mean()

            total_loss += beta_kl * 0.5 * ((v - v_ref)**2).sum(dim=[1,2,3]).mean()

    total_loss = total_loss / (group_size * num_steps)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()`,
        hints: [
            "SDE drift = v + sigma^2/(2t) * (x + (1-t)*v)",
            "sigma(t) = a * sqrt(t/(1-t))，t 接近 0 时 sigma 趋于 0",
            "转移概率: 高斯分布，均值是 x - drift*dt，方差是 sigma^2*dt",
            "log_prob 对空间维 sum，对 batch 维 mean",
            "KL 用速度场 MSE: ||v - v_ref||^2，而非 log prob 差"
        ],
        keyPoints: [
            "ODE -> SDE 转换注入随机性",
            "SDE drift 公式",
            "转移概率的高斯似然",
            "PPO 裁剪 + 速度场 KL"
        ]
    },
    {
        id: 9,
        title: "DiffusionNFT 手撕",
        category: "diffusion-nft",
        difficulty: "hard",
        description: `## DiffusionNFT

实现 DiffusionNFT 的对比训练方法。

### 核心思想
- 不需要轨迹存储和似然估计
- 前向 Flow Matching loss：正样本拟合，负样本远离
- 用 ODE 采样生成样本，打分分正负组
- 正样本: 降低 FM loss（让模型更好地生成类似样本）
- 负样本: 增大 FM loss（让模型远离此类样本）

### 关键区别
vs Flow-GRPO: 无需 SDE、无需轨迹存储、无需似然估计`,
        template: `import torch
import torch.nn.functional as F


def diffusion_nft_train_step(model, reward_fn, cond, optimizer,
                             latent_shape, group_size=8, num_steps=28):
    # TODO
`,
        reference: `def diffusion_nft_train_step(model, reward_fn, cond, optimizer,
                             latent_shape, group_size=8, num_steps=28):
    device = next(model.parameters()).device

    samples = []
    with torch.no_grad():
        for _ in range(group_size):
            x = torch.randn(latent_shape, device=device)
            ts = torch.linspace(1, 0, num_steps + 1, device=device)
            for j in range(num_steps):
                t_batch = torch.full((x.shape[0],), ts[j], device=device)
                v = model(x, t_batch, cond)
                x = x + v * (ts[j+1] - ts[j])
            samples.append(x)

    rewards = torch.tensor([reward_fn(s, cond) for s in samples])
    mean_r = rewards.mean()

    loss = torch.tensor(0.0, device=device)
    count = 0

    for sample, r in zip(samples, rewards):
        t = torch.rand(sample.shape[0], device=device)
        eps = torch.randn_like(sample)
        t_exp = t.view(-1, 1, 1, 1)

        x_t = (1 - t_exp) * sample + t_exp * eps
        target_v = eps - sample
        pred_v = model(x_t, t, cond)
        fm_loss = F.mse_loss(pred_v, target_v)

        weight = abs(r - mean_r)
        if r > mean_r:
            loss = loss + weight * fm_loss
        else:
            loss = loss - weight * fm_loss
        count += 1

    loss = loss / count
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()`,
        hints: [
            "ODE 采样: x_new = x + v * (t_next - t_cur)，dt 是负值",
            "FM 训练对: x_t = (1-t)*x_0 + t*eps, target_v = eps - x_0",
            "正样本: 加 FM loss（拟合），负样本: 减 FM loss（远离）",
            "权重 weight = |r - mean_r|，偏离越大权重越大",
            "loss 初始化用 torch.tensor(0.0, device=device) 保持梯度"
        ],
        keyPoints: [
            "前向 FM loss 做对比训练",
            "正样本拟合，负样本远离",
            "无需轨迹存储和似然估计",
            "FM target: v = eps - x_0"
        ]
    },
    {
        id: 10,
        title: "MMDiT 手撕",
        category: "mmdit",
        difficulty: "hard",
        description: `## MMDiT (Multi-Modal Diffusion Transformer)

实现 SD3/FLUX 使用的 MMDiT 架构。

### 核心设计
- **双模态独立**: img 和 txt 各自有独立的 Norm + FFN
- **共享 Joint Attention**: 拼接 K/V，各自 Q
- **AdaLN-Zero**: 用时间步调制 (shift, scale, gate) x 2 = 6 个参数

### Joint Attention 流程
1. 分别计算 img 和 txt 的 Q, K, V
2. 拼接 K = [K_txt, K_img], V = [V_txt, V_img]
3. img 用 Q_img 查询拼接后的 K, V
4. txt 用 Q_txt 查询拼接后的 K, V

### AdaLN-Zero
\`modulate(x, shift, scale) = x * (1 + scale) + shift\`
每个子层有 gate 参数控制残差贡献`,
        template: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MMDiTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        # TODO

    def forward(self, img, txt, t_emb):
        # TODO


class MMDiT(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, num_layers=24, patch_size=2, in_ch=16):
        # TODO

    def forward(self, x, t, txt_emb):
        # TODO
`,
        reference: `class MMDiTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.norm1_img = nn.LayerNorm(d_model)
        self.norm1_txt = nn.LayerNorm(d_model)
        self.qkv_img = nn.Linear(d_model, d_model * 3)
        self.qkv_txt = nn.Linear(d_model, d_model * 3)
        self.proj_img = nn.Linear(d_model, d_model)
        self.proj_txt = nn.Linear(d_model, d_model)

        self.norm2_img = nn.LayerNorm(d_model)
        self.norm2_txt = nn.LayerNorm(d_model)
        self.ff_img = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.ff_txt = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

        self.adaln_img = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))
        self.adaln_txt = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))

    def forward(self, img, txt, t_emb):
        s1_i, c1_i, g1_i, s2_i, c2_i, g2_i = self.adaln_img(t_emb).chunk(6, dim=-1)
        s1_t, c1_t, g1_t, s2_t, c2_t, g2_t = self.adaln_txt(t_emb).chunk(6, dim=-1)

        def modulate(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        img_n = modulate(self.norm1_img(img), s1_i, c1_i)
        txt_n = modulate(self.norm1_txt(txt), s1_t, c1_t)

        B, N_i, _ = img_n.shape
        N_t = txt_n.shape[1]

        qi, ki, vi = [x.view(B, N_i, self.num_heads, self.d_k).transpose(1,2) for x in self.qkv_img(img_n).chunk(3, -1)]
        qt, kt, vt = [x.view(B, N_t, self.num_heads, self.d_k).transpose(1,2) for x in self.qkv_txt(txt_n).chunk(3, -1)]

        k_all = torch.cat([kt, ki], dim=2)
        v_all = torch.cat([vt, vi], dim=2)

        o_img = F.scaled_dot_product_attention(qi, k_all, v_all)
        o_img = self.proj_img(o_img.transpose(1,2).contiguous().view(B, N_i, -1))

        o_txt = F.scaled_dot_product_attention(qt, k_all, v_all)
        o_txt = self.proj_txt(o_txt.transpose(1,2).contiguous().view(B, N_t, -1))

        img = img + g1_i.unsqueeze(1) * o_img
        txt = txt + g1_t.unsqueeze(1) * o_txt

        img = img + g2_i.unsqueeze(1) * self.ff_img(modulate(self.norm2_img(img), s2_i, c2_i))
        txt = txt + g2_t.unsqueeze(1) * self.ff_txt(modulate(self.norm2_txt(txt), s2_t, c2_t))
        return img, txt


class MMDiT(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, num_layers=24, patch_size=2, in_ch=16):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_ch, d_model, patch_size, patch_size)
        self.txt_proj = nn.Linear(d_model, d_model)
        self.time_embed = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([MMDiTBlock(d_model, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model, patch_size**2 * in_ch)

    def forward(self, x, t, txt_emb):
        B, C, H, W = x.shape

        half = 128
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        t_emb = self.time_embed(torch.cat([torch.cos(t[:,None]*freqs), torch.sin(t[:,None]*freqs)], -1))

        img = self.patch_embed(x).flatten(2).transpose(1, 2)
        txt = self.txt_proj(txt_emb)

        for block in self.blocks:
            img, txt = block(img, txt, t_emb)

        img = self.final_proj(self.final_norm(img))
        p = self.patch_size
        h, w = H // p, W // p
        img = img.view(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return img`,
        hints: [
            "AdaLN-Zero 有 6 个参数: shift1, scale1, gate1, shift2, scale2, gate2",
            "modulate: x * (1 + scale) + shift，注意 unsqueeze(1) 广播序列维",
            "Joint Attention: K = cat([K_txt, K_img])，V 同理",
            "Gate 控制残差: img = img + gate * output",
            "Unpatchify: view(B,h,w,p,p,C).permute(0,5,1,3,2,4).reshape(B,C,H,W)"
        ],
        keyPoints: [
            "双模态独立 Norm+FFN，共享 Joint Attention",
            "AdaLN-Zero: modulate + gate",
            "Joint Attention: 拼接 K/V，各自 Q",
            "Patchify/Unpatchify 的维度变换"
        ]
    },
    {
        id: 11,
        title: "DPO 损失 (Direct Preference Optimization)",
        category: "dpo",
        difficulty: "medium",
        description: `## DPO 损失函数

实现 Direct Preference Optimization 的损失函数。

### 核心思想
DPO 将 RLHF 中的奖励建模和 PPO 优化合二为一，直接用偏好数据优化策略模型，无需训练 reward model。

### 核心公式
\`\`\`
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
\`\`\`

其中:
- \`π_θ\`: 当前策略模型
- \`π_ref\`: 参考模型 (冻结的 SFT 模型)
- \`y_w\`: preferred (chosen) response
- \`y_l\`: dispreferred (rejected) response
- \`β\`: 温度参数，控制偏离参考模型的程度

### 要求
1. 输入: policy 和 reference 模型对 chosen/rejected 的 log probabilities
2. 计算 log ratios (policy vs reference)
3. 计算 DPO 损失
4. 返回损失值以及 chosen/rejected 的 reward 用于监控`,
        template: `import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    # TODO
`,
        reference: `import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    chosen_log_ratios = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratios = policy_rejected_logps - reference_rejected_logps

    logits = beta * (chosen_log_ratios - rejected_log_ratios)

    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = beta * chosen_log_ratios.detach()
    rejected_rewards = beta * rejected_log_ratios.detach()

    return loss, chosen_rewards, rejected_rewards`,
        hints: [
            "log ratio = log π_θ(y|x) - log π_ref(y|x)",
            "DPO logits = β * (chosen_log_ratio - rejected_log_ratio)",
            "损失用 -logsigmoid(logits) 而不是手动算 BCE",
            "reward 用于监控，需要 detach() 断梯度"
        ],
        keyPoints: [
            "log ratio 的计算: policy_logps - reference_logps",
            "logsigmoid 比 log(sigmoid()) 数值更稳定",
            "β 参数控制 KL 惩罚强度",
            "chosen_rewards 和 rejected_rewards 需要 detach"
        ]
    },
    {
        id: 12,
        title: "Mixture of Experts (MoE) 路由",
        category: "moe",
        difficulty: "hard",
        description: `## Mixture of Experts 路由与前向

实现 MoE 层，包含 Top-K 门控路由和负载均衡辅助损失。

### 核心设计
- **门控网络 (Router)**: 线性层输出 logits，softmax 后 Top-K 选择专家
- **Top-K 路由**: 每个 token 选 K 个专家处理，按 softmax 权重加权
- **负载均衡损失 (Auxiliary Loss)**: 防止所有 token 涌向同一个专家

### 路由流程
\`\`\`
1. router_logits = x @ W_gate          # (B*L, num_experts)
2. router_probs = softmax(router_logits)
3. top_k_probs, top_k_indices = topk(router_probs, k)
4. top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 归一化
5. 分发 token 给对应专家，加权求和
\`\`\`

### 负载均衡辅助损失
\`\`\`
L_aux = num_experts * Σ(f_i · p_i)
f_i = 分配到专家 i 的 token 比例
p_i = router 对专家 i 的平均概率
\`\`\`

### 要求
- 实现 Router (Top-K 门控)
- 实现 MoELayer，包含多个 FFN 专家
- 计算辅助负载均衡损失`,
        template: `import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        # TODO

    def forward(self, x):
        # TODO


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        # TODO

    def forward(self, x):
        # TODO
`,
        reference: `import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = x_flat[token_indices]
                expert_output = expert(expert_input)
                weights_for_expert = top_k_probs[top_k_indices == i]
                output[token_indices] += weights_for_expert.unsqueeze(-1) * expert_output

        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f[i] = (top_k_indices == i).float().any(dim=-1).mean()
        p = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * p).sum()

        return output.view(B, L, D), aux_loss`,
        hints: [
            "gate 是一个 Linear(d_model, num_experts, bias=False)",
            "Top-K 后要对选中的概率重新归一化",
            "遍历每个专家，用 mask 找到分配给它的 token",
            "负载均衡损失: num_experts * sum(f_i * p_i)"
        ],
        keyPoints: [
            "Router softmax + Top-K + 归一化",
            "按专家遍历 token 分发和加权求和",
            "负载均衡辅助损失的 f_i 和 p_i 计算",
            "x_flat = x.view(-1, D) 展平 batch 和 seq"
        ]
    },
    {
        id: 13,
        title: "Beam Search 解码",
        category: "decoding",
        difficulty: "medium",
        description: `## Beam Search 解码

实现 Beam Search 解码算法。

### 核心思想
Beam Search 在每一步保留 beam_size 个最优候选序列，在搜索空间和计算效率之间取得平衡。

### 算法流程
\`\`\`
1. 初始化: beam = [(score=0, tokens=[BOS])] × beam_size
2. 每一步:
   a. 对每个候选序列，用模型预测下一个 token 的 log_probs
   b. 扩展: 每个候选 × vocab_size 个可能
   c. 从所有扩展中选 top beam_size 个 (按累积 log_prob)
   d. 如果某个候选生成了 EOS，移入完成队列
3. 返回得分最高的完成序列 (可选 length penalty)
\`\`\`

### Length Penalty
\`\`\`
score = log_prob / ((5 + length) / 6) ^ α
\`\`\`

### 要求
- 输入: 一个返回 log_probs 的模型函数、起始 token、beam_size
- 支持 EOS 终止
- 支持 max_length 限制
- 支持 length penalty`,
        template: `import torch
import torch.nn.functional as F


def beam_search(model_fn, start_token_id, beam_size=5, max_length=50, eos_token_id=2, length_penalty=0.6):
    # TODO
`,
        reference: `import torch
import torch.nn.functional as F


def beam_search(model_fn, start_token_id, beam_size=5, max_length=50, eos_token_id=2, length_penalty=0.6):
    device = next(iter([])) if False else torch.device('cpu')

    beams = [(torch.tensor([[start_token_id]]), 0.0)]
    completed = []

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == eos_token_id:
                completed.append((seq, score))
                continue

            logits = model_fn(seq)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)

            for i in range(beam_size):
                token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=-1)
                new_score = score + top_log_probs[0, i].item()
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    completed.extend(beams)

    def final_score(score, length):
        return score / ((5.0 + length) / 6.0) ** length_penalty

    completed.sort(key=lambda x: final_score(x[1], x[0].shape[1]), reverse=True)
    return completed[0][0]`,
        hints: [
            "beams 存的是 (序列 tensor, 累积 log_prob) 的列表",
            "每步只对每个 beam 取 top beam_size 个扩展即可，不需要全 vocab",
            "遇到 EOS 的候选移入 completed，不再扩展",
            "最后用 length_penalty 归一化分数来排序"
        ],
        keyPoints: [
            "累积 log_prob 而不是 prob（避免下溢）",
            "每步从所有候选中选 top beam_size 个",
            "EOS 检测和完成队列管理",
            "Length penalty: score / ((5+len)/6)^α"
        ]
    },
    {
        id: 14,
        title: "GQA / MQA (Grouped-Query Attention)",
        category: "gqa",
        difficulty: "medium",
        description: `## Grouped-Query Attention

实现 GQA (Grouped-Query Attention)，同时兼容 MHA 和 MQA。

### 核心设计
- **MHA**: num_kv_heads == num_heads，每个 head 有独立的 K, V
- **MQA**: num_kv_heads == 1，所有 head 共享一组 K, V
- **GQA**: num_kv_heads 介于两者之间，若干 head 共享一组 K, V

### 关键点
\`\`\`
num_heads = 32, num_kv_heads = 8  →  每 4 个 Q head 共享 1 组 KV
n_rep = num_heads // num_kv_heads  →  每组 KV 复制 n_rep 次
\`\`\`

### KV 复制方式
\`\`\`python
# (B, num_kv_heads, L, d_k) -> (B, num_heads, L, d_k)
# expand: 先 unsqueeze 插入维度，再 expand 复制
K = K.unsqueeze(2).expand(B, num_kv_heads, n_rep, L, d_k).reshape(B, num_heads, L, d_k)
\`\`\`

### 要求
- 实现统一的 GQA 模块
- 参数: d_model, num_heads, num_kv_heads
- 当 num_kv_heads == num_heads 时退化为 MHA
- 当 num_kv_heads == 1 时退化为 MQA
- K, V 使用较少的 head，通过 repeat/expand 匹配 Q 的 head 数`,
        template: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        # TODO

    def forward(self, x, mask=None):
        # TODO
`,
        reference: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_kv_heads, self.d_k).transpose(1, 2)

        K = K.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.d_k).reshape(B, self.num_heads, L, self.d_k)
        V = V.unsqueeze(2).expand(B, self.num_kv_heads, self.n_rep, L, self.d_k).reshape(B, self.num_heads, L, self.d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)`,
        hints: [
            "W_k 和 W_v 的输出维度是 num_kv_heads * d_k，而非 num_heads * d_k",
            "n_rep = num_heads // num_kv_heads 是每组 KV 需要复制的次数",
            "KV 复制: unsqueeze(2) -> expand -> reshape 到 num_heads 维",
            "Q 的分头和标准 MHA 完全一样"
        ],
        keyPoints: [
            "W_q vs W_k/W_v 输出维度不同",
            "KV 的 expand/repeat 到 num_heads 的维度操作",
            "assert num_heads % num_kv_heads == 0",
            "退化: num_kv_heads==num_heads → MHA, ==1 → MQA"
        ]
    }
];
