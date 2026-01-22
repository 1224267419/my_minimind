from transformers import PretrainedConfig
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# RMSNorm implementation
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 初始化参数权重, 使其能被优化
        self.weight = nn.Parameter(torch.ones(dim))

    def _layerNorm(self, x):
        # 计算均值和方差
        mean = x.mean(-1, keepdim=True)
        ## var 计算的是方差： E[(x - mean)^2]
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)

    def _norm(self, x):
        # 计算均方值 (Mean Square) E[x^2]
        # 使用 rsqrt (Reciprocal Square Root) 进行归一化: x * (1 / sqrt(mean(x^2) + eps))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 为了数值稳定性，通常在 float32 下计算 Norm
        # .type_as(x)确保类型不变
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# RoPE+YaRN
def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    预计算 RoPE (Rotary Positional Embeddings) 的频率项。
    包含 YaRN (Yet another RoPE for Nontrivial context) 逻辑，用于支持更长的上下文窗口。
    """
    # 1. 计算标准 RoPE 频率
    # 公式: theta_i = base^(-2i/dim) i是从0,d/2-1的index
    # 频率从高到低排列: freqs[0] 是最高频, freqs[-1] 是最低频
    # [:dim // 2] 避免边缘情况导致的形状不匹配
    freqs, attn_factor = rope_base ** (
        -torch.arange(0, dim, 2)[: dim // 2].float() / dim
    )
    # attn_factor :调节 Softmax 的“温度” , 修正处理比训练长度更长的序列时出现的注意力分布偏移问题(Entropy Shift)
    # 2. YaRN 上下文扩展逻辑 (如果配置了 rope_scaling)
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp ,使用llama版的外推,能保留更多高频信息
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(
                math.ceil(inv_dim(beta_slow)), dim // 2 - 1
            )
            ##
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置索引和旋转矩阵
    # t: 位置索引序列 (0, 1, 2, ..., end-1)
    # freqs: 频率向量 (shape: dim/2)
    # torch.outer(t, freqs): 生成位置-频率矩阵 (shape: end, dim//2)
    # theta[m][i]=m*theta[i]
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 返回的 freqs_cos 和 freqs_sin 是两个形状为 (max_seq_len, head_dim) 的巨大张量 , 看下面apply_rotary_pos_emb的实现
    # 输入了第 100 个 token，模型就直接取第 100 行，与当前的 Query/Key 向量做计算
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(xq, xk, freqs_cos, freqs_sin, unsqueeze_dim=1):
    """
    应用 RoPE (Rotary Positional Embeddings) 旋转位置编码。
    将 Query 和 Key 向量在复平面上进行旋转，注入绝对位置信息，同时自然保留相对位置信息。

    Args:
        xq (torch.Tensor): Query 向量 [batch_size, seq_len, num_heads, head_dim]
        xk (torch.Tensor): Key 向量 [batch_size, seq_len, num_heads, head_dim]
        freqs_cos (torch.Tensor): 预计算的 Cos 值 [seq_len, head_dim] (或 [seq_len, 1, head_dim])
        freqs_sin (torch.Tensor): 预计算的 Sin 值 [seq_len, head_dim]
        unsqueeze_dim (int): 需要扩展维度的位置，默认 1 (对应 num_heads 维度，用于广播)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 应用 RoPE 后的 q 和 k
    """

    # 定义旋转辅助函数
    # 逻辑: 将向量 [x1, x2] 变换为 [-x2, x1]
    # 几何意义: 相当于在复平面上旋转 90 度 (乘以 i)
    # 配合正弦余弦公式: (x1 + i*x2) * (cos + i*sin) 的实部和虚部计算需要此变换
    # half_ratate定义的旋转位置编码效果和分组的效果在数学上一致,且在矩阵运算过程中更高效
    def rotate_half(x):
        # 将最后一维切分为两半: x1, x2
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        # 拼接为 [-x2, x1]
        return torch.cat((-x2, x1), dim=-1)

    # 调整频率 Tensor 的维度以便进行广播运算
    # 假设输入 xq 为 [batch, seq, head, dim]
    # freqs_cos 原本为 [seq, dim] -> unsqueeze(1) -> [seq, 1, dim]
    # 这样可以自动广播到 [batch, seq, head, dim]
    freqs_cos = freqs_cos.unsqueeze(unsqueeze_dim)
    freqs_sin = freqs_sin.unsqueeze(unsqueeze_dim)

    # 应用 Euler 公式展开后的旋转逻辑:
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # 展开来看:
    # real = x1 * cos - x2 * sin
    # imag = x2 * cos + x1 * sin
    # 维度扩展是为了便于后续计算
    xq_embed = (xq * freqs_cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(xq) * freqs_sin.unsqueeze(unsqueeze_dim)
    )
    xk_embed = (xk * freqs_cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(xk) * freqs_sin.unsqueeze(unsqueeze_dim)
    )

    return xq_embed, xk_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复n次,用于实现GQA
    """
    bs, slen, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expend(bs, slen, kv_heads, n_rep, head_dim)
        .reshape(bs, slen, kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):

    def __init__(self, args: MiniMindConfig) -> torch.Tensor:
        super().__init__()
        self.args = args
        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )

        assert (args.num_attention_heads % self.num_key_value_heads) == 0
        "num_attention_heads must be divisible by num_key_value_heads"
        # 这个其实是q的head数
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的dim为总dim//num_head
        self.head_dim = args.hidden_size // self.n_local_heads

        # QKV投影层
        self.q_proj = nn.Linear(
            args.hidden_size, self.head_dim * self.n_local_heads, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.head_dim * self.n_local_kv_heads, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.head_dim * self.n_local_kv_heads, bias=False
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.n_local_heads, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # 含义： 检查当前的 PyTorch 库中是否存在 scaled_dot_product_attention 这个函数。
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attn
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        # x->q,k,v
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # q,k,v->(bs,seq_len,num_heads,head_dim) 拆分成多个head
        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        cos, sin = position_embedding

        # 对q, k应用旋转位置编码
        # 仅对前面token的应用旋转位置编码,后面部分会被mask掉
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        # 缓存kv    cache
        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            # [bs,seq_len,num_heads,head_dim] -> [bs,num_heads,seq_len,head_dim]
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 对k, v进行扩展
        xk, xv = repeat_kv(xk, self.n_rep), repeat_kv(xv, self.n_rep)
        # 使用 flash attention
        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(xq, xk, xv)
        # 手动计算attention
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(
                0
            )  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(
            bs, seq_len, self.head_dim * self.n_local_heads
        )
        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 2.66这个倍数使得矩阵增加一个(旁路)且参数量不变
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到 64 的倍数,提高GPU运算效率
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        # 使用指定的激活函数,在transformer包里面,不找了
        self.act_fn = ACT2FN[config.hidden_act]
        # self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class AttentionBlock(nn.Module):
    def __init__(self, config: MiniMindConfig, layer_id: int):
        super().__init__()
        # 初始化参数时要提供,使得参数方差为1(因为有residual connection)
        self.layer_id = layer_id

        self.attn = Attention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, x, position_embedding, past_kv=None, use_cache=False, attention_mask=None
    ):
        res = x
        x, past_kv = self.attn(
            self.input_layernorm(x),
            position_embedding,
            past_kv,
            use_cache,
            attention_mask,
        )

        res = x + res
        x = res

        x = self.mlp(self.post_attention_layernorm(x)) + res
        return x, past_kv


class MiniMindBlock(nn.Module):
    # GQA和FFN都要用residual connection
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    # embedding->attention->norm

    """

    def __init__(
        self,
        config: MiniMindConfig,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            self.layers.append(MiniMindBlock(layer_id, config))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 输出的linear层
        self.linear_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        # RoPE计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_base,
            rope_scaling=config.rope_scaling,
        )
        # 注册缓冲区, 这些值不参与优化,但是会保存在模型中
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None  # 处理huggingface的兼容问题

        # 为未计算的kv_cache添加None
        past_key_values = past_key_values or [None] * len(self.layers)

        # embedding
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        # 参与计算的部分添加位置编码
        position_embeddings = self.dropout(
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    # 采用transformer的标准模块,简化输出方式
    # PreTrainedModel:提供模型管理标准
    # GenerationMixin:提供生成pipeline
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # 标准做法,将model和lm_head分开,
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 输入层和输出层权重共享 , 因为是做embedding 和 反向embedding , 做矩阵转置即可
        self.lm_head.weight = self.model.embed_tokens.weight

        # hf封装的模型输出
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask, past_key_values, use_cache, attention_mask
        )
        # 构造切片索引，用于决定保留 hidden_states 中哪些 token 的位置来计算 logits。
        # 如果 logits_to_keep 是 整数 → 创建 slice(-logits_to_keep, None)
        # 如果 logits_to_keep 是 其他类型（如 Tensor）→ 直接使用它作为索引
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)  # 判断是否为int,如果是tensor,则直接使用
            else logits_to_keep
        )
        # 只解码hidden_states 要求的那个元素,即 slice_indices
        # 例如logits_to_keep=1,则只解码hidden_states的最后一个元素 ,
        # 例如logits_to_keep=0,则只解码hidden_states的所有元素
        # 例如logits_to_keep=Tensor([1,3,10]),则只解码hidden_states的第1,3,10个元素
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = self.OUT(
            logits=logits, past_key_values=past_key_values, hidden_states=hidden_states
        )
        return output
