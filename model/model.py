from transformers import PretrainedConfig
import torch
import torch.nn as nn
import math
from typing import Optional

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            dim: int = 512,
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
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.dim = dim
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
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
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
        #.type_as(x)确保类型不变
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# RoPE+YaRN
def precomput_freq_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE (Rotary Positional Embeddings) 的频率项。
    包含 YaRN (Yet another RoPE for Nontrivial context) 逻辑，用于支持更长的上下文窗口。
    """
    # 1. 计算标准 RoPE 频率
    # 公式: theta_i = base^(-2i/dim)
    # 频率从高到低排列: freqs[0] 是最高频, freqs[-1] 是最低频
    freqs = rope_base ** (-torch.arange(0, dim, 2)[:dim // 2].float() / dim)

    # 2. YaRN 上下文扩展逻辑 (如果配置了 rope_scaling)
    if rope_scaling is not None:
        # 提取参数:
        # original_max_position_embeddings: 原始训练时的上下文长度 (如 2048)
        # factor: 扩展倍数 (如 4, 目标 8k)
        # beta_fast/beta_slow: 混合插值和外推的阈值参数
        original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 32)
        beta_slow = rope_scaling.get("beta_slow", 1)

        # 3. 计算修正维度 (Correction Dimension)
        # 波长 (Wavelength) = 2 * pi / freq
        # 我们寻找波长超过原始上下文长度的频率分量。
        # 高频分量 (短波长) 不受位置索引溢出影响，通常不需要处理。
        # 低频分量 (长波长) 需要通过 YaRN 插值来适应更长的上下文。
        def get_wavelength(freq):
            return 2 * math.pi / freq

        # 找到第一个波长大于原始长度的维度索引
        # 这个索引区分了 "高频不处理区" 和 "低频需插值区"
        corr_dim = next(
            (i for i in range(dim // 2) if get_wavelength(freqs[i]) > original_max_position_embeddings),
            dim // 2
        )
        #计算power
        power=torch.arange(0,dim//2,device=freqs.device).float/(max(dim//2-1,1))
            

        #计算scale
        scale=torch.where(
            torch.arange(0,dim//2,device=freqs.device)<corr_dim,
            (beta_fast*factor-beta_fast+1)/(beta_fast*factor),
            1.0/factor
        )

        #将scale应用到freqs上
        freqs=freqs*scale

    # 生成位置索引和旋转矩阵
    # t: 位置索引序列 (0, 1, 2, ..., end-1)
    # freqs: 频率向量 (shape: dim/2)
    # torch.outer(t, freqs): 生成位置-频率矩阵 (shape: end, dim/2)
    # 每一行代表一个位置，每一列代表一个频率分量
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() #[end,dim//2]

    # 5. 计算旋转矩阵 (Rotation Matrix)
    # freqs_cos: 旋转矩阵的余弦部分 (shape: end, dim)
    # freqs_sin: 旋转矩阵的正弦部分 (shape: end, dim)
    # torch.cat: 将 cos 和 sin 拼接，形成完整的旋转矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos,freqs_sin 

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
    #half_ratate定义的旋转位置编码效果和分组的效果在数学上一致,且在矩阵运算过程中更高效
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
    #维度扩展是为了便于后续计算
    xq_embed = (xq * freqs_cos.unsqueeze(unsqueeze_dim)) + (rotate_half(xq) * freqs_sin.unsqueeze(unsqueeze_dim))
    xk_embed = (xk * freqs_cos.unsqueeze(unsqueeze_dim)) + (rotate_half(xk) * freqs_sin.unsqueeze(unsqueeze_dim))

    return xq_embed, xk_embed