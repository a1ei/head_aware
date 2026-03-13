import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import FlashAttentionKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
from typing import Callable

import math
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class self_Qwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, model):
        super().__init__()
        self.layer_type = model.layer_type
        self.config = model.config
        self.layer_idx = model.layer_idx
        self.head_dim = model.head_dim
        self.num_key_value_groups = model.num_key_value_groups
        self.scaling = model.scaling
        self.attention_dropout = model.attention_dropout
        self.is_causal = True

        self.q_proj = model.q_proj
        self.k_proj = model.k_proj
        self.v_proj = model.v_proj
        self.o_proj = model.o_proj
        self.q_norm = model.q_norm
        self.k_norm = model.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output_2,attn_weight_2 = self.manual_attention_with_weights(query_states, key_states, value_states, attention_mask, scaling=self.scaling, dropout_p=0.0)

        self.plot_multimodal_attention_32_heads(attn_weight_2, self.vision_mask,None)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def manual_attention_with_weights(self, query, key, value, attention_mask=None, scaling=None, dropout_p=0.0):
        """
        手动实现 Attention，以获取 attn_weights。
        注意：这会消耗大量显存，不要在超长上下文（如 > 8k）时使用！
        """
        if hasattr(self, "num_key_value_groups"):

            # if not use_gqa_in_sdpa(attention_mask, key):

            key = repeat_kv(key, self.num_key_value_groups)

            value = repeat_kv(value, self.num_key_value_groups)
        # 1. 确定缩放因子
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.size(-1))

        # 2. 计算原始的 Q * K^T
        # query 形状: (batch, q_heads, q_len, head_dim)
        # key 形状:   (batch, kv_heads, kv_len, head_dim)
        # 注意：如果使用了 GQA 且没有 repeat_kv，这里直接 matmul 会报错尺寸不匹配！
        # 所以如果你要手动计算，必须确保 Q 和 K 的头数已经对齐（即经过了 repeat_kv）。
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling

        # 3. 加上 Attention Mask (因果掩码或 Padding 掩码)
        if attention_mask is not None:
            # 获取当前数据类型能表示的极小负数
            min_dtype = -1e9

            # 找到所有为 False 的位置，把 attn_weights 里的值替换成极小负数
            attn_weights = attn_weights.masked_fill(attention_mask == False, min_dtype)

        # 4. Softmax 归一化，得到真正的注意力分数！
        # 这里的 attn_weights 就是你想提取的东西
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if attention_mask is not None:
            # 直接把本来就该被屏蔽的地方，在概率分布上强行掐死为 0.0
            attn_weights = attn_weights.masked_fill(attention_mask == False, 0.0)

        # 5. Dropout (如果在训练阶段)
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # 6. 乘以 Value 得到输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


    def plot_multimodal_attention_32_heads(self,attn_weights, vision_mask, text_mask):
        """
        绘制 1x32x624x624 的多头注意力热力图，并划分 Vision 和 Text 区域。

        参数:
        attn_weights: Tensor, shape (1, 32, 624, 624)
        vision_mask: Tensor, shape (1, 624) 或 (624,), 布尔值或 0/1, True表示视觉Token
        text_mask: Tensor, shape (1, 624) 或 (624,), 布尔值或 0/1, True表示文本Token
        """
        # 1. 数据转移到 CPU 并转换为 Numpy
        # 去掉 batch 维度，变成 (32, 624, 624)
        attn = attn_weights.squeeze(0).abs().cpu().detach().float().numpy()
        v_mask = vision_mask.squeeze().cpu().detach().numpy().astype(bool)

        # 2. 寻找 Vision 和 Text 的边界 (Transitions)
        # 通过比较相邻元素是否不同，找出模态切换的索引位置
        transitions = np.where(v_mask[:-1] != v_mask[1:])[0] + 0.5

        # 3. 创建 4 行 8 列的画布 (适合 32 个头宽屏显示)
        fig, axes = plt.subplots(4, 8, figsize=(28, 14))
        fig.suptitle("Multi-Head Attention Heatmaps (32 Heads)\nCyan Lines Separate Vision & Text Tokens",
                    fontsize=24, fontweight='bold', color='white')

        # 设置黑色背景，让“亮色”更加突出
        fig.patch.set_facecolor('black')

        for i, ax in enumerate(axes.flat):
            # 使用 99% 分位数作为最大亮度，防止个别 token 权重过大导致其余部分全黑
            vmax = np.percentile(attn[i], 99.5)

            # 画热力图: magma 颜色图 (数值大=黄色/白色亮光，数值小=黑色/暗紫)
            im = ax.imshow(attn[i], cmap='magma', aspect='auto', vmin=0, vmax=vmax)

            # 4. 画边界十字线，划分四个象限
            for t in transitions:
                # 垂直线 (划分 Key 序列的 V 和 T)
                ax.axvline(x=t, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)
                # 水平线 (划分 Query 序列的 V 和 T)
                ax.axhline(y=t, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)

            # 设置标题和样式
            ax.set_title(f"Head {i}", fontsize=14, color='white', pad=5)

            # 关掉刻度数字，保持画面干净
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # 5. 在第一个子图上标注出 V (Vision) 和 T (Text) 的大体位置，作为图例
            if i == 0:
                # 简单假设: 第一段是 Text 还是 Vision？
                is_first_vision = v_mask[0]
                label_1 = "V" #if is_first_vision else "T"
                label_2 = "T" #if is_first_vision else "V"

                # 找到第一个区域的中点和第二个区域的中点用于打标签
                mid_1 = transitions[0] / 2 if len(transitions) > 0 else 312
                mid_2 = (transitions[0] + 624) / 2 if len(transitions) > 0 else 312

                # X 轴标签 (代表 Key)
                ax.text(mid_1, 650, label_1, color='cyan', fontsize=14, ha='center', fontweight='bold')
                if len(transitions) > 0:
                    ax.text(mid_2, 650, label_2, color='cyan', fontsize=14, ha='center', fontweight='bold')

                # Y 轴标签 (代表 Query)
                ax.text(-30, mid_1, label_1, color='cyan', fontsize=14, va='center', fontweight='bold')
                if len(transitions) > 0:
                    ax.text(-30, mid_2, label_2, color='cyan', fontsize=14, va='center', fontweight='bold')

        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        import os
        os.makedirs(f"z_attention_comparison/{self.data_num}", exist_ok=True)

        output_path =  get_next_filename(f"z_attention_comparison/{self.data_num}")
        plt.savefig(output_path)


def get_next_filename(folder, prefix="exp", extension=".png"):
    import os
    import re
    """
    自动获取下一个可用的文件名 (例如: exp1.png -> exp2.png)
    """
    # 1. 确保文件夹存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 2. 获取文件夹中所有匹配前缀和后缀的文件
    # 构造正则表达式: 例如 ^exp(\d+)\.png$
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+){re.escape(extension)}$')

    existing_indices = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            # 提取数字部分并转为整数
            num = int(match.group(1))
            existing_indices.append(num)

    # 3. 计算下一个编号
    if existing_indices:
        next_num = max(existing_indices) + 1
    else:
        next_num = 0 # 如果文件夹是空的，从 1 开始

    # 4. 返回完整路径
    return os.path.join(folder, f"{prefix}{next_num}{extension}")