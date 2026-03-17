import math
import os  #change here
from typing import Callable

import matplotlib.pyplot as plt  #change here
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Patch  #change here

from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import FlashAttentionKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


ATTN_OUTPUT_DIR = "/home/liuzhilei/python_project/head_aware/num_attention"  #change here


def compute_cross_head_scores(attn_weights, vision_mask, text_mask, token_mask=None):
    attn_weights = attn_weights.detach().to(torch.float32)
    vision_mask = vision_mask.to(attn_weights.device).bool()
    text_mask = text_mask.to(attn_weights.device).bool()

    if token_mask is None:
        token_mask = torch.ones_like(vision_mask, dtype=torch.bool, device=attn_weights.device)
    else:
        token_mask = token_mask.to(attn_weights.device).bool()

    cross_scores = []
    for batch_idx in range(attn_weights.shape[0]):
        valid_mask = token_mask[batch_idx]
        cur_vision_mask = vision_mask[batch_idx] & valid_mask
        cur_text_mask = text_mask[batch_idx] & valid_mask

        if cur_vision_mask.sum() == 0 or cur_text_mask.sum() == 0:
            continue

        cur_attn = attn_weights[batch_idx]
        vision_to_text = cur_attn[:, cur_vision_mask, :][:, :, cur_text_mask].sum(dim=-1).mean(dim=-1)
        text_to_vision = cur_attn[:, cur_text_mask, :][:, :, cur_vision_mask].sum(dim=-1).mean(dim=-1)
        cross_scores.append( (vision_to_text + text_to_vision))

    if not cross_scores:
        return None

    return torch.stack(cross_scores).mean(dim=0)


class HeadCrossScoreRecorder:
    def __init__(self):
        self.storage = {}

    def update(self, layer_idx, cross_scores):
        cross_scores = cross_scores.detach().cpu().float()
        if layer_idx not in self.storage:
            self.storage[layer_idx] = {
                "cross_scores": torch.zeros_like(cross_scores),
                "count": 0,
            }
        self.storage[layer_idx]["cross_scores"] += cross_scores
        self.storage[layer_idx]["count"] += 1

    def export(self):
        results = {}
        for layer_idx, item in self.storage.items():
            results[layer_idx] = {
                "cross_scores": item["cross_scores"] / max(item["count"], 1)
            }
        return results


class CrossScoreQwen3VLTextAttention(nn.Module):
    def __init__(self, module, recorder):
        super().__init__()
        self.layer_type = module.layer_type
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = True

        self.q_proj = module.q_proj
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.o_proj = module.o_proj
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm

        self.recorder = recorder
        self.vision_mask = None
        self.text_mask = None
        self.token_mask = None
        self.data_index = None  #change here
        self.save_attn_output = False  #change here

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

        with torch.no_grad():
            _, manual_attn_weights = self.manual_attention_with_weights(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout_p=0.0,
            )
            self.record_cross_scores(manual_attn_weights)
            # self.save_attn_output_histograms(attn_output)  #change here

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def record_cross_scores(self, attn_weights):
        if self.recorder is None or self.vision_mask is None or self.text_mask is None:
            return
        cross_scores = compute_cross_head_scores(attn_weights, self.vision_mask, self.text_mask, self.token_mask)
        if cross_scores is None:
            return
        self.recorder.update(self.layer_idx, cross_scores)

    def save_attn_output_histograms(self, attn_output):
        if not self.save_attn_output or self.data_index is None:  #change here
            return

        layer_dir = os.path.join(ATTN_OUTPUT_DIR, str(self.data_index), f"layer_{self.layer_idx}")  #change here
        first_output_path = os.path.join(layer_dir, "part_0.png")  #change here
        if os.path.exists(first_output_path):  #change here
            return

        os.makedirs(layer_dir, exist_ok=True)  #change here
        sample_output = attn_output[0].detach().float().cpu().permute(1, 0, 2)  #change here
        num_heads = sample_output.shape[0]
        seq_len = sample_output.shape[1]  #change here
        head_dim = sample_output.shape[2]  #change here
        heads_per_figure = 8  #change here
        num_cols = 4  #change here
        num_rows = 2  #change here
        token_stride = max(1, seq_len // 256)  #change here
        channel_stride = max(1, head_dim // 96)  #change here

        token_indices = torch.arange(0, seq_len, token_stride)  #change here
        channel_indices = torch.arange(0, head_dim, channel_stride)  #change here
        token_grid, channel_grid = torch.meshgrid(token_indices, channel_indices, indexing="ij")  #change here

        token_mask = torch.ones(seq_len, dtype=torch.bool)  #change here
        if self.token_mask is not None:
            token_mask = self.token_mask[0].detach().cpu().bool()  #change here
        vision_mask = torch.zeros(seq_len, dtype=torch.bool)  #change here
        if self.vision_mask is not None:
            vision_mask = self.vision_mask[0].detach().cpu().bool()  #change here
        text_mask = (~vision_mask) & token_mask  #change here
        masked_mask = ~token_mask  #change here

        token_colors = torch.zeros(seq_len, 4, dtype=torch.float32)  #change here
        token_colors[vision_mask] = torch.tensor([0.12, 0.47, 0.71, 0.85])  #change here
        token_colors[text_mask] = torch.tensor([1.00, 0.50, 0.05, 0.85])  #change here
        token_colors[masked_mask] = torch.tensor([0.65, 0.65, 0.65, 0.30])  #change here

        global_abs_max = float(sample_output.abs().max())  #change here

        for part_idx, head_start in enumerate(range(0, num_heads, heads_per_figure)):  #change here
            fig = plt.figure(figsize=(26, 12), dpi=220)  #change here
            axes = []
            head_stop = min(head_start + heads_per_figure, num_heads)  #change here

            for slot_idx, head_idx in enumerate(range(head_start, head_stop)):  #change here
                ax = fig.add_subplot(num_rows, num_cols, slot_idx + 1, projection="3d")  #change here
                axes.append(ax)
                head_matrix = sample_output[head_idx]  #change here
                head_tensor = head_matrix.reshape(-1)  #change here
                head_min = float(head_tensor.min())  #change here
                head_max = float(head_tensor.max())  #change here
                head_mean = float(head_tensor.mean())  #change here
                head_std = float(head_tensor.std())  #change here

                sampled_values = head_matrix[token_indices][:, channel_indices].numpy()  #change here
                sampled_colors = token_colors[token_indices][:, None, :].expand(-1, channel_indices.numel(), -1).numpy()  #change here

                ax.plot_surface(  #change here
                    token_grid.numpy(),
                    channel_grid.numpy(),
                    sampled_values,
                    facecolors=sampled_colors,
                    linewidth=0,
                    antialiased=False,
                    shade=False,
                )
                ax.set_title(
                    f"head_{head_idx}\nmin={head_min:.2e} max={head_max:.2e}\nmean={head_mean:.2e} std={head_std:.2e}"
                )  #change here
                ax.set_xlabel("token", fontsize=7)  #change here
                ax.set_ylabel("channel", fontsize=7)  #change here
                ax.set_zlabel("value", fontsize=7)  #change here
                ax.set_zlim(-global_abs_max, global_abs_max)  #change here
                ax.tick_params(labelsize=7)  #change here
                ax.view_init(elev=28, azim=-58)  #change here
                ax.set_box_aspect((2.8, 1.2, 1.4))  #change here

            for slot_idx in range(head_stop - head_start, heads_per_figure):  #change here
                ax = fig.add_subplot(num_rows, num_cols, slot_idx + 1, projection="3d")  #change here
                ax.axis("off")

            fig.legend(  #change here
                handles=[
                    Patch(facecolor=(0.12, 0.47, 0.71, 0.85), label="vision token"),
                    Patch(facecolor=(1.00, 0.50, 0.05, 0.85), label="text token"),
                    Patch(facecolor=(0.65, 0.65, 0.65, 0.30), label="masked token"),
                ],
                loc="upper right",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(layer_dir, f"part_{part_idx}.png"), bbox_inches="tight", dpi=220)  #change here
            plt.close(fig)

    def manual_attention_with_weights(self, query, key, value, attention_mask=None, scaling=None, dropout_p=0.0):
        if hasattr(self, "num_key_value_groups"):
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)

        if scaling is None:
            scaling = 1.0 / math.sqrt(query.size(-1))

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling

        bool_mask = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                bool_mask = attention_mask
                attn_weights = attn_weights.masked_fill(~bool_mask, -1e9)
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if bool_mask is not None:
            attn_weights = attn_weights.masked_fill(~bool_mask, 0.0)

        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def patch_qwen3vl_attention_modules(model):
    recorder = HeadCrossScoreRecorder()
    replaced_modules = []

    language_model = getattr(model.model.model, "language_model", None)
    if language_model is None:
        return recorder, replaced_modules

    for layer in language_model.layers:
        if not hasattr(layer, "self_attn"):
            continue
        original_attn = layer.self_attn
        layer.self_attn = CrossScoreQwen3VLTextAttention(original_attn, recorder)
        replaced_modules.append((layer, original_attn))

    return recorder, replaced_modules


def restore_qwen3vl_attention_modules(replaced_modules):
    for layer, original_attn in replaced_modules:
        layer.self_attn = original_attn
