import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_submodules
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from qmllm.utils.search import get_hidden_states, get_op_by_name, get_op_name


OPTIM_STEPS = 20
OPTIM_LR = 5e-3
OPTIM_MAX_SAMPLES = 8
Mini_batvh = 4


def strip_dispatch_hooks(module):
    if any(hasattr(submodule, "_hf_hook") for submodule in module.modules()):
        remove_hook_from_submodules(module)
    return module


class ScaledSTEQuantLinear(nn.Module):
    def __init__(self, linear, w_bit, a_bit, q_config, log_scale=None):
        super().__init__()
        self.linear = linear
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.q_config = q_config
        self.log_scale = log_scale

    def forward(self, x):
        weight = self.linear.weight
        x = x.to(weight.dtype)
        if self.log_scale is not None:
            scale = torch.exp(self.log_scale).clamp(min=1e-4, max=1e4)
            x = x / scale.view(*([1] * (x.dim() - 1)), -1)
            weight = weight * scale.view(1, -1)

        if self.a_bit is not None and self.a_bit < 16:
            x = fake_quant_activation_ste(x, self.a_bit)
        if self.w_bit is not None and self.w_bit < 16:
            weight = fake_quant_weight_ste(weight, self.w_bit, self.q_config)

        return F.linear(x, weight, self.linear.bias)


def fake_quant_weight_ste(weight, n_bits, q_config):
    return fake_quant_tensor_ste(
        weight,
        n_bits=n_bits,
        zero_point=q_config.get("zero_point", True),
        q_group_size=q_config.get("q_group_size", -1),
        per_tensor=False,
    )


def fake_quant_activation_ste(x, n_bits):
    return fake_quant_tensor_ste(
        x,
        n_bits=n_bits,
        zero_point=False,
        q_group_size=-1,
        per_tensor=False,
    )


def fake_quant_tensor_ste(tensor, n_bits, zero_point=True, q_group_size=-1, per_tensor=False):
    if n_bits is None or n_bits >= 16:
        return tensor

    original_shape = tensor.shape
    quant_tensor = tensor
    if q_group_size > 0:
        quant_tensor = quant_tensor.reshape(-1, q_group_size)
    elif per_tensor:
        quant_tensor = quant_tensor.reshape(1, -1)
    else:
        quant_tensor = quant_tensor.reshape(-1, original_shape[-1])

    if zero_point:
        max_val = quant_tensor.amax(dim=1, keepdim=True)
        min_val = quant_tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = quant_tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    dequant = (torch.clamp(torch.round(quant_tensor / scales) + zeros, min_int, max_int) - zeros) * scales
    dequant = dequant.reshape(original_shape)
    return tensor + (dequant - tensor).detach()


def reconstruction_loss(target, pred, loss_mode):
    diff = target - pred
    if loss_mode == "mse":
        return diff.float().pow(2).mean()
    return diff.float().abs().mean()


def weighted_head_loss(target, pred, head_scores, head_dim, loss_mode):
    if head_scores is None:
        return reconstruction_loss(target, pred, loss_mode)

    num_heads = head_scores.numel()
    if target.shape[-1] != num_heads * head_dim:
        return reconstruction_loss(target, pred, loss_mode)

    target = target.reshape(-1, num_heads, head_dim)
    pred = pred.reshape(-1, num_heads, head_dim)
    diff = target - pred
    if loss_mode == "mse":
        diff = diff.float().pow(2)
    else:
        diff = diff.float().abs()

    weights = head_scores.to(diff.device).view(1, num_heads, 1) / head_scores[0].to(diff.device)
    return (diff * weights).mean()


def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)
    # ln.weight.div_(scales)
    with torch.no_grad():
        ori_dtype = ln.weight.dtype
        ln.weight.data = (ln.weight.data / scales).to(ori_dtype)

    if hasattr(ln, "bias") and ln.bias is not None:
        with torch.no_grad():
            ln.bias.data = ln.bias.data / (scales).to(ori_dtype)

    for fc in fcs:
        # fc.weight.mul_(scales.view(1, -1))
        with torch.no_grad():
            fc.weight.data = (fc.weight.data * scales.view(1, -1)).to(ori_dtype)


def apply_scale(module, scales_list):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()

        if isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ in {"Qwen2RMSNorm", "Qwen3VLTextRMSNorm"}:
            scale_ln_fcs(prev_op, layers, scales.cuda())
        else:
            raise NotImplementedError(f"Unsupported previous op for head-aware scaling: {type(prev_op)}")

        # prev_op.cpu()
        # for layer in layers:
        #     layer.cpu()
        scales.cpu()


def build_scale_list(module, attn_scale, mlp_scale):
    return [
        (
            get_op_name(module, module.input_layernorm),
            (
                get_op_name(module, module.self_attn.q_proj),
                get_op_name(module, module.self_attn.k_proj),
                get_op_name(module, module.self_attn.v_proj),
            ),
            attn_scale.detach().cpu(),
        ),
        (
            get_op_name(module, module.post_attention_layernorm),
            (
                get_op_name(module, module.mlp.gate_proj),
                get_op_name(module, module.mlp.up_proj),
            ),
            mlp_scale.detach().cpu(),
        ),
    ]


def capture_qwen_block_tensors(layer, layer_input, layer_kwargs):
    execution_device = next(layer.parameters()).device
    layer_input = layer_input.to(execution_device)
    layer_kwargs = {
        key: value.to(execution_device) if isinstance(value, torch.Tensor) else value
        for key, value in layer_kwargs.items()
    }
    layer_kwargs['position_embeddings'] = (k.to(execution_device) for k in layer_kwargs['position_embeddings'])

    cache = {}
    handles = []

    def save_input(name):
        def hook(_, inputs, __):
            cache[name] = inputs[0]
        return hook

    handles.append(layer.self_attn.o_proj.register_forward_hook(save_input("attn_hidden")))
    handles.append(layer.mlp.down_proj.register_forward_hook(save_input("mlp_hidden")))

    layer_output = layer(layer_input, **layer_kwargs)
    cache["layer_output"] = get_hidden_states(layer_output)

    for handle in handles:
        handle.remove()

    return cache


def wrap_qwen_block_for_optimization(layer, attn_log_scale, mlp_log_scale, w_bit, a_bit, q_config):
    wrapped_layer = strip_dispatch_hooks(copy.deepcopy(layer)).cuda().eval()
    for param in wrapped_layer.parameters():
        param.requires_grad = False

    wrapped_layer.self_attn.q_proj = ScaledSTEQuantLinear(wrapped_layer.self_attn.q_proj, w_bit, a_bit, q_config, attn_log_scale)
    wrapped_layer.self_attn.k_proj = ScaledSTEQuantLinear(wrapped_layer.self_attn.k_proj, w_bit, a_bit, q_config, attn_log_scale)
    wrapped_layer.self_attn.v_proj = ScaledSTEQuantLinear(wrapped_layer.self_attn.v_proj, w_bit, a_bit, q_config, attn_log_scale)
    wrapped_layer.self_attn.o_proj = ScaledSTEQuantLinear(wrapped_layer.self_attn.o_proj, w_bit, a_bit, q_config)

    wrapped_layer.mlp.gate_proj = ScaledSTEQuantLinear(wrapped_layer.mlp.gate_proj, w_bit, a_bit, q_config, mlp_log_scale)
    wrapped_layer.mlp.up_proj = ScaledSTEQuantLinear(wrapped_layer.mlp.up_proj, w_bit, a_bit, q_config, mlp_log_scale)
    wrapped_layer.mlp.down_proj = ScaledSTEQuantLinear(wrapped_layer.mlp.down_proj, w_bit, a_bit, q_config)
    return wrapped_layer


def optimize_qwen_block_scales(layer, layer_input, layer_kwargs, head_scores, w_bit, a_bit, q_config, loss_mode="mae"):
    layer = strip_dispatch_hooks(layer)
    if layer.__class__.__name__ not in {"Qwen2VLDecoderLayer", "Qwen3VLTextDecoderLayer"}:
        raise NotImplementedError(f"Head-aware optimization currently supports Qwen2VL/Qwen3VL decoder layers, got {type(layer)}")

    total_batch_size = layer_input.shape[0]
    mini_batch_size = min(total_batch_size, OPTIM_MAX_SAMPLES)
    epochs = OPTIM_STEPS

    def slice_batch(value, start, end, key=None):
        if isinstance(value, torch.Tensor):
            if key == "position_ids" and value.dim() >= 2 and value.shape[0] in {3, 4} and value.shape[1] == total_batch_size:
                return value[:, start:end]
            if value.dim() > 0 and value.shape[0] == total_batch_size:
                return value[start:end]
            return value

        if isinstance(value, tuple):
            if key == "position_embeddings":
                sliced = []
                for item in value:
                    if torch.is_tensor(item) and item.dim() >= 2 and item.shape[0] == total_batch_size:
                        sliced.append(item[start:end])
                    else:
                        sliced.append(item)
                return tuple(sliced)
            return tuple(slice_batch(item, start, end, key) for item in value)

        if isinstance(value, list):
            return [slice_batch(item, start, end, key) for item in value]

        return value

    cached_batches = []
    for start_idx in range(0, total_batch_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, total_batch_size)
        mini_layer_input = layer_input[start_idx:end_idx].detach()
        mini_layer_kwargs = {
            key: slice_batch(value, start_idx, end_idx, key)
            for key, value in layer_kwargs.items()
        }
        with torch.no_grad():
            reference = capture_qwen_block_tensors(layer, mini_layer_input, mini_layer_kwargs)
        cached_batches.append((mini_layer_input, mini_layer_kwargs, reference))

    attn_log_scale = nn.Parameter(torch.zeros(layer.input_layernorm.weight.shape[0], device=layer_input.device))
    mlp_log_scale = nn.Parameter(torch.zeros(layer.post_attention_layernorm.weight.shape[0], device=layer_input.device))

    quant_layer = wrap_qwen_block_for_optimization(layer, attn_log_scale, mlp_log_scale, w_bit, a_bit, q_config)
    optimizer = torch.optim.Adam([attn_log_scale, mlp_log_scale], lr=OPTIM_LR)

    for _ in range(epochs):
        for mini_layer_input, mini_layer_kwargs, reference in cached_batches:
            optimizer.zero_grad()
            current = capture_qwen_block_tensors(quant_layer, mini_layer_input, mini_layer_kwargs)

            attn_loss = weighted_head_loss(
                reference["attn_hidden"],
                current["attn_hidden"],
                head_scores,
                layer.self_attn.head_dim,
                loss_mode,
            )
            mlp_loss = reconstruction_loss(reference["mlp_hidden"], current["mlp_hidden"], loss_mode)
            output_loss = reconstruction_loss(reference["layer_output"], current["layer_output"], loss_mode)
            loss = attn_loss + mlp_loss + output_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                attn_log_scale.clamp_(min=-4.0, max=4.0)
                mlp_log_scale.clamp_(min=-4.0, max=4.0)

    return build_scale_list(layer, torch.exp(attn_log_scale), torch.exp(mlp_log_scale))
