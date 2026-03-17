import copy
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_submodules
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from qmllm.utils.search import get_hidden_states, get_op_by_name, get_op_name


OPTIM_STEPS = 20
OPTIM_LR = 5e-3
OPTIM_MAX_SAMPLES = 8
LOG_SCALE_MIN = -4.0
LOG_SCALE_MAX = 4.0

ATTN_INPUT_GROUP = "attn_input"
ATTN_OUTPUT_GROUP = "attn_output"
MLP_INPUT_GROUP = "mlp_input"
MLP_OUTPUT_GROUP = "mlp_output"

SUPPORTED_QWEN_LAYERS = {"Qwen2VLDecoderLayer", "Qwen3VLTextDecoderLayer"}
SUPPORTED_RMSNORMS = {"Qwen2RMSNorm", "Qwen3VLTextRMSNorm"}


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

    weights = head_scores.to(diff.device, dtype=diff.dtype).clamp(min=0)
    weights = weights / weights.max().clamp(min=1e-5)
    return (diff * weights.view(1, num_heads, 1)).mean()


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device, dtype=ln.weight.dtype)
    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales.to(ln.bias.dtype))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device, dtype=fc.weight.dtype))


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError(f"Expected linear layers, got {type(fc1)} and {type(fc2)}")

    num_channels = scales.numel()
    if fc1.weight.shape[0] < num_channels:
        raise ValueError(
            f"Cannot scale {fc1.__class__.__name__}: out_features {fc1.weight.shape[0]} < scale size {num_channels}"
        )
    if fc2.weight.shape[1] != num_channels:
        raise ValueError(
            f"Cannot scale {fc2.__class__.__name__}: in_features {fc2.weight.shape[1]} != scale size {num_channels}"
        )

    row_scale = scales.to(fc1.weight.device, dtype=fc1.weight.dtype).view(-1, 1)
    fc1.weight[-num_channels:].div_(row_scale)
    if fc1.bias is not None:
        fc1.bias[-num_channels:].div_(scales.to(fc1.bias.device, dtype=fc1.bias.dtype))

    fc2.weight.mul_(scales.to(fc2.weight.device, dtype=fc2.weight.dtype).view(1, -1))


def _get_module_device(module):
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _get_compute_dtype(module):
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            return submodule.weight.dtype
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def apply_scale(module, scales_list):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        execution_device = _get_module_device(prev_op)
        prev_op.to(execution_device)
        for layer in layers:
            layer.to(execution_device)
        scales = scales.to(execution_device)

        if isinstance(prev_op, nn.Linear):
            if len(layers) != 1:
                raise ValueError("Linear-to-linear scaling expects exactly one following layer.")
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ in SUPPORTED_RMSNORMS:
            scale_ln_fcs(prev_op, layers, scales)
        else:
            raise NotImplementedError(f"Unsupported previous op for head-aware scaling: {type(prev_op)}")


def _move_to_device(value, device, dtype=None):
    if isinstance(value, torch.Tensor):
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device, dtype) for item in value]
    return value


def _slice_batch(value, start, end, total_batch_size, key=None):
    if isinstance(value, torch.Tensor):
        if key == "position_ids" and value.dim() >= 2 and value.shape[0] in {3, 4} and value.shape[1] == total_batch_size:
            return value[:, start:end]
        if value.dim() > 0 and value.shape[0] == total_batch_size:
            return value[start:end]
        return value

    if isinstance(value, tuple):
        return tuple(_slice_batch(item, start, end, total_batch_size) for item in value)

    if isinstance(value, list):
        return [_slice_batch(item, start, end, total_batch_size) for item in value]

    return value


def forward_qwen_block(layer, layer_input, layer_kwargs):
    execution_device = _get_module_device(layer)
    compute_dtype = _get_compute_dtype(layer)
    layer_input = _move_to_device(layer_input, execution_device, compute_dtype)
    layer_kwargs = {
        key: _move_to_device(value, execution_device, compute_dtype)
        for key, value in layer_kwargs.items()
    }

    if execution_device.type == "cuda" and compute_dtype in {torch.float16, torch.bfloat16}:
        autocast_context = torch.autocast(device_type="cuda", dtype=compute_dtype)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        return layer(layer_input, **layer_kwargs)


def capture_qwen_block_tensors(layer, layer_input, layer_kwargs):
    cache = {}
    handles = []

    def save_input(name):
        def hook(_, inputs, __):
            cache[name] = inputs[0]

        return hook

    def save_output(name):
        def hook(_, __, output):
            cache[name] = get_hidden_states(output)

        return hook

    handles.append(layer.self_attn.o_proj.register_forward_hook(save_input("attn_hidden")))
    handles.append(layer.self_attn.register_forward_hook(save_output("attn_output")))
    handles.append(layer.mlp.down_proj.register_forward_hook(save_input("mlp_hidden")))
    handles.append(layer.mlp.register_forward_hook(save_output("mlp_output")))

    try:
        layer_output = forward_qwen_block(layer, layer_input, layer_kwargs)
        cache["layer_output"] = get_hidden_states(layer_output)
    finally:
        for handle in handles:
            handle.remove()

    return cache


def _build_cached_batches(layer, layer_input, layer_kwargs):
    total_batch_size = layer_input.shape[0]
    mini_batch_size = min(total_batch_size, OPTIM_MAX_SAMPLES)
    cached_batches = []

    layer_kwargs = dict(layer_kwargs)
    layer_kwargs["use_cache"] = False

    for start_idx in range(0, total_batch_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, total_batch_size)
        mini_layer_input = layer_input[start_idx:end_idx].detach()
        mini_layer_kwargs = {
            key: _slice_batch(value, start_idx, end_idx, total_batch_size, key)
            for key, value in layer_kwargs.items()
        }
        with torch.no_grad():
            reference = capture_qwen_block_tensors(layer, mini_layer_input, mini_layer_kwargs)
        cached_batches.append((mini_layer_input, mini_layer_kwargs, reference))

    return cached_batches


def _build_qwen_scale_specs(layer):
    specs = [
        {
            "group_name": ATTN_INPUT_GROUP,
            "prev_op": layer.input_layernorm,
            "layers": [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        },
        {
            "group_name": MLP_INPUT_GROUP,
            "prev_op": layer.post_attention_layernorm,
            "layers": [layer.mlp.gate_proj, layer.mlp.up_proj],
        },
        {
            "group_name": MLP_OUTPUT_GROUP,
            "prev_op": layer.mlp.up_proj,
            "layers": [layer.mlp.down_proj],
        },
    ]

    if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
        specs.insert(
            1,
            {
                "group_name": ATTN_OUTPUT_GROUP,
                "prev_op": layer.self_attn.v_proj,
                "layers": [layer.self_attn.o_proj],
            },
        )

    return specs


def _get_scale_size(layer, group_name):
    if group_name == ATTN_INPUT_GROUP:
        return layer.self_attn.q_proj.in_features
    if group_name == ATTN_OUTPUT_GROUP:
        return layer.self_attn.o_proj.in_features
    if group_name == MLP_INPUT_GROUP:
        return layer.mlp.gate_proj.in_features
    if group_name == MLP_OUTPUT_GROUP:
        return layer.mlp.down_proj.in_features
    raise ValueError(f"Unknown scale group: {group_name}")


def _wrap_qwen_block_for_group(layer, group_name, log_scale, w_bit, a_bit, q_config):
    execution_device = _get_module_device(layer)
    wrapped_layer = strip_dispatch_hooks(copy.deepcopy(layer)).to(execution_device).eval()
    for param in wrapped_layer.parameters():
        param.requires_grad = False

    if group_name == ATTN_INPUT_GROUP:
        wrapped_layer.self_attn.q_proj = ScaledSTEQuantLinear(
            wrapped_layer.self_attn.q_proj, w_bit, a_bit, q_config, log_scale
        )
        wrapped_layer.self_attn.k_proj = ScaledSTEQuantLinear(
            wrapped_layer.self_attn.k_proj, w_bit, a_bit, q_config, log_scale
        )
        wrapped_layer.self_attn.v_proj = ScaledSTEQuantLinear(
            wrapped_layer.self_attn.v_proj, w_bit, a_bit, q_config, log_scale
        )
    elif group_name == ATTN_OUTPUT_GROUP:
        wrapped_layer.self_attn.o_proj = ScaledSTEQuantLinear(
            wrapped_layer.self_attn.o_proj, w_bit, a_bit, q_config, log_scale
        )
    elif group_name == MLP_INPUT_GROUP:
        wrapped_layer.mlp.gate_proj = ScaledSTEQuantLinear(
            wrapped_layer.mlp.gate_proj, w_bit, a_bit, q_config, log_scale
        )
        wrapped_layer.mlp.up_proj = ScaledSTEQuantLinear(
            wrapped_layer.mlp.up_proj, w_bit, a_bit, q_config, log_scale
        )
    elif group_name == MLP_OUTPUT_GROUP:
        wrapped_layer.mlp.down_proj = ScaledSTEQuantLinear(
            wrapped_layer.mlp.down_proj, w_bit, a_bit, q_config, log_scale
        )
    else:
        raise ValueError(f"Unknown scale group: {group_name}")

    return wrapped_layer


def _compute_group_loss(group_name, reference, current, head_scores, head_dim, loss_mode):
    if group_name == ATTN_INPUT_GROUP:
        # return weighted_head_loss(
        #     reference["attn_hidden"],
        #     current["attn_hidden"],
        #     head_scores,
        #     head_dim,
        #     loss_mode, 
        # ) #+ reconstruction_loss(reference["attn_output"], current["attn_output"], loss_mode)
        return reconstruction_loss(reference["attn_hidden"], current["attn_hidden"], loss_mode)

    if group_name == ATTN_OUTPUT_GROUP:
        return reconstruction_loss(reference["attn_output"], current["attn_output"], loss_mode)

    if group_name in {MLP_INPUT_GROUP, MLP_OUTPUT_GROUP}:
        return reconstruction_loss(reference["mlp_output"], current["mlp_output"], loss_mode)

    raise ValueError(f"Unknown scale group: {group_name}")


def _search_qwen_scale_group(layer, group_name, cached_batches, head_scores, w_bit, a_bit, q_config, loss_mode):
    if (w_bit is None or w_bit >= 16) and (a_bit is None or a_bit >= 16):
        return torch.ones(_get_scale_size(layer, group_name), device=_get_module_device(layer))

    execution_device = _get_module_device(layer)
    log_scale = nn.Parameter(torch.zeros(_get_scale_size(layer, group_name), device=execution_device))
    quant_layer = _wrap_qwen_block_for_group(layer, group_name, log_scale, w_bit, a_bit, q_config)
    optimizer = torch.optim.Adam([log_scale], lr=OPTIM_LR)

    for _ in range(OPTIM_STEPS):
        for mini_layer_input, mini_layer_kwargs, reference in cached_batches:
            optimizer.zero_grad(set_to_none=True)
            current = capture_qwen_block_tensors(quant_layer, mini_layer_input, mini_layer_kwargs)
            loss = _compute_group_loss(
                group_name,
                reference,
                current,
                head_scores,
                layer.self_attn.head_dim,
                loss_mode,
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                log_scale.clamp_(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX)

    scales = torch.exp(log_scale).detach()
    del optimizer
    del quant_layer
    if execution_device.type == "cuda":
        torch.cuda.empty_cache()
    return scales


def optimize_qwen_block_scales(layer, layer_input, layer_kwargs, head_scores, w_bit, a_bit, q_config, loss_mode="mae"):
    layer = strip_dispatch_hooks(layer)
    if layer.__class__.__name__ not in SUPPORTED_QWEN_LAYERS:
        raise NotImplementedError(
            f"Head-aware optimization currently supports Qwen2VL/Qwen3VL decoder layers, got {type(layer)}"
        )

    cached_batches = _build_cached_batches(layer, layer_input, layer_kwargs)

    def _auto_get_scale(scale_spec):
        scales = _search_qwen_scale_group(
            layer,
            scale_spec["group_name"],
            cached_batches,
            head_scores,
            w_bit,
            a_bit,
            q_config,
            loss_mode,
        )
        return (
            get_op_name(layer, scale_spec["prev_op"]),
            tuple(get_op_name(layer, module) for module in scale_spec["layers"]),
            scales.detach().cpu(),
        )

    scales_list = []
    for scale_spec in _build_qwen_scale_specs(layer):
        scales_list.append(_auto_get_scale(scale_spec))

    return scales_list
