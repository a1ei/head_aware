import gc

import torch
import torch.nn as nn
import tqdm

from qmllm.methods.head_aware.attention import patch_qwen3vl_attention_modules, restore_qwen3vl_attention_modules
from qmllm.methods.head_aware.quantize.auto_scale import apply_scale, forward_qwen_block, optimize_qwen_block_scales
from qmllm.methods.mbq.quantize.pre_quant import get_blocks, move_embed, process_input
from qmllm.utils.search import append_str_prefix, get_hidden_states, get_op_name


def _get_batch_size(prompt_inputs, prompt_kwargs):
    for collection in (prompt_inputs, prompt_kwargs):
        for value in collection.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
    raise ValueError("Cannot infer batch size from calibration inputs.")


def collect_head_cross_scores(model, prompt_inputs, prompt_kwargs):
    if model.model.__class__.__name__ != "Qwen3VLForConditionalGeneration":
        return {}

    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)
    if vision_mask is None:
        return {}

    if caption_mask is None:
        caption_mask = ~vision_mask

    recorder, replaced_modules = patch_qwen3vl_attention_modules(model)
    if not replaced_modules:
        return {}

    total_samples = _get_batch_size(prompt_inputs, prompt_kwargs)
    model.to_cuda()

    try:
        for index in tqdm.tqdm(range(total_samples), desc="Collecting cross-head scores..."):
            mini_inputs = {}
            for key, value in inputs.items():
                if key == "labels":
                    continue
                if isinstance(value, torch.Tensor):
                    mini_inputs[key] = value[index : index + 1]
                else:
                    mini_inputs[key] = value

            token_mask = mini_inputs.get("attention_mask")
            if token_mask is None:
                token_mask = torch.ones_like(vision_mask[index : index + 1], dtype=torch.bool)

            for layer, _ in replaced_modules:
                layer.self_attn.vision_mask = vision_mask[index : index + 1]
                layer.self_attn.text_mask = caption_mask[index : index + 1]
                layer.self_attn.token_mask = token_mask
                layer.self_attn.data_index = index  #change here
                layer.self_attn.save_attn_output = True  #change here

            model(**mini_inputs)
    finally:
        restore_qwen3vl_attention_modules(replaced_modules)
        model.to_cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return recorder.export()


def run_head_aware(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    auto_scale=True,
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    distort=False,
    head_boost=2.0,
):
    del auto_scale, wa_quant, reweight, distort, head_boost

    if "bigcode" in str(model.model.__class__).lower():
        model.transformer.bias = model.transformer.bias.to("cuda")

    head_scores = collect_head_cross_scores(model, prompt_inputs, prompt_kwargs)
    layers = get_blocks(model.model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model.model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError

    inputs, _, _ = process_input(prompt_inputs, prompt_kwargs)
    layers[0] = Catcher(layers[0])

    model.to_cuda()
    try:
        model(**inputs)
    except ValueError:
        pass

    model.to_cpu()
    layers[0] = layers[0].module
    inps = inps[0]
    layer_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    head_aware_results = {
        "scale": [],
        "head_scores": head_scores,
    }

    for layer_idx in tqdm.tqdm(range(len(layers)), desc="Running head-aware quantization..."):
        layer = layers[layer_idx].cuda().eval()
        layer_input = inps.to(next(layer.parameters()).device)

        current_kwargs = {}
        for key, value in layer_kwargs.items():
            if isinstance(value, torch.Tensor):
                current_kwargs[key] = value.to(next(layer.parameters()).device)
            else:
                current_kwargs[key] = value

        current_head_scores = None
        if layer_idx in head_scores:
            current_head_scores = head_scores[layer_idx]["cross_scores"].to(layer_input.device)

        scales_list = optimize_qwen_block_scales(
            layer,
            layer_input,
            current_kwargs,
            current_head_scores,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            loss_mode=loss_mode,
        )
        apply_scale(layers[layer_idx], scales_list)
        head_aware_results["scale"] += append_str_prefix(scales_list, get_op_name(model.model, layer) + ".")

        with torch.no_grad():
            inps = get_hidden_states(forward_qwen_block(layer, layer_input, current_kwargs))

        layers[layer_idx] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return head_aware_results


def apply_head_aware(model, head_aware_results):
    apply_scale(model, head_aware_results["scale"])
