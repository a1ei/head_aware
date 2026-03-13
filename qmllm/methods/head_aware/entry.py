import os

import torch

from qmllm.methods.head_aware.quantize.pre_quant import apply_head_aware, run_head_aware
from qmllm.methods.head_aware.quantize.quantizer import (
    pseudo_quantize_model_act,
    pseudo_quantize_model_weight,
    pseudo_quantize_model_weight_act,
)

def head_aware_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_head_aware_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    a_quant: bool = False,
    reweight: bool = False,
    distort: bool = False,
    loss_mode: str = "mae",
    head_boost: float = 2.0,
):
    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
    }

    assert scale_path is not None

    scale_exist = os.path.exists(scale_path)
    if run_head_aware_process and not scale_exist:
        model.to_cpu()
        head_aware_results = run_head_aware(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
            head_boost=head_boost,
        )

        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)

        torch.save(head_aware_results, scale_path)
        print("Head-aware results saved at", scale_path)

    if pseudo_quant:
        head_aware_results = torch.load(scale_path, map_location="cpu")
        apply_head_aware(model.model, head_aware_results)

        if a_quant:
            pseudo_quantize_model_act(model.model, w_bit=w_bit, a_bit=a_bit)
        elif not wa_quant:
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
