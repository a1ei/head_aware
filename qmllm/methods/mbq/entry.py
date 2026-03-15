import os
import torch

from qmllm.methods.mbq.quantize.pre_quant import run_mbq, apply_mbq ,process_input
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act , pseudo_quantize_model_act
import tqdm

def mbq_entry(model, prompt_inputs, prompt_kwargs, run_mbq_process: bool, pseudo_quant: bool, scale_path: str=None, zero_point: str=True, q_group_size: int=128, w_bit: int=4, a_bit: int=16,wa_quant: bool=False, a_quant: bool=False, reweight: bool=False, distort: bool=False, loss_mode: str="mae"):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    run_mbq_process: whether to run the MBQ process
    '''
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    scale_exist = os.path.exists(scale_path)
    # reparameterization
    if run_mbq_process and not scale_exist:
        model.to_cpu()
        mbq_results = run_mbq(
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
        )
        
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        
        torch.save(mbq_results, scale_path)
        print("MBQ results saved at", scale_path)
    #————————————————————————————————————————————————————————————————————————————————————                               
    # inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)                                     
    # from qmllm.models.qwen3_vl.qwen3_vl_attention import self_Qwen3VLTextAttention                                      
    # for layer in model.model.model.language_model.layers:                                                               
    #     layer.self_attn = self_Qwen3VLTextAttention(layer.self_attn)                                                    
                                                                                                                        
    # with torch.no_grad():                                                                                               
    #     mini_batch = 1                                                                                                  
    #     total_samples = next(iter(prompt_inputs.values())).shape[0]                                                     
    #     accum_steps = int(total_samples/mini_batch)                                                                     
                                                                                                                        
    #     for i in tqdm.tqdm(range(0, total_samples, mini_batch), desc="compute ....."):                                  
    #         mini_inputs = {}                                                                                            
    #         for k in inputs:                                                                                            
    #             if isinstance(inputs[k], torch.Tensor):                                                                 
    #                 mini_inputs[k] = inputs[k][i:i+mini_batch]                                                          
    #         for layer in model.model.model.language_model.layers:                                                       
    #             layer.self_attn.vision_mask = vision_mask[i:i+mini_batch]                                               
    #             layer.self_attn.caption_mask = caption_mask[i:i+mini_batch]                                             
    #             layer.self_attn.data_num = i                                                                            
    #         outputs = model(**mini_inputs)                                                                              
    #     print("finish")                                                                                                 
    #————————————————————————————————————————————————————————————————————————————————————
    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cpu")
        apply_mbq(model.model, mbq_results)

        if a_quant:
            #activation quantization
            pseudo_quantize_model_act(model.model, w_bit=w_bit, a_bit=a_bit)
        elif not wa_quant:
            # weight quantization
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            # weight activation quantization
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()

    return model
