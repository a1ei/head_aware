import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from qmllm.methods.mbq.quantize.qmodule import ScaledActivation
from qmllm.utils.search import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear

EMBEDDING_KEYWORDS = ["embed"]  
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)

@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            # m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bits=w_bit, **q_config
            )
            # m.cpu()


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


@torch.no_grad()
def pseudo_quantize_model_weight_act(
    model,
    w_bit,
    a_bit,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight activation quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
            father_module = get_module_by_name_suffix(layers[i], '.'.join(n.split(".")[:-1]))
            setattr(father_module, n.split('.')[-1], new_linear)
            del new_linear, m
            torch.cuda.empty_cache()

@torch.no_grad()
def pseudo_quantize_model_act(
    model,
    w_bit,
    a_bit,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight activation quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            new_linear = WALinear.from_float(m, weight_quant="only_act", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
            father_module = get_module_by_name_suffix(layers[i], '.'.join(n.split(".")[:-1]))
            setattr(father_module, n.split('.')[-1], new_linear)
            del new_linear, m
            torch.cuda.empty_cache()


# import torch
# import torch.nn as nn

class PCASimilarityHook:
    def __init__(self, vision_mask, text_mask, k=10, max_tokens=2000):
        """
        vision_mask, text_mask: (batch, seq_len) bool
        k: number of principal directions
        max_tokens: subsample to avoid OOM
        """
        self.vision_mask = vision_mask
        self.text_mask = text_mask
        self.k = k
        self.max_tokens = max_tokens
        self.layer_sims = []
        self.layer_sims_text = []

    def compute_pca_basis(self, X):
        # X: (N, d)
        X = X - X.mean(dim=0, keepdim=True)

        # optional subsample
        if X.shape[0] > self.max_tokens:
            idx = torch.randperm(X.shape[0])[:self.max_tokens]
            X = X[idx]

        # SVD
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        basis = Vh[:self.k].T  # (d, k)
        return basis

    def subspace_similarity(self, Uv, Ut):
        # Uv, Ut: (d, k)
        M = Uv.T @ Ut  # (k, k)
        sim = torch.norm(M, p='fro') / self.k
        return sim.item()

    def hook_fn(self, module, input, output):
        # input[0] shape: (batch, seq_len, hidden)
        hidden = input[0].detach().to(torch.float32)

        B, S, D = hidden.shape
        hidden = hidden.reshape(B * S, D)

        vision_mask = self.vision_mask.reshape(-1).to(hidden.device)
        text_mask = self.text_mask.reshape(-1).to(hidden.device)

        Xv = hidden[vision_mask]
        Xt = hidden[text_mask]

        if Xv.shape[0] < self.k or Xt.shape[0] < self.k:
            self.layer_sims.append(None)
            return

        Uv = self.compute_pca_basis(Xv)
        Ut = self.compute_pca_basis(Xt)

        sim = self.subspace_similarity(Uv, Ut)
        self.layer_sims.append(sim)
        print(self.run_test(input[0]))
    
    def hook_fn_for_text_diversity(self, module, input, output):
        hidden = input[0].detach().to(torch.float32)  # (B, S, D)
        B, S, D = hidden.shape

        # 展平
        hidden = hidden.reshape(B * S, D)
        text_mask = self.text_mask.reshape(-1).to(hidden.device)
        Xt = hidden[text_mask]  # (N, D)

        if Xt.shape[0] < 2 * self.k:
            self.layer_sims_text.append(None)
            return

        # 方案：按样本 split 成两组（比如前一半 vs 后一半）
        # 先恢复成 (B, S, D)，再按 batch 维度分
        hidden_batch = input[0].detach().to(torch.float32)  # (B, S, D)
        text_mask_batch = self.text_mask.to(hidden.device)  # (B, S)

        # 提取每个样本的有效文本 token 表示（变长）
        group1_X, group2_X = [], []
        for i in range(B):
            tokens = hidden_batch[i][text_mask_batch[i]]  # (s_i, D)
            if tokens.shape[0] == 0:
                continue
            if i % 2 == 0:
                group1_X.append(tokens)
            else:
                group2_X.append(tokens)

        if not group1_X or not group2_X:
            self.layer_sims_text.append(None)
            return

        X1 = torch.cat(group1_X, dim=0)  # (N1, D)
        X2 = torch.cat(group2_X, dim=0)  # (N2, D)

        if X1.shape[0] < self.k or X2.shape[0] < self.k:
            self.layer_sims_text.append(None)
            return

        U1 = self.compute_pca_basis(X1)
        U2 = self.compute_pca_basis(X2)

        sim = self.subspace_similarity(U1, U2)
        self.layer_sims_text.append(sim)



    def fake_quant(self,x):
        scale = x.abs().max() / 127
        x_q = torch.round(x / scale) * scale
        return x_q

    def quant_error(self,x):
        x_q = self.fake_quant(x)
        return ((x - x_q) ** 2).mean().item()

    def quant_error_with_rotation(self, x, R):
        # rotate
        x_rot = x @ R
        # quantize
        x_rot_q = self.fake_quant(x_rot)
        # rotate back
        x_recon = x_rot_q @ R.T
        # compare in original space
        return ((x - x_recon) ** 2).mean().item()

    def compute_pca_rotation(self,x, k=None):
        # x: [N, D]
        x = x - x.mean(dim=0, keepdim=True)
        cov = x.T @ x / x.shape[0]
        U, S, V = torch.linalg.svd(cov)
        return U  # full rotation


    def run_test(self,x):
        x = x.detach().to(torch.float32)
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        vision_mask=self.vision_mask.to(x.device)
        text_mask=self.text_mask.to(x.device)

        vision = x_flat[vision_mask.view(-1)]
        text = x_flat[text_mask.view(-1)]

        # 原始误差
        vision_err = self.quant_error(vision)
        text_err = self.quant_error(text)

        # 学 text PCA rotation
        R = self.compute_pca_rotation(text)

        vision_rot = vision @ R
        text_rot = text @ R

        vision_err_rot = self.quant_error_with_rotation(vision_rot,R)
        text_err_rot = self.quant_error_with_rotation(text_rot,R)


        R_vision = self.compute_pca_rotation(vision)
        vision_rot_vision = vision @ R_vision
        text_rot_vision = text @ R_vision
        vision_err_rot_vision = self.quant_error_with_rotation(vision_rot_vision,R_vision)
        text_err_rot_vision = self.quant_error_with_rotation(text_rot_vision,R_vision)

        return {
            "vision_err": vision_err,
            "text_err": text_err,
            "vision_err_rot_text": vision_err_rot,
            "text_err_rot_text": text_err_rot,
            "vision_err_rot_vision": vision_err_rot_vision,
            "text_err_rot_vision": text_err_rot_vision,
        }
