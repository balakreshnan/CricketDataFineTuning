#!/usr/bin/env python
"""Does this Python environment work on a Vera Rubin GPU? Each check is independent; prints PASS/FAIL per check."""
import sys
import traceback


def check(name, fn):
    try:
        print(f"PASS {name}: {fn()}", flush=True)
    except Exception as e:  # noqa: BLE001
        last = traceback.format_exc().strip().splitlines()
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:500]}", flush=True)
        print("     " + " | ".join(l.strip() for l in last[-4:])[:700], flush=True)


print("python", sys.version.split()[0], flush=True)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list(), flush=True)
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), flush=True)
d = "cuda:0"


def matmul():
    a = torch.randn(2048, 2048, device=d, dtype=torch.bfloat16)
    return float((a @ a).float().abs().mean())


def sdpa():
    q = torch.randn(2, 8, 128, 64, device=d, dtype=torch.bfloat16)
    return tuple(torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True).shape)


def conv_backward():
    m = torch.nn.Conv1d(64, 64, 4, padding=3, groups=64).to(d)
    x = torch.randn(4, 64, 256, device=d, requires_grad=True)
    m(x).sum().backward()
    return float(x.grad.abs().mean())


def triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def add_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    x = torch.randn(4096, device=d)
    o = torch.empty_like(x)
    add_k[(4,)](x, x, o, 4096, BLOCK=1024)
    return f"triton {triton.__version__}, max err {float((o - 2 * x).abs().max())}"


def compile_fn():
    f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)
    return float(f(torch.randn(64, 64, device=d)).mean())


def flashinfer_ops():
    import flashinfer

    p = torch.softmax(torch.randn(4, 1000, device=d), -1)
    s = flashinfer.sampling.top_p_sampling_from_probs(p, 0.9)
    return f"flashinfer {flashinfer.__version__}, sampled {tuple(s.shape)}"


def vllm_import():
    import vllm

    return f"vllm {vllm.__version__}"


def transformers_nemotron():
    import transformers
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(sys.argv[1])
    return f"transformers {transformers.__version__}, model_type {c.model_type}"


def mamba_kernels():
    import mamba_ssm
    from mamba_ssm import Mamba2

    m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=64).to(d)
    x = torch.randn(2, 128, 256, device=d, requires_grad=True)
    m(x).sum().backward()
    return f"mamba_ssm {mamba_ssm.__version__}, grad {float(x.grad.abs().mean()):.4f}"


def flash_attn_op():
    import flash_attn
    from flash_attn import flash_attn_func

    q = torch.randn(2, 128, 8, 64, device=d, dtype=torch.bfloat16)
    return f"flash_attn {flash_attn.__version__}, out {tuple(flash_attn_func(q, q, q, causal=True).shape)}"


def transformer_engine_op():
    import transformer_engine
    import transformer_engine.pytorch as te

    lin = te.Linear(256, 256).to(d)
    y = lin(torch.randn(4, 256, device=d))
    return f"transformer_engine {transformer_engine.__version__}, cudnn {te.get_cudnn_version()}, out {tuple(y.shape)}"


def automodel_import():
    import nemo_automodel

    return f"nemo_automodel {getattr(nemo_automodel, '__version__', '?')}"


def peft_import():
    import peft

    return f"peft {peft.__version__}"


for n, f in (("matmul bf16", matmul), ("attention (sdpa)", sdpa), ("conv1d backward", conv_backward),
             ("triton jit kernel", triton_kernel), ("torch.compile", compile_fn), ("flashinfer sampling", flashinfer_ops),
             ("vllm import", vllm_import), ("mamba_ssm kernels", mamba_kernels),
             ("flash_attn", flash_attn_op), ("transformer_engine", transformer_engine_op),
             ("nemo_automodel import", automodel_import), ("peft import", peft_import)):
    check(n, f)
if len(sys.argv) > 1:
    check("nemotron config", transformers_nemotron)
