"""Does this container's Triton compile for this GPU? tl.sum lowers to shfl.sync.bfly (the failing intrinsic)."""
import torch, triton, triton.language as tl

@triton.jit
def _sum(x, y, N: tl.constexpr):
    o = tl.arange(0, N)
    tl.store(y, tl.sum(tl.load(x + o), 0))

print("torch", torch.__version__, "| triton", triton.__version__, "| gpu", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
x = torch.randn(128, device="cuda"); y = torch.empty(1, device="cuda")
_sum[(1,)](x, y, 128)
torch.cuda.synchronize()
print("TRITON OK", round(float(y), 4), round(float(x.sum()), 4))
