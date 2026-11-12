"""Loaded at start-up by every Python process when this directory is on PYTHONPATH.

The Dynamo SGLang 1.5.1 image ships Triton 3.7.1 / CUDA 13.0, whose LLVM and ptxas do not know Vera Rubin
(sm_107a), so every Triton kernel fails to compile. Rubin runs sm_100 binaries (that is what the image's own
PyTorch does: its arch list stops at sm_100/110/120 and GPU ops work), so compile Triton kernels for plain sm_100
(not sm_100a: the "a" variants are architecture-specific and Rubin refuses to load them):
  1) the LLVM processor / ptxas --gpu-name come from sm_arch_from_capability
  2) Triton also rewrites the PTX ".target" line from the raw capability, so fix that line after make_ptx
"""
try:
    import re

    import triton.backends.nvidia.compiler as _c

    _orig_arch = _c.sm_arch_from_capability

    def _arch(capability):
        return "sm_100" if 101 <= int(capability) <= 109 else _orig_arch(capability)

    _c.sm_arch_from_capability = _arch

    _orig_ptx = _c.CUDABackend.make_ptx

    def _make_ptx(self, *args, **kwargs):
        ptx = _orig_ptx(self, *args, **kwargs)
        return re.sub(r"\.target\s+sm_10[1-9]a?", ".target sm_100", ptx) if isinstance(ptx, str) else ptx

    _c.CUDABackend.make_ptx = _make_ptx
except Exception as e:  # noqa: BLE001  (never break interpreter start-up)
    import sys

    print(f"rubinfix: triton patch not applied: {e!r}", file=sys.stderr)
