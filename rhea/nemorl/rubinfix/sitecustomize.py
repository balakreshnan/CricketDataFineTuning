"""Loaded at start-up by every Python process that has this directory on PYTHONPATH.

Makes the stock Triton (3.7 from PyPI, as pinned by NeMo RL) usable on Vera Rubin (GPU architecture sm_107).
What was measured on Rhea:
  - its LLVM and ptxas do not know "sm_107a"                              -> every kernel fails to compile
  - "sm_100a" kernels are architecture-specific and Rubin refuses them    -> "no kernel image"
  - "sm_80" kernels are refused as well                                   -> "no kernel image"
  - plain "sm_100" kernels load and run, but Triton's code generation for capability 10.x emits Blackwell-only
    matrix-multiply instructions (tcgen05) that only exist under "sm_100a" -> LLVM abort in kernels that use tl.dot
So, when the GPU is a 10.1 to 10.9 device:
  1. report capability 10.0 to Triton (driver)                 -> ptx and cubin stages target 100
  2. name that target "sm_100", without the "a" suffix (compiler)
  3. run the code-generation passes as for capability 8.0 (knobs.runtime.override_arch = "sm80"), which emit the
     generic matrix-multiply instructions that are valid in sm_100 code
Cost: Triton kernels do not use the newest matrix-multiply instructions.
"""
import importlib.abc
import importlib.util
import sys

_DRIVER = "triton.backends.nvidia.driver"
_COMPILER = "triton.backends.nvidia.compiler"
_state = {"rubin": False}


def _patch_driver(module):
    cls = module.CudaDriver
    orig = cls.get_current_target

    def get_current_target(self):
        t = orig(self)
        if 101 <= int(t.arch) <= 109:
            _state["rubin"] = True
            import triton.knobs as knobs

            if not knobs.runtime.override_arch:
                knobs.runtime.override_arch = "sm80"
            return type(t)(t.backend, 100, t.warp_size)
        return t

    cls.get_current_target = get_current_target


def _patch_compiler(module):
    orig = module.sm_arch_from_capability

    def sm_arch_from_capability(capability):
        if _state["rubin"] and int(capability) == 100:
            return "sm_100"
        return orig(capability)

    module.sm_arch_from_capability = sm_arch_from_capability

    # PyTorch's compiler (inductor) builds the target itself from the device capability, bypassing the driver.
    backend = module.CUDABackend
    orig_init = backend.__init__

    def __init__(self, target):
        if 101 <= int(target.arch) <= 109:
            _state["rubin"] = True
            import triton.knobs as knobs

            if not knobs.runtime.override_arch:
                knobs.runtime.override_arch = "sm80"
            target = type(target)(target.backend, 100, target.warp_size)
        orig_init(self, target)

    backend.__init__ = __init__


_PATCHES = {_DRIVER: _patch_driver, _COMPILER: _patch_compiler}


class _Finder(importlib.abc.MetaPathFinder):
    """Patch the Triton modules when (and only if) something imports them."""

    _busy = False

    def find_spec(self, name, path, target=None):
        if name not in _PATCHES or _Finder._busy:
            return None
        _Finder._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            _Finder._busy = False
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _PATCHES[name](module)
            except Exception as e:  # noqa: BLE001  (never break the import)
                print(f"rubinfix: {name} not patched: {e!r}", file=sys.stderr)

        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
