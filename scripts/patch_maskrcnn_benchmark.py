#!/usr/bin/env python
"""Port vqa-maskrcnn-benchmark's CUDA kernels to a modern PyTorch.

The bottom-up feature extractor is the only way left to regenerate the image
features (every published dump is offline), but it was written against PyTorch
1.x and does not build against 2.x: its kernels include `THC/THC.h`, a legacy
library PyTorch removed in 2.0.

The replacements are mechanical and semantics-preserving:

| legacy                        | modern                                            |
| ----------------------------- | ------------------------------------------------- |
| `THC/THC.h`                   | dropped; `ATen/cuda/CUDAContext.h` already present |
| `THC/THCAtomics.cuh`          | `ATen/cuda/Atomic.cuh`                             |
| `THC/THCDeviceUtils.cuh`      | `ATen/ceil_div.h`                                  |
| `THCCeilDiv(a, b)`            | `at::ceil_div(a, b)`                               |
| `THCudaCheck(x)`              | `AT_CUDA_CHECK(x)`                                 |
| `THCudaMalloc(state, n)`      | `c10::cuda::CUDACachingAllocator::raw_alloc(n)`    |
| `THCudaFree(state, p)`        | `c10::cuda::CUDACachingAllocator::raw_delete(p)`   |
| `THCState *state = ...`       | dropped; the caching allocator needs no handle     |
| `tensor.data<T>()`            | `tensor.data_ptr<T>()` (removed in 2.0)            |

Idempotent: re-running on already-patched sources is a no-op.

```bash
python scripts/patch_maskrcnn_benchmark.py --repo ~/vqa-maskrcnn-benchmark
```
"""

import argparse
import os
import re
import sys

# (pattern, replacement) applied in order to every source file under csrc/.
# Order matters: `.type().is_cuda()` must be rewritten before the bare `.type()`
# rule, or it would become the nonsensical `.scalar_type().is_cuda()`.
RULES = [
    (r"#include <THC/THC\.h>\n", ""),
    (r"#include <THC/THCAtomics\.cuh>", "#include <ATen/cuda/Atomic.cuh>"),
    (
        r"#include <THC/THCDeviceUtils\.cuh>",
        "#include <ATen/ceil_div.h>\n#include <c10/cuda/CUDACachingAllocator.h>\n#include <c10/cuda/CUDAException.h>",
    ),
    (r"\bTHCCeilDiv\b", "at::ceil_div"),
    (r"\bTHCudaCheck\b", "AT_CUDA_CHECK"),
    # Drop the THCState handle; the caching allocator is stateless from here.
    (r"[ \t]*THCState\s*\*\s*state\s*=[^;]*;[^\n]*\n", ""),
    (r"THCudaMalloc\(\s*state\s*,\s*", "c10::cuda::CUDACachingAllocator::raw_alloc("),
    (r"THCudaFree\(\s*state\s*,\s*", "c10::cuda::CUDACachingAllocator::raw_delete("),
    # `Tensor::type()` returned DeprecatedTypeProperties and is gone. Device
    # queries move to the tensor itself...
    (r"\.type\(\)\.is_cuda\(\)", ".is_cuda()"),
    # ...and every other use here wants a ScalarType: AT_DISPATCH_* arguments and
    # dtype comparisons alike. Safe only because `.is_cuda()` was rewritten above.
    (r"\.type\(\)", ".scalar_type()"),
    # `Tensor::data<T>()` was removed in favour of `data_ptr<T>()`.
    (r"\.data<([^>]+)>\(\)", r".data_ptr<\1>()"),
]

#: Everything under this directory is patched, so new files are not missed.
CSRC = "maskrcnn_benchmark/csrc"
SOURCE_SUFFIXES = (".cu", ".cpp", ".h", ".cuh")

#: The Python side broke too: `torch._six` was removed in 2.0. It existed to
#: paper over Python 2 vs 3, so on any supported interpreter `PY3` is just True.
PYTHON_RULES = [
    (r"torch\._six\.PY3", "True"),
    (r"from torch\._six import string_classes", "string_classes = str"),
    (r"from torch\._six import container_abcs", "import collections.abc as container_abcs"),
    (r"from torch\._six import int_classes", "int_classes = int"),
    # NumPy 1.24 removed the aliases for the builtin scalar types. The trailing
    # lookahead keeps `np.float32` / `np.int64` and friends intact.
    (r"\bnp\.float(?![0-9_a-zA-Z])", "float"),
    (r"\bnp\.int(?![0-9_a-zA-Z])", "int"),
    (r"\bnp\.bool(?![0-9_a-zA-Z])", "bool"),
    (r"\bnp\.object(?![0-9_a-zA-Z])", "object"),
    (r"\bnp\.str(?![0-9_a-zA-Z])", "str"),
    (r"\bnumpy\.float(?![0-9_a-zA-Z])", "float"),
    (r"\bnumpy\.int(?![0-9_a-zA-Z])", "int"),
]

#: Python files to scan; the package is small enough to walk entirely.
PYTHON_ROOT = "maskrcnn_benchmark"


def find_sources(repo: str):
    """Every C/C++/CUDA source under csrc, so nothing is missed by hand-listing."""
    root = os.path.join(repo, CSRC)
    if not os.path.isdir(root):
        raise SystemExit(f"{root} does not exist; is --repo pointing at a vqa-maskrcnn-benchmark checkout?")
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(SOURCE_SUFFIXES):
                yield os.path.join(dirpath, name)


def find_python_sources(repo: str):
    """Python modules that may reference APIs removed in torch 2.x."""
    root = os.path.join(repo, PYTHON_ROOT)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "build"]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def patch_file(path: str, rules=None) -> bool:
    with open(path, "r") as handle:
        original = handle.read()

    patched = original
    for pattern, replacement in rules if rules is not None else RULES:
        patched = re.sub(pattern, replacement, patched)

    if patched == original:
        return False

    if not os.path.exists(path + ".orig"):
        with open(path + ".orig", "w") as handle:
            handle.write(original)
    with open(path, "w") as handle:
        handle.write(patched)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Path to the vqa-maskrcnn-benchmark checkout")
    args = parser.parse_args()

    sources = list(find_sources(args.repo))
    changed = [path for path in sources if patch_file(path)]

    python_sources = list(find_python_sources(args.repo))
    python_changed = [path for path in python_sources if patch_file(path, PYTHON_RULES)]
    changed += python_changed

    for path in sources + python_changed:
        state = "patched  " if path in changed else "unchanged"
        print(f"{state} {os.path.relpath(path, args.repo)}")

    stale = {"THC": [], ".data<": [], ".type()": []}
    for path in sources:
        with open(path) as handle:
            content = handle.read()
        for marker in stale:
            if marker in content:
                stale[marker].append(os.path.relpath(path, args.repo))

    leftovers = {k: v for k, v in stale.items() if v}
    if leftovers:
        print("\nRemoved APIs still present, needs a manual look:")
        for marker, files in leftovers.items():
            print(f"  {marker}: {files}")
        return 1

    for path in python_sources:
        with open(path) as handle:
            if "torch._six" in handle.read():
                print(f"\ntorch._six still present in {os.path.relpath(path, args.repo)}")
                return 1

    print(f"\n{len(changed)} files patched ({len(python_changed)} python); no removed APIs remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
