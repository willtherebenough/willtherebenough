"""
Standalone GPU check. Run this in any venv to see what it actually has.

    python check_gpu.py

Run it in the project where the GPU already works: it prints a ready-to-paste
pip line that reproduces that exact build. Then run it here to confirm.
"""

import sys

print(f"Interpreter : {sys.executable}")

try:
    import torch
except ImportError:
    print("torch       : NOT INSTALLED in this environment")
    raise SystemExit(1)

version = torch.__version__
base = version.split("+")[0]
cuda_build = torch.version.cuda

print(f"torch       : {version}")
print(f"CUDA build  : {cuda_build}")
print(f"available   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM        : {total:.1f} GB")

print()

# torchaudio must match torch exactly. A mismatch is what produces
# "WinError 127: The specified procedure could not be found" on import.
try:
    import torchaudio

    print(f"torchaudio  : {torchaudio.__version__}", end="")
    if torchaudio.__version__.split("+")[0] != base:
        print("   <-- MISMATCH, must equal torch")
    else:
        print("   (matches)")
except ImportError:
    print("torchaudio  : not installed (fine — main.py doesn't import it)")
except OSError as exc:
    print(f"torchaudio  : INSTALLED BUT BROKEN\n              {exc}")
    print("              Reinstall it at the same version as torch.")

print()

if cuda_build:
    tag = "cu" + cuda_build.replace(".", "")
    print("To reproduce this exact build in another venv:")
    print()
    print(f"  pip install torch=={base} torchaudio=={base} \\")
    print(f"      --index-url https://download.pytorch.org/whl/{tag}")
    print()
    print("Or as a requirements file:")
    print()
    print(f"  --index-url https://download.pytorch.org/whl/{tag}")
    print(f"  torch=={base}")
    print(f"  torchaudio=={base}")
else:
    print("This is the CPU-only wheel — it will never see the GPU.")
    print()
    print("Run this script in the project where CUDA works and copy the pip")
    print("line it prints. Then, in THIS venv:")
    print()
    print("  pip uninstall -y torch torchvision torchaudio")
    print("  <paste the line from the working project>")
    print()
    print("Note: the cu124 index no longer carries current torch releases.")
    print("CUDA 13.0 (cu130) is the default now, with cu126 for older drivers.")

if not torch.cuda.is_available() and cuda_build:
    print()
    print("CUDA build present but no GPU visible. Check that nvidia-smi runs")
    print("and that CUDA_VISIBLE_DEVICES is not set to an empty value.")
