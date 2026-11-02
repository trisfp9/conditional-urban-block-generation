"""
R5. Software and hardware report (reviewer comments 3 and 5).

Run this INSIDE the environment you trained in (for example a Colab notebook cell:
  !python r5_environment_report.py
or paste the body into a cell). If training was on Colab and that runtime is gone, run it in a
fresh Colab runtime and note the GPU type you remember using.
"""
import importlib
import json
import platform
import sys

PKGS = ["numpy", "pandas", "scipy", "torch", "torchvision", "osmnx", "geopandas", "shapely",
        "networkx", "rasterio", "PIL", "matplotlib", "pytorch_fid", "torchmetrics", "torch_fidelity",
        "cleanfid", "statsmodels", "sklearn"]
rep = {"python": sys.version.split()[0], "platform": platform.platform()}
for p in PKGS:
    try:
        m = importlib.import_module(p)
        rep[p] = getattr(m, "__version__", "installed")
    except Exception:
        rep[p] = None
try:
    import torch
    rep["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        rep["gpu"] = torch.cuda.get_device_name(0)
        rep["cuda"] = torch.version.cuda
        rep["cudnn"] = torch.backends.cudnn.version()
except Exception:
    pass
print(json.dumps(rep, indent=2))
