import torch
import os
from  embedwise.utils import NotInstalledError

device: torch.device
if torch.cuda.is_available():
    device_id = os.environ.get("device")
    device = torch.device(f"cuda:{device_id}") if device_id else torch.device("cuda:0")
else:
    device = torch.device("cpu")

__version__ = "0.0.1"

__all__ = [
    "device",
    "__version__",
    "NotInstalledError",
]