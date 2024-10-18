# gpu_option.py

import torch

# Check if a GPU is available, else fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
