import torch
import torchvision
import numpy
import matplotlib
import psutil

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU setup successful!")