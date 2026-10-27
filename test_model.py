
import torch
import torch.nn as nn
from torchvision.models import resnet18

# 1. Select CPU
device = torch.device("cpu")
print("Using device:", device)

# 2. Create ResNet-18 without pretrained weights
model = resnet18(weights=None)

# 3. Adapt the first convolution for 32x32 images
model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

# 4. Remove max pooling for small images
model.maxpool = nn.Identity()

# 5. Set the output layer for 10 CIFAR-10 classes
model.fc = nn.Linear(
    model.fc.in_features,
    10
)

# 6. Move model to CPU
model = model.to(device)

# 7. Create four dummy RGB images
dummy_images = torch.randn(4, 3, 32, 32)
dummy_images = dummy_images.to(device)

# 8. Test the model
model.eval()

with torch.no_grad():
    outputs = model(dummy_images)

# 9. Display results
print("\nModel created successfully!")
print("Input shape:", dummy_images.shape)
print("Output shape:", outputs.shape)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Total parameters:", total_parameters)

# 10. Verify output dimensions
assert outputs.shape == (4, 10)

print("\nForward-pass test passed!")
