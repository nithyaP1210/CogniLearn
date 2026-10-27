
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.models import resnet18

# 1. Select CPU
device = torch.device("cpu")

# 2. CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 3. Recreate the exact ResNet-18 architecture used in training
model = resnet18(weights=None)

model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

# 4. Load your saved model weights
model_path = (
    "resnet18_reference_results/"
    "resnet18_cifar10_reference.pth"
)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Saved ResNet-18 model loaded successfully!")

# 5. Load the CIFAR-10 test dataset
# Keep the original image for display.
display_transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=display_transform
)

# 6. Select one test image
image_index = 0

image, actual_label = test_dataset[image_index]

# 7. Apply the same normalization used during training
normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)

input_image = normalize(image).unsqueeze(0).to(device)

# Shape becomes [1, 3, 32, 32]:
# 1 image, 3 color channels, 32 x 32 pixels.

# 8. Predict the class
with torch.no_grad():
    outputs = model(input_image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities, dim=1
    )

predicted_label = predicted_index.item()
confidence_percent = confidence.item() * 100

# 9. Display the results
print("\n--- Prediction Results ---")
print("Actual class:", class_names[actual_label])
print("Predicted class:", class_names[predicted_label])
print(f"Prediction confidence: {confidence_percent:.2f}%")

if actual_label == predicted_label:
    print("Result: Correct prediction")
else:
    print("Result: Incorrect prediction")

# 10. Display the test image
plt.imshow(image.permute(1, 2, 0))
plt.title(
    f"Actual: {class_names[actual_label]}\n"
    f"Predicted: {class_names[predicted_label]}"
)
plt.axis("off")
plt.show()
