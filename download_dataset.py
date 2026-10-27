
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Convert images into PyTorch tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Download the training dataset
train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download the testing dataset
test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("Dataset downloaded successfully!")
print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print("Classes:", train_data.classes)

# Display a sample image
image, label = train_data[0]

# Undo normalization for display
mean = image.new_tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
std = image.new_tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
image = image * std + mean
image = image.permute(1, 2, 0).clamp(0, 1)

plt.imshow(image)
plt.title("Class: " + train_data.classes[label])
plt.axis("off")
plt.show()
