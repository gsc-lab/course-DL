import torch
from torch import nn, optim, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# reproducibility
torch.manual_seed(42)

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataset
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 30),
    nn.ReLU(),
    nn.Linear(30, 120),
    nn.ReLU(),
    nn.Linear(120, 10),
)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# train loop
epochs = 5
for epoch in range(1, epochs + 1):
    train_loss = 0.0
    for inputs, targets in train_loader:
        logits: Tensor = model(inputs)
        loss: Tensor = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    avg_loss = train_loss / len(train_dataset)
    print(f"Epoch {epoch:2d} | loss: {avg_loss:.6f}")

# inference
with torch.no_grad():
    correct = 0
    for images, targets in test_loader:
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()

    accuracy = correct / len(test_dataset)
    print(f"Test accuracy: {accuracy:.4f}")

