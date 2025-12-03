import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Feature extractor ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 14x14 → 14x14
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # 28x28 --conv1,2--> 28x28 --pool--> 14x14
        # 14x14 --conv3--> 14x14 --pool--> 7x7
        # 채널: 128, 크기: 7x7 → 128 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))   # (N, 32, 28, 28)
        x = F.relu(self.bn2(self.conv2(x)))   # (N, 64, 28, 28)
        x = self.pool(x)                      # (N, 64, 14, 14)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))   # (N, 128, 14, 14)
        x = self.pool(x)                      # (N, 128, 7, 7)

        x = self.classifier(x)                # (N, 10)
        return x



model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 18


for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    loss:Tensor
    X:Tensor
    labels:Tensor
    
    for X, labels in train_dataloader:
        X = X.to(device)
        labels = labels.to(device)
        
        logits = model(X)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss +=  loss.item() * X.size(0)
        
    
    print(f"{epoch} th loss : {(running_loss / len(train_dataset)):.6f}")
        
        
model.to(torch.device("cpu"))
model_info = {
    "model_state": model.state_dict(),
    "comments": "MNIST CNN Model"
}
torch.save(model_info, "models/mnist_cnn.pth")
