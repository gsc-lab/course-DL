import torch
from torch import nn, optim, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# reproducibility
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# dataset (MNIST는 28x28 작은 이미지 → GPU 이득이 매우 작음)
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
).to(device=device)     # ★ 모델 파라미터를 GPU로 이동

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 15
for epoch in range(1, epochs + 1):
        
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward: GPU에서 행렬 연산 수행
        logits: Tensor = model(inputs)

        # loss 계산 (GPU 텐서 기반 연산)
        loss: Tensor = criterion(logits, targets)

        optimizer.zero_grad()

        # backward: GPU에서 gradient 계산 (행렬 미분 연산)
        loss.backward()

        # optimizer: 파라미터 업데이트 (파라미터가 GPU에 있으므로 GPU 연산)
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    avg_loss = train_loss / len(train_dataset)
    print(f"Epoch {epoch:2d} | loss: {avg_loss:.6f}")


model.to(torch.device("cpu"))
torch.save(model, "models/mnist.pth")

