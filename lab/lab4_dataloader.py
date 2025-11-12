import torch
from torch.utils.data import TensorDataset, DataLoader

# === 원본 데이터 준비 ===
data = torch.arange(6)           # tensor([0,1,2,3,4,5])
label = torch.arange(20, 26)     # tensor([20,21,22,23,24,25])

train_data = TensorDataset(data, label)

print("원본 데이터셋 샘플:")
for d, l in train_data:
    print(f"x={d}, y={l}")
print("-" * 40)

# === Case 1: Shuffle X, Batch X ===
loader_no_shuffle = DataLoader(train_data)
print("\nDataLoader : Shuffling ✗, Mini-Batching ✗")
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    for i, (x, y) in enumerate(loader_no_shuffle, start=1):
        print(f"  Sample {i}: x={x.item()}, y={y.item()}")

# === Case 2: Shuffle O, Batch X ===
loader_with_shuffle = DataLoader(train_data, shuffle=True)
print("\nDataLoader : Shuffling ✓, Mini-Batching ✗")
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    for i, (x, y) in enumerate(loader_with_shuffle, start=1):
        print(f"  Sample {i}: x={x.item()}, y={y.item()}")

# === Case 3: Shuffle X, Batch O ===
loader_batch = DataLoader(train_data, batch_size=2)
print("\nDataLoader : Shuffling ✗, Mini-Batching ✓")
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    for i, (x, y) in enumerate(loader_batch, start=1):
        print(f"  Batch {i}: x={x.tolist()}, y={y.tolist()}")

# === Case 4: Shuffle O, Batch O ===
loader_batch_shuffle = DataLoader(train_data, batch_size=2, shuffle=True)
print("\nDataLoader : Shuffling ✓, Mini-Batching ✓")
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    for i, (x, y) in enumerate(loader_batch_shuffle, start=1):
        print(f"  Batch {i}: x={x.tolist()}, y={y.tolist()}")
