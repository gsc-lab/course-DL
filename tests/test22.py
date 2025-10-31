

import torch

from torch import nn


x = torch.randint(1, 10, (1, 2), dtype=torch.float16)
w = torch.rand(2, 3, dtype=torch.float16)

z = x @ w

print(z)