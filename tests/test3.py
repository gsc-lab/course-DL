import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()  
    
    def forward(self, x): 
        print("Forward is invoked")   
        return x                      

module = MyModule()

input = torch.arange(1, 11)   # tensor([1, 2, 3, ..., 10])

module(input) # 매직메서드 __call__ 호출




