import torch
from torch import nn

class Bar:
    def __setattr__(self, name, value):
        print(name, value)
        
        
obj = Bar()
obj.test = 2

print(obj.test)
