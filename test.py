import torch
from torch import nn

class BadFirst(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_slice = x[:, 0]
        print(f"x_slice: {x_slice}")
        return x_slice

if __name__ == "__main__":
    m = BadFirst().eval()
    x = torch.rand(10, 5)
    
    res = m(x) # this works
    torch.onnx.export(m, x, "badfirst.onnx") 