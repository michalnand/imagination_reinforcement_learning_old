import torch
import torch.nn as nn
import numpy

class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias = True, sigma = 0.1):
        super(NoisyLinear, self).__init__()
        
        self.out_features   = out_features
        self.sigma          = sigma

        self.layer          = nn.Linear(in_features + out_features, out_features, bias = bias)
        torch.nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, x):
        noise = self.sigma*torch.randn((x.shape[0], self.out_features)).to(x.device)
        input = torch.cat([x, noise], dim = 1)
        
        return self.layer(input)


if __name__ == "__main__":
    in_features     = 6*6*64
    out_features    = 8


    layer = NoisyLinear(in_features, out_features)

    for j in range(4):
        input  = torch.randn(in_features).unsqueeze(0)
        for i in range(10):
            output = layer.forward(input)
            print(output)
        print("\n\n\n")
    