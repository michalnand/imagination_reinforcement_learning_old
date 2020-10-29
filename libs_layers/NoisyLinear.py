import torch
import torch.nn as nn
import numpy

class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma = 0.1):
        super(NoisyLinear, self).__init__()
        
        self.out_features   = out_features
        self.in_features    = in_features
        self.sigma          = sigma

        self.weight  = nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias  = nn.Parameter(torch.zeros(out_features))
 

        self.weight_noise  = nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight_noise)

        self.bias_noise  = nn.Parameter(0.01*torch.randn(out_features)) 


    def forward(self, x): 
        weight_noise    = self.sigma*torch.randn((self.in_features, self.out_features)).to(x.device).detach()
        bias_noise      = self.sigma*torch.randn((self.out_features)).to(x.device).detach()

        weight_noised   = self.weight + self.weight_noise*weight_noise
        bias_noised     = self.bias   + self.bias_noise*bias_noise 

        return x.matmul(weight_noised) + bias_noised


if __name__ == "__main__":
    in_features     = 6*6*64
    out_features    = 8

    layer = NoisyLinear(in_features, out_features)

    for j in range(4):
        input  = torch.randn(in_features)
        for i in range(10):
            output = layer.forward(input)
            print(output)
        print("\n\n\n")
    