import torch
import torch.nn as nn
import numpy

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True, sigma = 1.0):

        super(NoisyLinear, self).__init__()

        self.sigma = sigma
        self.layer = nn.Linear(in_features*2, out_features, bias = bias)


    def forward(self, x):
        noise_x =  self.sigma*torch.randn(x.shape).to(x.device)
        x_input = torch.cat([x, noise_x], dim = 1)

        return self.layer(x_input) 


if __name__ == "__main__":
    in_features     = 1024
    out_features    = 5


    layer = NoisyLinear(in_features, out_features)

    for j in range(4):
        input  = torch.randn(in_features).unsqueeze(0)
        for i in range(10):
            output = layer.forward(input)
            print(output)
        print("\n\n\n")
    