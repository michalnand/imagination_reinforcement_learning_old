import torch
import torch.nn as nn
import numpy

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma = 0.1, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        self.in_features    = in_features
        self.out_features   = out_features
        self.sigma          = sigma


        self.weight_noise = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_noise = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_noise', None)

        r = numpy.sqrt(1.0/(self.in_features + self.out_features))
        
        self.weight_noise.data.uniform_(-r, r)
        
        if self.bias_noise is not None:
            self.bias_noise.data.uniform_(-0.0001, 0.0001)

    def forward(self, x):
        y       = x.mm(self.weight.t()) + self.bias

        noise_x =  self.sigma*torch.randn(x.shape)
        noise_b =  self.sigma*torch.randn(y.shape)

        y_noise = noise_x.mm(self.weight_noise.t()) + self.bias_noise*noise_b
        y_noise = y_noise.to(x.device)

        return y + y_noise

    def to(self, device):
        super(NoisyLinear, self).__init__(device)

        self.weight_noise.to(self.device)

        if self.bias_noise is not None:
            self.bias_noise.to(self.device)

        print("TO ", self.device)



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