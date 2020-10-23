import torch
import torch.nn as nn
import numpy

class SkipInit2D(nn.Module):
    def __init__(self, in_channels):

        super(SkipInit2D, self).__init__()

        self.alpha  = nn.Parameter(torch.ones(1))
        self.beta   = nn.Parameter(torch.zeros(1))
        

    def forward(self, x): 
        batch_size  = x.shape[0]
        channels    = x.shape[1]
        height      = x.shape[2]
        width       = x.shape[3]

        alpha  = self.alpha.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(batch_size, channels, height, width)
        beta   = self.beta.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(batch_size, channels, height, width)

        return alpha*x + beta


if __name__ == "__main__":
    batch_size      = 64
    width           = 96
    height          = 96
    in_channels     = 16


    model           = SkipInit2D(in_channels)
    optimizer       = torch.optim.Adam(model.parameters(), lr=0.1)

    data_mean       = 0.47 #5.3
    data_variance   = 0.21 #7.9

    target_mean     = 0.0
    target_variance = 1.0


    for epoch in range(256):
        input  = data_variance*torch.randn(batch_size, in_channels, height, width) + data_mean

        output = model.forward(input)

        mean         = output.view(output.size(0), -1).mean()
        loss_mean    = ((target_mean - mean)**2).mean()

        std         = output.view(output.size(0), -1).std()
        loss_std    = ((target_variance - std)**2.0).mean()

        optimizer.zero_grad() 
        loss        = loss_mean + loss_std
        loss.backward()
        optimizer.step()    

        print("mean = ", mean.detach().to("cpu").numpy())
        print("std  = ", std.detach().to("cpu").numpy())
        print("parameters \n", model.alpha, "\n", model.beta)
        print("\n\n\n")


    
    