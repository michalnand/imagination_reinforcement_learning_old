import torch
import torch.nn as nn


class TrainerWeightWiseFC(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden = [], device = "cpu"):
        super(TrainerWeightWiseFC, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.device = device

        inputs  = input_features + output_features
        outputs = output_features

        self.layers = []

        for i in range(len(hidden)):
            outputs = hidden[i]
            self.layers.append(nn.Linear(inputs, outputs))
            self.layers.append(nn.ReLU())
            
            inputs = outputs
        
        weights_count = input_features*output_features
        self.layers.append(nn.Linear(inputs, weights_count))

        self.model = nn.Sequential(*self.layers)
        self.model.to(device) 

        print(self.model, "\n\n")

    def forward(self, input, output):
        x = torch.cat([input, output], dim = 1)
        y  = self.model(x)
        dw = y.reshape((-1, self.output_features, self.input_features))

        print("trainer ", torch.mean(self.layers[0].weight.data), torch.std(self.layers[0].weight.data))
 
        return dw




class HebbianLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, wa, wb, wc, wd):        
        output = input.mm(weight.t())
        
        if bias is not None:
            output+= bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, wa, wb, wc, wd)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, wa, wb, wc, wd = ctx.saved_tensors
        
        grad_weight = grad_bias = grad_wa = grad_wb = grad_wc = grad_wd = None

        grad_input = grad_output.mm(weight)
        
        grad_weight = grad_output.t().mm(input)
        

        if bias is not None:
            grad_bias = grad_output.sum(0)

        
        #TODO
        '''
        output = input.mm(weight.t())
        grad_trainer_output = grad_output.t().mm(input)*weight

        grad_weight_, grad_wa, grad_wb, grad_wc, grad_wd = HebbianLinearFunc.trainer_output(input, output, grad_trainer_output, wa, wb, wc, wd)

        grad_weight = 0.01*grad_weight_
        '''
        return grad_input, grad_weight, grad_bias, grad_wa, grad_wb, grad_wc, grad_wd

    @staticmethod
    def trainer_output(input, output, grad_output, wa, wb, wc, wd):
        in_a = output.t().mm(input)
        in_b = input.unsqueeze(1).repeat(1, output.shape[1], 1).sum(0)
        in_c = output.unsqueeze(2).repeat(1, 1, input.shape[1]).sum(0)
 
        result = 0 
        result+= in_a*wa
        result+= in_b*wb
        result+= in_c*wc
        result+= wd

        grad_wa = grad_output*in_a 
        grad_wb = grad_output*in_b
        grad_wc = grad_output*in_c
        grad_wd = grad_output 

        return result, grad_wa, grad_wb, grad_wc, grad_wd


class HebbianLinearLayer(nn.Module):
    def __init__(self, input_features, output_features, bias):
        super(HebbianLinearLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-1.0/output_features, 1.0/output_features)
        if self.bias is not None:
            self.bias.data.zero_()

        k = 1.0

        self.wa = nn.Parameter(torch.Tensor(output_features, input_features))
        self.wa.data.uniform_(-k/output_features, k/output_features)

        self.wb = nn.Parameter(torch.Tensor(output_features, input_features))
        self.wb.data.uniform_(-k/output_features, k/output_features)

        self.wc = nn.Parameter(torch.Tensor(output_features, input_features))
        self.wc.data.uniform_(-k/output_features, k/output_features)

        self.wd = nn.Parameter(torch.Tensor(output_features, input_features))
        self.wd.data.uniform_(-k/output_features, k/output_features)

    def forward(self, x):
        y = HebbianLinearFunc.apply(x, self.weight, self.bias, self.wa, self.wb, self.wc, self.wd)

        print(torch.max(self.wa).detach().numpy(), torch.mean(self.wa).detach().numpy(), torch.std(self.wa).detach().numpy())
        print(torch.max(self.wb).detach().numpy(),torch.mean(self.wb).detach().numpy(), torch.std(self.wb).detach().numpy())
        print(torch.max(self.wc).detach().numpy(),torch.mean(self.wc).detach().numpy(), torch.std(self.wc).detach().numpy())
        print(torch.max(self.wd).detach().numpy(),torch.mean(self.wd).detach().numpy(), torch.std(self.wd).detach().numpy())
        print(torch.max(self.weight).detach().numpy(),torch.mean(self.weight).detach().numpy(), torch.std(self.weight).detach().numpy())

        print("\n\n\n")

        return y




'''
class HebbianLinearLayer(nn.Module):
    def __init__(self, input_features, output_features, bias, Trainer):
        super(HebbianLinearLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-1.0/output_features, 1.0/output_features)
        if self.bias is not None:
            self.bias.data.zero_()

        self.trainer = Trainer(input_features, output_features, hidden=[input_features])

    def forward(self, x):
        dw = self.trainer(x.detach(), self._eval(x).detach())
        
        y  = HebbianLinearFunc.apply(x, self.weight, dw.mean(dim = 0), self.bias)
        return y

    def _eval(self, x):
        output = x.mm(self.weight.t())
        
        if self.bias is not None: 
            output+= self.bias.unsqueeze(0).expand_as(output)

        return output
'''




from torchviz import make_dot


if __name__ == "__main__":
    input_features    = 30
    output_features   = 10
    hidden_count      = 50
    batch_size        = 1

    model = nn.Sequential(
        HebbianLinearLayer(input_features, hidden_count, True),
        nn.ReLU(),
        HebbianLinearLayer(hidden_count, hidden_count, True),
        nn.ReLU(),
        HebbianLinearLayer(hidden_count, output_features, True)
    )


    input  = torch.randn(batch_size, input_features)

    output = model.forward(input)
    print(output.shape)

    make_dot(output).render("hebbian_linear_layer", format="png")
