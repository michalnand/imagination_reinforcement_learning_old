import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../..')

import libs_layers




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.layers = []
        
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=2, stride=1, padding=0))
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=2, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(channels))
        
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[3].weight)

        self.model = nn.Sequential(*self.layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.model(x) 
        return self.activation(y + x)




class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]    

        kernels_count   = [input_channels, 32, 32, 64, 64]
        residual_count  = [2, 2, 2, 2]

        ratio           = 2**(len(kernels_count) - 1)
        fc_inputs_count = kernels_count[-1]*((fc_input_width)//ratio)*((fc_input_height)//ratio)

        self.layers_features = []

        for i in range(len(kernels_count)-1):
            self.layers_features.append(nn.Conv2d(kernels_count[i], kernels_count[i+1], kernel_size=3, stride=2, padding=1))
            self.layers_features.append(nn.ReLU()) 

            for j in range(residual_count[i]):
                self.layers_features.append(ResidualBlock(kernels_count[i+1]))

        self.layers_features.append(Flatten())

        self.layers_value = [
                            libs_layers.NoisyLinear(fc_inputs_count, 128),
                            nn.ReLU(),                      
                            libs_layers.NoisyLinear(128, 1) 
        ]

        self.layers_advantage = [
                                libs_layers.NoisyLinear(fc_inputs_count, 128),
                                nn.ReLU(),                      
                                libs_layers.NoisyLinear(128, outputs_count)
        ] 

  
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_advantage = nn.Sequential(*self.layers_advantage)
        self.model_advantage.to(self.device)

        print(self.model_features)
        print(self.model_value)
        print(self.model_advantage)

    def forward(self, state):
        features    = self.model_features(state)

        value       = self.model_value(features)
        advantage   = self.model_advantage(features)

        result = value + advantage - advantage.mean()
        return result

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_value.state_dict(), path + "trained/model_value.pt")
        torch.save(self.model_advantage.state_dict(), path + "trained/model_advantage.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "trained/model_value.pt", map_location = self.device))
        self.model_advantage.load_state_dict(torch.load(path + "trained/model_advantage.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_value.eval() 
        self.model_advantage.eval() 


    def get_activity_map(self, state):
 
        state_t     = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
        features    = self.model_features(state_t)
        features    = features.reshape((1, 64, 6, 6))

        upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

        features = upsample(features).sum(dim = 1)

        result = features[0].to("cpu").detach().numpy()

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()
        result = k*result + q
        
        return result
