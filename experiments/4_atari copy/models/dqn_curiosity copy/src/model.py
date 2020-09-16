import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class NoiseLayer(torch.nn.Module):
    def __init__(self, inputs_count, init_range = 0.1):
        super(NoiseLayer, self).__init__()
        
        self.inputs_count   = inputs_count
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        w_initial   = init_range*torch.rand(self.inputs_count, device = self.device)
        
        self.w      = torch.nn.Parameter(w_initial, requires_grad = True)     
        self.distribution = torch.distributions.normal.Normal(0.0, 1.0)
 
    def forward(self, x):
        noise =  self.distribution.sample((self.inputs_count, )).detach().to(self.device)
        return x + self.w*noise

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]    

        ratio           = 2**4
        fc_inputs_count = 64*((fc_input_width)//ratio)*((fc_input_height)//ratio)
 

        self.layers_features = [ 
                        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
 
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        
                        Flatten(),
                        NoiseLayer(fc_inputs_count, 0.001)
                    ]


        self.layers_value = [
                            nn.Linear(fc_inputs_count, 128),
                            nn.ReLU(),                      
                            nn.Linear(128, 1) 
                        ]

        self.layers_advantage = [
                                nn.Linear(fc_inputs_count, 128),
                                nn.ReLU(),                      
                                nn.Linear(128, outputs_count)
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
        with torch.no_grad():
            x  = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)

            last_layer = len(self.layers_features) - 2
            for i in range(last_layer): 
                x = self.layers_features[i].forward(x)

            upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

            x = upsample(x)
            x = x.sum(dim = 1)
            result = x[0].to("cpu").detach().numpy()

            k = 1.0/(result.max() - result.min())
            q = 1.0 - k*result.max()
            result = k*result + q
            
            return result
