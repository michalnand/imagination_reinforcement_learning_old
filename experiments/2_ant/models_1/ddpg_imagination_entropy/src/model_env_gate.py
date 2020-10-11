import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"


        self.layers_features = [ 
                                    nn.Linear(input_shape[0] + outputs_count, hidden_count),
                                    nn.ReLU()
        ]

        self.layers_gate = [
                                nn.Linear(hidden_count, hidden_count//2),
                                nn.ReLU(),
                                nn.Linear(hidden_count//2, input_shape[0]),
                                nn.Sigmoid()
        ]

        self.layers_output = [
                                nn.Linear(hidden_count, hidden_count//2),
                                nn.ReLU(),
                                nn.Linear(hidden_count//2, input_shape[0])
        ]


                     

        torch.nn.init.xavier_uniform_(self.layers_features[0].weight)

        torch.nn.init.xavier_uniform_(self.layers_gate[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_gate[2].weight)

        torch.nn.init.xavier_uniform_(self.layers_output[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_output[2].weight)

 
        self.model_features = nn.Sequential(*self.layers_features)
        self.model_gate     = nn.Sequential(*self.layers_gate)
        self.model_output   = nn.Sequential(*self.layers_output)

        self.model_features.to(self.device)
        self.model_gate.to(self.device)
        self.model_output.to(self.device)

        print(self.model_features)
        print(self.model_gate)
        print(self.model_output)
       

    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        
        features = self.model_features(x)
        gate     = self.model_gate(features)
        output   = self.model_output(features)

        y = (1.0 - gate)*state + gate*output
        return y

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_features.state_dict(), path + "trained/model_env_features.pt")
        torch.save(self.model_gate.state_dict(), path + "trained/model_env_gate.pt")
        torch.save(self.model_output.state_dict(), path + "trained/model_env_output.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_features.load_state_dict(torch.load(path + "trained/model_env_features.pt", map_location = self.device))
        self.model_gate.load_state_dict(torch.load(path + "trained/model_env_gate.pt", map_location = self.device))
        self.model_output.load_state_dict(torch.load(path + "trained/model_env_outmodel_output.pt", map_location = self.device))

        self.model_features.eval()
        self.model_gate.eval()
        self.model_output.eval()  
    
