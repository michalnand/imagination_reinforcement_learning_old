import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"


        self.layers = [ 
                        nn.Linear(2*input_shape[0] + outputs_count, hidden_count),
                        nn.ReLU(),
                        nn.Linear(hidden_count, hidden_count//2),
                        nn.ReLU(),            
                        nn.Linear(hidden_count//2, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.xavier_uniform_(self.layers[4].weight)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print(self.model)
       

    def forward(self, state, state_next, action):
        x = torch.cat([state, state_next, action], dim = 1)
        return self.model(x) 

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_entropy.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_entropy.pt", map_location = self.device))
        self.model.eval()  
    
