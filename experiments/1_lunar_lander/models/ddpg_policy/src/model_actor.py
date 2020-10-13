import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers_features = [ 
                                    nn.Linear(input_shape[0], hidden_count),
                                    nn.ReLU()
        ]

        self.layers_mu = [
                            nn.Linear(hidden_count, hidden_count),
                            nn.ReLU(),    
                            nn.Linear(hidden_count, outputs_count),
                            nn.Tanh()
        ]

        self.layers_var = [
                            nn.Linear(hidden_count, hidden_count),
                            nn.ReLU(),    
                            nn.Linear(hidden_count, outputs_count),
                            nn.Softplus()
        ]

        torch.nn.init.xavier_uniform_(self.layers_features[0].weight)
        
        torch.nn.init.xavier_uniform_(self.layers_mu[0].weight)
        torch.nn.init.uniform_(self.layers_mu[2].weight, -0.3, 0.3)

        torch.nn.init.xavier_uniform_(self.layers_var[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_var[2].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_mu = nn.Sequential(*self.layers_mu)
        self.model_mu.to(self.device)

        self.model_var = nn.Sequential(*self.layers_var)
        self.model_var.to(self.device)

        print(self.model_features)
        print(self.model_mu)
        print(self.model_var)
       
    def forward(self, state):
        features = self.model_features(state)
        mu       = self.model_mu(features)
        var      = self.model_var(features)

        y = mu + var*torch.randn(mu.shape)
        return y

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_features.state_dict(), path + "trained/model_features_actor.pt")
        torch.save(self.model_mu.state_dict(), path + "trained/model_mu_actor.pt")
        torch.save(self.model_var.state_dict(), path + "trained/model_var_actor.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_features.load_state_dict(torch.load(path + "trained/model_features_actor.pt", map_location = self.device))
        self.model_features.eval()  

        self.model_mu.load_state_dict(torch.load(path + "trained/model_mu_actor.pt", map_location = self.device))
        self.model_mu.eval()  

        self.model_var.load_state_dict(torch.load(path + "trained/model_var_actor.pt", map_location = self.device))
        self.model_var.eval()  
    
