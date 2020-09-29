import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, imagination_rollouts, imagination_steps, hidden_count = 256, imagination_features_count = 64):
        super(Model, self).__init__()

        self.device = "cpu"

        features_size = input_shape[0] + outputs_count + 1
        self.imagination_features_layers =[
            nn.Conv2d(in_channels = features_size, out_channels=imagination_features_count, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            
            Flatten(),
            
            nn.Linear(imagination_rollouts*imagination_steps*imagination_features_count ,imagination_features_count),
            nn.ReLU()
        ]

        self.imagination_features_model = nn.Sequential(*self.imagination_features_layers) 
        self.imagination_features_model.to(self.device)

        self.layers = [ 
                        nn.Linear(input_shape[0] + outputs_count + imagination_features_count, hidden_count),
                        nn.ReLU(),
                        nn.Linear(hidden_count, hidden_count//2),
                        nn.ReLU(),            
                        nn.Linear(hidden_count//2, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.uniform_(self.layers[4].weight, -0.003, 0.003)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        
        print(self.imagination_features_model)
        print(self.model)
        
       
 
    def forward(self, state, action, im_state, im_actions, im_rewards):
        imagination_features_in = torch.cat([im_state, im_actions, im_rewards], dim = 3).permute(0, 3, 1, 2)
        imagination_features    = self.imagination_features_model(imagination_features_in)

        critic_input = torch.cat([state, action, imagination_features], dim = 1)
        return self.model(critic_input)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_critic.pt")
        torch.save(self.imagination_features_model.state_dict(), path + "trained/imagination_features_model_critic.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
        self.model.eval()  

        self.imagination_features_model.load_state_dict(torch.load(path + "trained/imagination_features_model_critic.pt", map_location = self.device))
        self.imagination_features_model.eval()  
    
