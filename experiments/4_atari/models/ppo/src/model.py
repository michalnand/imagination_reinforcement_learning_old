import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.layers = []
        
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)

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
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        fc_inputs_count = 64*(input_width//8)*(input_height//8)

        self.layers_features = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        ]

      
        self.layers_actor = [
            nn.Linear(fc_inputs_count, 256),
            nn.ReLU(),                      
            nn.Linear(256, outputs_count)
        ] 


        self.layers_critic = [
            nn.Linear(fc_inputs_count, 256),
            nn.ReLU(),                       
            nn.Linear(256, 1)  
        ]  

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_actor)):
            if hasattr(self.layers_actor[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_actor[i].weight)

        for i in range(len(self.layers_critic)):
            if hasattr(self.layers_critic[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_critic[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_critic = nn.Sequential(*self.layers_critic)
        self.model_critic.to(self.device)

        self.model_actor = nn.Sequential(*self.layers_actor)
        self.model_actor.to(self.device)

        print(self.model_features)
        print(self.model_critic)
        print(self.model_actor)

    def forward(self, state):
        features    = self.model_features(state)

        print(">>>> ", features.shape)

        logits      = self.model_actor(features)
        value       = self.model_critic(features)

        return logits, value

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")
        torch.save(self.model_actor.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
        self.model_actor.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_critic.eval() 
        self.model_actor.eval() 


    def get_activity_map(self, state):
 
        state_t     = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
        features    = self.model_features(state_t)
        features    = features.reshape((1, 64, 12, 12))

        upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

        features = upsample(features).sum(dim = 1)

        result = features[0].to("cpu").detach().numpy()

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()
        result = k*result + q
        
        return result


if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 9


    state   = torch.rand((batch_size, channels, height, width))

    model = Model((channels, height, width), actions_count)


    logits, value = model.forward(state)

    print(logits.shape, value.shape)

