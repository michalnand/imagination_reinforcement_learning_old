import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, kernels_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape 
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]   


        fc_input_height = input_shape[1]//(4*2*2)
        fc_input_width  = input_shape[2]//(4*2*2)

        self.layers = [
            nn.Conv2d(input_channels + outputs_count, kernels_count, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),

            nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(kernels_count, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            Flatten(),
            nn.Linear(16*fc_input_height*fc_input_width, 1)
        ]

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)
        print(self.model)
       
    def forward(self, state, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)

        model_input         = torch.cat([state, action_], dim = 1)
        reward_prediction   = self.model(model_input)
        
        return reward_prediction

    def save(self, path):
        print("saving ", path)
        torch.save(self.model.state_dict(), path + "trained/model_reward.pt")


    def load(self, path):
        print("loading ", path, " device = ", self.device) 

        self.model.load_state_dict(torch.load(path + "trained/model_reward.pt", map_location = self.device))
        self.model.eval() 
        

  
if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 7


    state   = torch.rand((batch_size, channels, height, width))
    action  = torch.rand((batch_size, actions_count))


    model = Model((channels, height, width), actions_count)


    y = model.forward(state, action)

    print(y.shape)

