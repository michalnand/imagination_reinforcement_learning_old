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
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(channels))
        
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[3].weight)

        self.model = nn.Sequential(*self.layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.model(x) 
        return self.activation(y + x)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, kernels_count = 32):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape 
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]   

        input_kernel_size = 4

        fc_input_height = input_shape[1]//(input_kernel_size)
        fc_input_width  = input_shape[2]//(input_kernel_size)


        self.conv = nn.Sequential(
                                    nn.Conv2d(input_channels + outputs_count, kernels_count, kernel_size=input_kernel_size, stride=input_kernel_size, padding=1),
                                    nn.ReLU(),
                                    ResidualBlock(kernels_count),
                                    ResidualBlock(kernels_count),
                                    ResidualBlock(kernels_count)
        )

        self.deconv = nn.Sequential(
                                    nn.Conv2d(kernels_count, kernels_count, kernel_size=1, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(kernels_count, 1, kernel_size=input_kernel_size, stride=input_kernel_size, padding=0),
        )

        self.reward = nn.Sequential(
                                        nn.Conv2d(kernels_count, 16, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Linear(fc_input_height*fc_input_width*16, 1)
        ) 

        self.conv.to(self.device) 
        self.deconv.to(self.device) 
        self.reward.to(self.device) 

        print(self.conv)
        print(self.deconv)  
        print(self.reward)                      

    def forward(self, state, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)

        model_input      = torch.cat([state, action_], dim = 1)
        conv_output      = self.conv(model_input)

        frame_prediction       = self.deconv(conv_output) 
        reward_prediction      = self.reward(conv_output)

        frames_count            = state.shape[1]
        state_tmp               = torch.narrow(state, 1, 0, frames_count-1)
        frame_prediction        = frame_prediction + torch.narrow(state, 1, 0, 1)
        observation_prediction  = torch.cat([frame_prediction, state_tmp], dim = 1)
  
        return observation_prediction, reward_prediction

    def save(self, path):
        print("saving ", path)

        torch.save(self.conv.state_dict(), path + "trained/model_env_conv.pt")
        torch.save(self.deconv.state_dict(), path + "trained/model_env_deconv.pt")
        torch.save(self.reward.state_dict(), path + "trained/model_env_reward.pt")


    def load(self, path):
        print("loading ", path, " device = ", self.device) 

        self.conv.load_state_dict(torch.load(path + "trained/model_env_conv.pt", map_location = self.device))
        self.deconv.load_state_dict(torch.load(path + "trained/model_env_deconv.pt", map_location = self.device))
        self.reward.load_state_dict(torch.load(path + "trained/model_env_reward.pt", map_location = self.device))

        self.conv.eval() 
        self.deconv.eval() 
        self.reward.eval() 


  
if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 7


    state   = torch.rand((batch_size, channels, height, width))
    action  = torch.rand((batch_size, actions_count))


    model = Model((channels, height, width), actions_count)


    y, r = model.forward(state, action)

    print(y.shape)
    print(r.shape)

    model.save("./")
