import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"

        self.model_features = nn.Sequential(
                                            nn.Linear(input_shape[0] + outputs_count, hidden_count),
                                            nn.ReLU()
        )

        self.model_update_gate = nn.Sequential(
                                            nn.Linear(hidden_count, hidden_count//2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_count//2, input_shape[0]),
                                            nn.Sigmoid()
        )

        self.model_state = nn.Sequential(
                                            nn.Linear(hidden_count, hidden_count//2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_count//2, input_shape[0])
        )


        self.model_reward = nn.Sequential(
                                            nn.Linear(hidden_count, hidden_count//2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_count//2, 1)
        ) 


        self.model_features.to(self.device)
        self.model_update_gate.to(self.device)
        self.model_state.to(self.device)
        self.model_reward.to(self.device)

        print(self.model_features)
        print(self.model_update_gate)
        print(self.model_state)
        print(self.model_reward)
        print("\n")


    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)

        features        = self.model_features(x)

        udpate_gate     = self.model_update_gate(features)
        state_candidate = self.model_state(features)

        state_prediction = (1.0 - udpate_gate)*state + udpate_gate*state_candidate

        return state_prediction, self.model_reward(features)

    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "trained/model_env_features.pt")
        torch.save(self.model_update_gate.state_dict(), path + "trained/model_env_update_gate.pt")
        torch.save(self.model_state.state_dict(), path + "trained/model_env_state.pt")
        torch.save(self.model_reward.state_dict(), path + "trained/model_env_reward.pt")


    def load(self, path):       
        self.model_features.load_state_dict(torch.load(path + "trained/model_env_features.pt", map_location = self.device))
        self.model_update_gate.load_state_dict(torch.load(path + "trained/model_env_update_gate.pt", map_location = self.device))
        self.model_state.load_state_dict(torch.load(path + "trained/model_env_state.pt", map_location = self.device))
        self.model_reward.load_state_dict(torch.load(path + "trained/model_env_reward.pt", map_location = self.device))

        self.model_features.eval() 
        self.model_update_gate.eval()
        self.model_state.eval() 
        self.model_reward.eval()
