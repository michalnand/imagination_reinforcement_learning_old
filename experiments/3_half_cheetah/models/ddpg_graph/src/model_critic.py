import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../..')
#sys.path.insert(0, '../../../../..')

import libs_layers

from torchviz import make_dot

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"

        self.dynamic_state_graph = libs_layers.DynamicGraphState(input_shape[0] + outputs_count, 0.001)

        inputs_count    = input_shape[0] + outputs_count

         
        self.gconv      = libs_layers.GConvSeq([inputs_count, hidden_count, hidden_count//2])

        self.output_layers = [
            nn.AvgPool1d(inputs_count),
            Flatten(),
            nn.Linear(hidden_count//2, 1)
        ]
        
        torch.nn.init.uniform_(self.output_layers[2].weight, -0.003, 0.003)

        self.output_model = nn.Sequential(*self.output_layers)
        self.output_model.to(self.device)
 

    def forward(self, state, action):

        x = torch.cat([state, action], dim = 1)

        #train graph estimator        
        for b in range(state.shape[0]):
            self.dynamic_state_graph.train(x[b].detach().to("cpu").numpy())
        
        edge_index = torch.from_numpy(self.dynamic_state_graph.edge_index)
        graph_x = self._graph_state_representation(x)

        #graph layers forward
        x = self.gconv(graph_x, edge_index)

        #channel last to channel first
        x = x.permute(0, 2, 1)
        
        #output layer forward
        return self.output_model(x)
        

    def _graph_state_representation(self, x):
        batch_size      = x.shape[0]
        result          = torch.zeros((batch_size,  x.shape[1] , x.shape[1]))

        am = torch.from_numpy(self.dynamic_state_graph.adjacency_matrix).to(self.device)
        for b in range(batch_size):
            x_          = torch.unsqueeze(x[b], 0)
            result[b]   = x_ * am

        return result

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.output_model.state_dict(), path + "/trained/model_critic_output.pt")
        self.gconv.save(path + "/trained/model_critic_gconv")

    def load(self, path):       
        print("loading from ", path)
        self.output_model.load_state_dict(torch.load(path + "/trained/model_critic_output.pt", map_location = self.device))
        self.output_model.eval()  
        self.gconv.load(path + "/trained/model_critic_gconv")

    

if __name__ == "__main__":
    state_shape     = (26, )
    actions_count   = 7
    batch_size      = 32

    model = Model(state_shape, actions_count)

    action   = torch.randn((batch_size, actions_count))
    state    = torch.randn((batch_size, ) + state_shape)

    q_values = model.forward(state, action)

    loss = q_values.mean()
    loss.backward()

    #make_dot(loss).render("model", format="png")

    #model.save("./")
    #model.load("./")

    print(q_values)
    print("program done")

    
    