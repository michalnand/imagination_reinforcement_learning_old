import numpy
import torch
from .ExperienceBuffer import *

from .CuriosityModule import *


class AgentDQNCuriosity():
    def __init__(self, env, ModelDQN, ModelCuriosity, Config):
        self.env = env

        self.action = 0

        config = Config.Config()

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.gamma          = config.gamma
        self.curiosity_beta = config.curiosity_beta
        self.update_frequency = config.update_frequency
        self.update_target_frequency = config.update_target_frequency

        if hasattr(config, 'bellman_steps'):
            self.bellman_steps = config.bellman_steps
        else:
            self.bellman_steps = 1


        self.iterations     = 0


        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.bellman_steps)

        self.model_dqn          = ModelDQN.Model(self.state_shape, self.actions_count)
        self.model_dqn_target   = ModelDQN.Model(self.state_shape, self.actions_count)
        self.optimizer_dqn      = torch.optim.Adam(self.model_dqn.parameters(), lr= config.learning_rate, weight_decay=config.learning_rate*0.0001)

        for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
            target_param.data.copy_(param.data)

        self.curiosity_module = CuriosityModule(ModelCuriosity, self.state_shape, self.actions_count, config.curiosity_learning_rate, self.experience_replay, False)

        self.state    = env.reset()
        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
        
        state_t     = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()
        q_values    = self.model_dqn(state_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()


        self.action = self.choose_action_e_greedy(q_values, epsilon)

        state_new, self.reward, done, self.info = self.env.step(self.action)

        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)
       
        if self.enabled_training and (self.iterations > self.experience_replay.size) and self.iterations%self.update_frequency == 0:
            self.train_model()
            self.curiosity_module.train()
            
            if self.iterations%self.update_target_frequency == 0:
                # update target network
                for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
                    target_param.data.copy_(param.data)
     
        
        self.state = state_new
            
        if done:
            self.env.reset()


        self.iterations+= 1
        return self.reward, done
        

    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_dqn.device)            

        #compute curiosity
        curiosity_t, _  = self.curiosity_module.eval(state_t, state_next_t, action_t)
        curiosity_t  = torch.clamp(self.curiosity_beta*curiosity_t, 0.0, 1.0)   


        #q values, state now, state next
        q_predicted      = self.model_dqn.forward(state_t)
        q_predicted_next = self.model_dqn_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for i in range(self.batch_size):
            gamma_        = self.gamma

            reward_sum = 0.0
            for j in range(self.bellman_steps):
                if done_t[j][i]:
                    gamma_ = 0.0
                reward_sum+= reward_t[j][i]*(gamma_**j)

            action_idx    = action_t[i]
            q_target[i][action_idx]   = curiosity_t[i] + reward_sum + gamma_*torch.max(q_predicted_next[i])

        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2).mean() 
        self.optimizer_dqn.zero_grad()
        loss.backward()
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_dqn.step()
 
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self, save_path):
        self.model_dqn.save(save_path)
        self.curiosity_module.save(save_path) 

    def load(self, save_path):
        self.model.load(save_path)
        self.curiosity_module.load(save_path)     