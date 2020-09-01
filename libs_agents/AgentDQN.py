import numpy
import torch
from .ExperienceBuffer import *


class AgentDQN():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.exploration    = config.exploration
        self.gamma          = config.gamma
        self.update_frequency = config.update_frequency
        self.update_target_frequency         = config.update_target_frequency

        if hasattr(config, 'bellman_steps'):
            self.bellman_steps = config.bellman_steps
        else:
            self.bellman_steps = 1

       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.bellman_steps)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate, weight_decay=config.learning_rate*0.0001)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.state    = env.reset()

        self.iterations     = 0

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

        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        q_values    = self.model(state_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()

        self.action = self.choose_action_e_greedy(q_values, epsilon)

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)


        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()
            
            if self.iterations%self.update_target_frequency == 0:
                # update target network
                for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                    target_param.data.copy_(param.data)
     

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        self.iterations+= 1

        return self.reward, done
    
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

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
            q_target[i][action_idx]   = reward_sum + gamma_*torch.max(q_predicted_next[i])

        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2).mean() 
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer.step()

       
        
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    