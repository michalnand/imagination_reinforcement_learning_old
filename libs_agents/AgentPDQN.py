import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentPDQN():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                = config.tau

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
        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        logits, _   = self.model(state_t)

        self.action = self._get_action(logits)

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()
            
        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        self.iterations+= 1
        return self.reward, done
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #q values, state now, state next
        logits, q_predicted     = self.model.forward(state_t)
        _, q_predicted_next     = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = torch.zeros(self.batch_size).to(self.model.device)
        for i in range(self.batch_size):
            gamma_        = self.gamma

            reward_sum = 0.0
            for j in range(self.bellman_steps):
                if done_t[j][i]:
                    gamma_ = 0.0
                reward_sum+= reward_t[j][i]*(gamma_**j)

            q_target[i]   = reward_sum + gamma_*q_predicted_next[i]

        #train policy DQN model

        #value loss, MSE
        loss_value  = (q_target.detach() - q_predicted)**2
        loss_value  = loss_value.mean()
        
        probs     = torch.nn.functional.softmax(logits, dim = 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim = 1)

        #policy loss,   
        #L = -log(pi(s, a))*Q
        loss_policy = -probs[range(self.batch_size), action_t]*q_predicted.detach()
        loss_policy = loss_policy.mean()


        #entropy loss, to avoid greedy strategy
        #L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = 0.01*loss_entropy.mean()

        
        #print(loss_value, loss_policy, loss_entropy)

         
        loss = loss_value + loss_policy + loss_entropy

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

        #seoft update target network
        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
    
        
    def _get_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample().detach().to("cpu").numpy()

        return action_t

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    