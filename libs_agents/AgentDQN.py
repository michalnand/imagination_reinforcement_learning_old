import numpy
import torch
from .ExperienceBufferPrioritized import *

import cv2


class AgentDQN():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        self.tau                = config.tau
        self.update_frequency   = config.update_frequency
        self.prioritized_buffer        = config.prioritized_buffer
        
        self.bellman_steps = config.bellman_steps
        
       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBufferPrioritized(config.experience_replay_size, self.bellman_steps)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.state    = env.reset()

        self.iterations     = 0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
             
        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        q_values    = self.model(state_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()
 
        action_idx_np, _,  = self._sample_action(state_t, epsilon)

        self.action = action_idx_np[0]

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)


        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()
            
            #soft update target network
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return self.reward, done
    
    def _show_activity(self, state, alpha = 0.6):
        activity_map    = self.model.get_activity_map(state)
        activity_map    = numpy.stack((activity_map,)*3, axis=-1)*[0, 0, 1]

        state_map    = numpy.stack((state[0],)*3, axis=-1)
        image        = alpha*state_map + (1.0 - alpha)*activity_map

        image        = (image - image.min())/(image.max() - image.min())

        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
        cv2.imshow('state activity', image)
        cv2.waitKey(1)
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size):
            gamma_        = self.gamma

            reward_sum = 0.0
            for i in range(self.bellman_steps):
                if done_t[j][i]:
                    gamma_ = 0.0
                reward_sum+= reward_t[j][i]*(gamma_**j)

            action_idx    = action_t[j]
            q_target[j][action_idx]   = reward_sum + gamma_*torch.max(q_predicted_next[j])

        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2)
        loss_ = loss.mean(dim=1)
        loss  = loss.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

        if self.prioritized_buffer:
            self.experience_replay.update_loss(loss_.detach().to("cpu").numpy())

    
    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t  = self.model(state_t)

        action_idx_t     = torch.zeros(batch_size).to(self.model.device)

        action_one_hot_t = torch.zeros((batch_size, self.actions_count)).to(self.model.device)

        #e-greedy strategy
        for b in range(batch_size):
            action = torch.argmax(q_values_t[b])
            if numpy.random.random() < epsilon:
                action = numpy.random.randint(self.actions_count)

            action_idx_t[b]                 = action
            action_one_hot_t[b][action]     = 1.0
        
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    