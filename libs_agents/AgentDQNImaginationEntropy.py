import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQNImaginationEntropy():
    def __init__(self, env, Model, ModelEnv, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.exploration    = config.exploration
        self.gamma          = config.gamma
        self.tau            = config.tau
        self.update_frequency = config.update_frequency

        if hasattr(config, 'bellman_steps'):
            self.bellman_steps = config.bellman_steps
        else:
            self.bellman_steps = 1

        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = config.imagination_steps
        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta
        self.env_learning_rate      = config.env_learning_rate
       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.bellman_steps)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.model_env          = ModelEnv.Model(self.state_shape, self.actions_count)
        self.optimizer_env      = torch.optim.Adam(self.model_env.parameters(), lr= config.env_learning_rate)

        self.entropy_alpha  = 0.01
        self.entropy_mean   = 0.0

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
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()

        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        q_values    = self.model(state_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()
 
        _, action_idx_np, _, _ = self._sample_action(state_t, self.epsilon)

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

        if show_activity:
            self._show_curiosity_activity(self.state, state_new, self.action)

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        
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

    def _show_curiosity_activity(self, state, state_next, action):
        size            = 400

        action_t        = torch.zeros(1,  dtype=int)
        action_t[0]     = action

        state_t              = torch.from_numpy(state).unsqueeze(dim=0).to(self.model_env.device)
        action_one_hot_t     = self._one_hot_encoding(action_t)
        state_prediction     = self.model_env(state_t, action_one_hot_t).squeeze().detach().to("cpu").numpy()

      
        space   = numpy.zeros((state.shape[1], 4)) 
        
        dif     =  (state_next - state_prediction)**2
        dif     =  (dif - dif.min())/(dif.max() - dif.min())


        image   =  numpy.hstack((state[0], space, state_prediction[0], space, dif[0]))

        image = cv2.resize(image, (3*size, size), interpolation = cv2.INTER_AREA)


        font                   = cv2.FONT_HERSHEY_SIMPLEX


        cv2.putText(image,'state', (10 + 0*size, int(size*0.98)), font, 1, (255,255,255), 2)

        cv2.putText(image,'prediction', (10 + 1*size, int(size*0.98)), font, 1, (255,255,255), 2)

        cv2.putText(image,'error', (10 + 2*size, int(size*0.98)), font, 1, (255,255,255), 2)


        cv2.imshow('curiosity activity', image)
        cv2.waitKey(1)
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #environment model state prediction
        action_one_hot_t    = self._one_hot_encoding(action_t)
        state_predicted_t   = self.model_env(state_t, action_one_hot_t)

        #env model loss
        env_loss = (state_next_t - state_predicted_t)**2
        env_loss = env_loss.mean()

        #update env model
        self.optimizer_env.zero_grad()
        env_loss.backward() 
        self.optimizer_env.step()


        #compute imagined states, use state_t as initial state
        states_imagined_t   = self._process_imagination(state_t, self.epsilon).detach()
        
        #compute entropy of imagined states
        entropy_t           = self._compute_entropy(states_imagined_t)

        #filtered entropy mean
        self.entropy_mean   = (1.0 - self.entropy_alpha)*self.entropy_mean + self.entropy_alpha*entropy_t.mean()

        
        '''
        #normalise entropy reward
        entropy       = (entropy_t - self.entropy_mean)/(self.entropy_mean + 0.000001)
        entropy       = torch.tanh(self.entropy_beta*(entropy**2))
        '''
        #normalise entropy, substract mean, squeeze into -1, 1 range
        entropy       = self.entropy_beta*torch.tanh(entropy_t - self.entropy_mean)

        #compute curiosity
        curiosity   = self.curiosity_beta*torch.tanh((state_next_t - state_predicted_t)**2).detach()
        curiosity   = curiosity.view(curiosity.size(0), -1).mean(dim = 1)

        #print(entropy)
        #print(reward_t[0].abs().mean(), entropy.abs().mean(), entropy.min(), entropy.max())
  

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

            reward_sum+= entropy[i]
            reward_sum+= curiosity[i]

            action_idx    = action_t[i]
            q_target[i][action_idx]   = reward_sum + gamma_*torch.max(q_predicted_next[i])

        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2).mean() 
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

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
        action_one_hot_np   = action_one_hot_t.detach().to("cpu").numpy()


        return action_idx_t, action_idx_np, action_one_hot_t, action_one_hot_np


    
    def _process_imagination(self, states_t, epsilon):
        batch_size  = states_t.shape[0]

        states_imagined_t      = torch.zeros((self.imagination_rollouts, batch_size, ) + self.state_shape ).to(self.model_env.device)

        for r in range(self.imagination_rollouts):
            states_imagined_t[r] = states_t.clone()

        for s in range(self.imagination_steps):
            #imagine rollouts
            for r in range(self.imagination_rollouts):

                _, _,action_one_hot_t, _= self._sample_action(states_imagined_t[r], epsilon)
                states_imagined_next_t  = self.model_env(states_imagined_t[r], action_one_hot_t)
                states_imagined_t[r]    = states_imagined_next_t.clone()
        

        #swap axis, target ordering : batch rollout state
        states_imagined_t = states_imagined_t.transpose(1, 0)

        return states_imagined_t


    def _compute_entropy(self, states_t, eps = 0.001):
        batch_size  = states_t.shape[0]

        result      = torch.zeros(batch_size).to(self.model_env.device)
        
        for b in range(batch_size):
            v           = torch.var(states_t[b], dim = 0)
            vt          = v[v > eps]
 
            if vt.size()[0] > 0: 
                result[b]   = vt.mean()
 
        return result
    
    def _one_hot_encoding(self, input):
        size = len(input)
        result = torch.zeros((size, self.actions_count))
        
        result[range(size), input] = 1.0
        
        return result.to(self.model.device)

  

    def save(self, save_path):
        self.model.save(save_path)
        self.model_env.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
        self.model_env.load(save_path)
    