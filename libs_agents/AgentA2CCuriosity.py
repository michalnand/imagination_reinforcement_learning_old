import numpy
import torch

from torch.distributions import Categorical
from .CuriosityModule import *

from .PolicyBuffer import *

class AgentA2CCuriosity():
    def __init__(self, envs, Model, ModelCuriosity, Config):
        self.envs = envs

        config = Config.Config()

        self.envs_count = len(self.envs)
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.batch_size     = config.batch_size
       
        self.state_shape = self.envs[0].observation_space.shape
        self.actions_count     = self.envs[0].action_space.n

        
        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.curiosity_update_steps = config.curiosity_update_steps
        self.curiosity_beta = config.curiosity_beta
        self.curiosity_module = CuriosityModule(ModelCuriosity, self.state_shape, self.actions_count, config.curiosity_learning_rate, config.curiosity_buffer_size)

        self.buffer = PolicyBuffer(self.envs_count, self.batch_size, self.state_shape, self.actions_count, self.model.device)

        self.states = []

        for env in self.envs:
            self.states.append(env.reset())

        self.enable_training()

        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
        
    def process_env(self, env_id = 0):
        state_t   = torch.tensor(self.states[env_id], dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)

        state_ = self.states[env_id].copy()

        logits, value   = self.model.forward(state_t)

        action_t = self._get_action(logits)
            
        self.states[env_id], reward, done, _ = self.envs[env_id].step(action_t.item())
        

        if self.enabled_training:
            self.buffer.add(env_id, state_t.squeeze(0), logits.squeeze(0), value.squeeze(0), action_t.item(), reward, done)

            if env_id == 0:
                self.curiosity_module.add(state_, action_t.item(), reward, done)

        if done:
            self.states[env_id] = self.envs[env_id].reset()

        return reward, done
        
    
    def main(self):
        reward = 0
        done = False
        for env_id in range(self.envs_count):
            tmp, tmp_done = self.process_env(env_id)
            if env_id == 0:
                reward = tmp
                done = tmp_done
       

        if self.enabled_training and self.iterations%self.curiosity_update_steps == 0:
            self.curiosity_module.train()
        

        if self.buffer.size() > self.batch_size-1:  
            self.buffer.calc_discounted_reward(self.gamma)

            loss = 0
            
            for env_id in range(self.envs_count):
                
                curiosity, _ = self.curiosity_module.eval(self.buffer.states_prev_b[env_id], self.buffer.states_b[env_id], self.buffer.actions_b[env_id])
                curiosity = torch.clamp(curiosity*self.curiosity_beta, 0.0, 1.0).unsqueeze(1)

                loss+= self._compute_loss(env_id, curiosity)


            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step() 

            #clear batch buffer
            self.buffer.clear()

        self.iterations+= 1

        return reward, done
            
    def save(self, save_path):
        self.model.save(save_path)
        self.curiosity_module.save(save_path + "trained/")

    def load(self, save_path):
        self.model.load(save_path)
        self.curiosity_module.load(save_path + "trained/")
    
   
    def _get_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t
      

    def _compute_loss(self, env_id, curiosity):
        
        target_values_b = torch.FloatTensor(self.buffer.discounted_rewards[env_id]).to(self.model.device)
        target_values_b = target_values_b + curiosity
       
        probs     = torch.nn.functional.softmax(self.buffer.logits_b[env_id], dim = 1)
        log_probs = torch.nn.functional.log_softmax(self.buffer.logits_b[env_id], dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - self.buffer.values_b[env_id])**2
        loss_value = loss_value.mean()


        ''' 
        compute actor loss 
        L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A 
        '''
        advantage   = (target_values_b - self.buffer.values_b[env_id]).detach()
        loss_policy = -log_probs[range(len(log_probs)), self.buffer.actions_b[env_id]]*advantage
        loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        #train network, with gradient cliping
        loss = loss_value + loss_policy + loss_entropy
        return loss

