import numpy
import torch

from torch.distributions import Categorical

from .PolicyBufferContinuous import *

class AgentPPOContinuous():
    def __init__(self, envs, Model, Config):
        self.envs = envs

        config = Config.Config()

        self.envs_count = len(self.envs)
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.eps_clip = config.eps_clip
        self.batch_size     = config.batch_size
        self.training_epochs    = config.training_epochs
       
        self.state_shape = self.envs[0].observation_space.shape
        self.actions_count     = self.envs[0].action_space.shape[0]

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_old      = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)


        self.buffer = PolicyBufferContinuous(self.envs_count, self.batch_size, self.state_shape, self.actions_count, self.model.device)

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
        

        mu, var, value   = self.model.forward(state_t)

        action_t = self._get_action(mu, var)
            
        self.states[env_id], reward, done, _ = self.envs[env_id].step(action_t.to("cpu").numpy())
        
        if self.enabled_training:
            self.buffer.add(env_id, state_t.squeeze(0), mu.squeeze(0), var.squeeze(0), value.squeeze(0), action_t, reward, done)
           
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
        

        if self.buffer.size() > self.batch_size-1:  

            self.buffer.calc_discounted_reward(self.gamma)

            for epoch in range(self.training_epochs):
                loss = 0
                for env_id in range(self.envs_count):
                    loss+= self._compute_loss(env_id)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step() 

            
                self.model_old.load_state_dict(self.model.state_dict())
            
            #clear batch buffer
            self.buffer.clear()

            

        self.iterations+= 1

        return reward, done
            
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
        self.model_old.load_state_dict(self.model.state_dict())

   
    def _get_action(self, mu, var):
        dist     = torch.distributions.Normal(mu, var)
        
        action_t = dist.sample()[0].detach().clamp(-1.0, 1.0)

        return action_t

    def _calc_log_prob(self, mu, var, action):
        result = -((action - mu)**2) / (2.0*var.clamp(min = 0.001))
        result+= -torch.log(torch.sqrt(2.0*3.141592654*var))

        return result


    def _compute_loss(self, env_id):
        
        target_values_b = torch.FloatTensor(self.buffer.discounted_rewards[env_id]).to(self.model.device).detach()

        mu_old_b      = self.buffer.mu_b[env_id].detach()
        var_old_b     = self.buffer.var_b[env_id].detach()

        
        state_t = self.buffer.states_b[env_id].detach().clone()

        mu_b, var_b, values   = self.model.forward(state_t)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - values)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss 
        '''
        log_probs_      = self._calc_log_prob(mu_b, var_b, self.buffer.actions_b[env_id])
        log_probs_old_  = self._calc_log_prob(mu_old_b, var_old_b, self.buffer.actions_b[env_id])

     
        # Finding Surrogate Loss:
        advantage   = (target_values_b - values).detach()

        
        ratios = torch.exp(log_probs_ - log_probs_old_)
        surr1 = ratios*advantage
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantage
        
        loss_policy = -torch.min(surr1, surr2) 
        loss_policy = loss_policy.mean()
    
        '''
        compute entropy loss, to avoid greedy strategy
        '''
        loss_entropy = -(1.0 + torch.log(2.0*3.141592654*var_b))*0.5
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        #print(loss_value, loss_policy, loss_entropy)

        #train network, with gradient cliping
        loss = loss_value + loss_policy + loss_entropy

        return loss

