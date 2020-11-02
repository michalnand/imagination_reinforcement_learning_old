import numpy
import torch

from torch.distributions import Categorical

from .PolicyBufferContinuous import *

class AgentA2CContinuous():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()
 
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.batch_size         = config.batch_size
        self.episodes_to_train  = config.episodes_to_train
       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBufferContinuous(self.batch_size, self.state_shape, self.actions_count, self.model.device)

        self.state = env.reset()

        self.enable_training()
        
        self.iterations = 0
        self.passed_episodes = 0



    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

   
    def main(self):
        state_t   = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        
        mu, var, value   = self.model.forward(state_t)

        action_t = self._sample_action(mu, var)

        action = action_t.squeeze(0).detach().to("cpu").numpy()
            
        self.state, reward, done, _ = self.env.step(action)
        
        if self.enabled_training:
            self.policy_buffer.add(state_t.squeeze(0), value.squeeze(0), action_t.squeeze(0), mu.squeeze(0), var.squeeze(0), reward, done)

            if done:
                self.passed_episodes+= 1

            if self.passed_episodes >= self.episodes_to_train or self.policy_buffer.is_full():
                self.policy_buffer.cut_zeros()
                self._train()
                self.policy_buffer.clear()   
                
                self.passed_episodes = 0

        if done:
            self.state = self.env.reset()

            if hasattr(self.model, "reset"):
                self.model.reset()

        self.iterations+= 1

        return reward, done
    

    
     
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    
   
  
    def _sample_action(self, mu, var):
        sigma    = torch.sqrt(var)
        dist     = torch.distributions.Normal(mu, sigma)
        
        action_t = dist.sample()[0].detach().clamp(-1.0, 1.0)

        return action_t

    def _calc_log_prob(self, mu, var, action):
        result = -((action - mu)**2) / (2.0*var.clamp(min = 0.001))
        result+= -torch.log(torch.sqrt(2.0*numpy.pi*var))

        return result

    def _train(self):
        self.policy_buffer.compute_returns(self.gamma, normalise=False) 
        
        loss = self._compute_loss()

        self.optimizer.zero_grad()        
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step() 

        
 

    def _compute_loss(self):
        returns = self.policy_buffer.returns_b.unsqueeze(1)

        log_probs   = self._calc_log_prob(self.policy_buffer.actions_mu_b, self.policy_buffer.actions_var_b, self.policy_buffer.actions_b)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (returns - self.policy_buffer.values_b)**2
        loss_value = loss_value.mean()

        '''
        compute actor loss 
        L = log(pi(s, a, mu, var))*(T - V(s)) = _calc_log_prob(mu, var, action)*A
        '''
        advantage   = (returns - self.policy_buffer.values_b).detach()

        loss_policy = -log_probs*advantage
        loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        ''' 
        loss_entropy = -0.5*torch.log(2.0*numpy.pi*self.policy_buffer.actions_var_b)
        loss_entropy = loss_entropy.mean()


        loss = loss_value + loss_policy + loss_entropy

        return loss

