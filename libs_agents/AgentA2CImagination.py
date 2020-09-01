import numpy
import torch

from torch.distributions import Categorical
from .ImaginationModule import *

from .PolicyBuffer import *

class AgentA2CImagination():
    def __init__(self, env, Model, ModelEnv, Config):
        self.env = env

        config = Config.Config()
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.batch_size     = config.batch_size
        self.rollouts       = config.rollouts
        
        self.model_env_update_steps     = config.model_env_update_steps

        self.state_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.imagination_module = ImaginationModule(ModelEnv, self.state_shape, self.actions_count, config.model_env_learning_rate, config.model_env_buffer_size)

        self.buffer = PolicyBuffer(self.rollouts, self.batch_size, self.state_shape, self.actions_count, self.model.device)

        self.state = self.env.reset()

        self.enable_training()

        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
        
    
    def main(self):

        state_t         = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        logits, value   = self.model.forward(state_t)
        action          = self._get_action(logits)
       
        state_new, reward, done, _ = self.env.step(action.item())

        if self.enabled_training:
            self.imagination_module.add(self.state, action.item(), reward, done)
            
            if self.iterations%self.model_env_update_steps == 0:
                self.imagination_module.train()

            if self.iterations%1024 == 0 and self.iterations > 4096:
                self.imagination_process()

                self.buffer.calc_discounted_reward(self.gamma)

                loss = 0.0
                for r in range(self.rollouts):
                    loss+= self._compute_loss(r)

                #update model
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step() 

                #clear batch buffer
                self.buffer.clear()

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()


        self.iterations+= 1

        return reward, done



    def imagination_process(self):
        
        states_initial_t   = self.imagination_module.get_state(self.rollouts).clone()
        for rollout in range(self.rollouts):
            
            state_t  = states_initial_t[rollout].clone().unsqueeze(0)

            
            for n in range(self.batch_size):
                
                logits, value   = self.model.forward(state_t)
                action          = self._get_action(logits)

                state_, reward = self.imagination_module.eval(state_t.detach(), action.item())

                if n == self.batch_size-1:
                    done = True
                else:
                    done = False

                self.buffer.add(rollout, state_t.squeeze(0), logits.squeeze(0), value.squeeze(0), action.item(), reward, done)

                state_t = state_.detach().clone()
             
                 
    def save(self, save_path):
        self.model.save(save_path)
        self.imagination_module.save(save_path + "trained/")

    def load(self, save_path):
        self.model.load(save_path)
        self.imagination_module.load(save_path + "trained/")
    
  

    def _get_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t
  

    def _compute_loss(self, env_id):
        
        target_values_b = torch.FloatTensor(self.buffer.discounted_rewards[env_id]).to(self.model.device)


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

