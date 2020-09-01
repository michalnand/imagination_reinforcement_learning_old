import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPO():
    def __init__(self, envs, Model, Config):
        self.envs = envs

        config = Config.Config()

        self.envs_count = len(self.envs)
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.eps_clip = config.eps_clip
        self.batch_size      = config.batch_size
        self.buffer_size     = config.buffer_size
        self.training_epochs    = config.training_epochs
       
        self.state_shape = self.envs[0].observation_space.shape
        self.actions_count     = self.envs[0].action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_old      = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)


        self.buffer = PolicyBuffer(self.envs_count, self.buffer_size, self.state_shape, self.actions_count, self.model.device)

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
        
        logits, value   = self.model_old.forward(state_t)

        action_t = self._get_action(logits)
            
        self.states[env_id], reward, done, _ = self.envs[env_id].step(action_t.item())
        
        if self.enabled_training:
            self.buffer.add(env_id, state_t.squeeze(0), logits.squeeze(0), value.squeeze(0), action_t.item(), reward, done)
           
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
        

        if self.buffer.size() > self.buffer_size-1:  

            self.buffer.calc_discounted_reward(self.gamma)

            batch_count = 1 + self.buffer_size//self.batch_size

            for epoch in range(self.training_epochs):
                for i in range(batch_count):

                    env_id = numpy.random.randint(self.envs_count)
                
                    loss= self._compute_loss(env_id)

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

   

    def _get_action(self, logits, dim = 0):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t
  

    def _compute_loss(self, env_id):
        
        state_t, logits_b, values_b, actions_b, rewards_b, dones_b, target_values_b = self.buffer.sample(env_id, self.batch_size, self.model.device)

        probs_old     = torch.nn.functional.softmax(logits_b, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(logits_b, dim = 1).detach()


        logits, values   = self.model.forward(state_t)

        probs     = torch.nn.functional.softmax(logits, dim = 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim = 1)

        actions_b = self._get_action(logits, 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - values)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss 
        '''
        log_probs_      = log_probs[range(len(log_probs)),  actions_b[env_id]]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions_b[env_id]]
        

                
        # Finding Surrogate Loss:
        advantage   = (target_values_b - values).detach()

        
        ratios = torch.exp(log_probs_ - log_probs_old_)
        surr1 = ratios*advantage
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantage
        
        loss_policy = -torch.min(surr1, surr2) 
        loss_policy = loss_policy.mean()
    

        #loss_policy = -log_probs[range(len(log_probs)), self.buffer.actions_b[env_id]]*advantage
        #loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        #print(loss_value, loss_policy, loss_entropy)

        #train network, with gradient cliping
        loss = loss_value + loss_policy + loss_entropy

        return loss

