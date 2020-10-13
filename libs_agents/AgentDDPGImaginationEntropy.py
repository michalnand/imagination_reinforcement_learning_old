import numpy
import torch
from .ExperienceBufferContinuous import *


class AgentDDPGImaginationEntropy():
    def __init__(self, env, ModelCritic, ModelActor, ModelEnv, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                = config.tau

        self.exploration    = config.exploration
    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = 1
        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta
        self.env_learning_rate      = config.env_learning_rate

        self.experience_replay      = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        self.model_env              = ModelEnv.Model(self.state_shape, self.actions_count)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)
        self.optimizer_env      = torch.optim.Adam(self.model_env.parameters(), lr= config.env_learning_rate)
                
        self.state          = env.reset()

        self.iterations     = 0
        
        self.entropy_alpha  = 0.01
        self.entropy_mean   = 0.0

        self.enable_training()


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False


    
    
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()
       
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        action_t, action = self._sample_action(state_t, self.epsilon)

        action = action.squeeze()

        state_new, self.reward, done, self.info = self.env.step(action)

        state_predicted = self.model_env(state_t, action_t).squeeze().detach().numpy()
       
    
        if self.enabled_training:
            self.experience_replay.add(self.state, action, self.reward, done)

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
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        #environment model state prediction
        state_predicted_t   = self.model_env(state_t, action_t)

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
        curiosity = self.curiosity_beta*torch.tanh( ((state_next_t - state_predicted_t)**2).mean(dim = 1) ).detach()

        action_next_t       = self.model_actor_target.forward(state_next_t).detach()
        value_next_t        = self.model_critic_target.forward(state_next_t, action_next_t).detach()
 
        #target value, Q-learning
        value_target    = reward_t + entropy + curiosity + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

        #critic loss
        critic_loss     = ((value_target - value_predicted)**2)
        critic_loss     = critic_loss.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward() 
        self.optimizer_critic.step()

        #actor loss
        actor_loss      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        actor_loss      = actor_loss.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target networks 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)


    def save(self, save_path):
        self.model_critic.save(save_path)
        self.model_actor.save(save_path)
        self.model_env.save(save_path)

    def load(self, save_path):
        self.model_critic.load(save_path)
        self.model_actor.load(save_path)
        self.model_env.load(save_path)

    def _sample_action(self, state_t, epsilon):
        action_t    = self.model_actor(state_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np


    def _process_imagination(self, states_t, epsilon):
        batch_size  = states_t.shape[0]

        states_imagined_t      = torch.zeros((self.imagination_rollouts, batch_size, ) + self.state_shape ).to(self.model_env.device)

        for r in range(self.imagination_rollouts):
            states_imagined_t[r] = states_t.clone()


        for s in range(self.imagination_steps):
            #imagine rollouts
            for r in range(self.imagination_rollouts):
                action_t, _ = self._sample_action(states_imagined_t[r], epsilon)
                states_imagined_next_t  = self.model_env(states_imagined_t[r], action_t)
                states_imagined_t[r]    = states_imagined_next_t.clone()
        

        #swap axis, target ordering : batch rollout state
        states_imagined_t = states_imagined_t.transpose(1, 0)

        return states_imagined_t


    def _compute_entropy(self, states_t):
        batch_size  = states_t.shape[0]

        result      = torch.zeros(batch_size).to(self.model_env.device)

        for b in range(batch_size):
            v           = torch.var(states_t[b], dim = 0)
            result[b]   = v.mean()

        return result
    