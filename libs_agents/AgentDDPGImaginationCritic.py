import numpy
import torch
from .ExperienceBufferContinuous import *

from .ImaginationModule import *


class AgentDDPGImaginationCritic():
    def __init__(self, env, ModelCritic, ModelActor, ModelImagination, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.gamma          = config.gamma
        self.update_frequency = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration

       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size)


        self.model_actor    = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target    = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic   = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target   = ModelCritic.Model(self.state_shape, self.actions_count)

        
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)

        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = config.imagination_steps
        self.imagination_module     = ImaginationModule(ModelImagination, self.state_shape, self.actions_count, config.imagination_learning_rate, self.experience_replay, True)

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

        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        _, actions_b, values_b  = self._process_imagination(state_t)
        action                  = self._sample_imagination(actions_b, values_b)
        
        state_new, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            self.experience_replay.add(self.state, action, self.reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size) and self.iterations%self.update_frequency == 0:
            self.train_model()
            self.imagination_module.train()
                   
        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new

        self.iterations+= 1

        return self.reward, done

    def _process_imagination(self, state_initial):
        states_b    = torch.zeros((self.imagination_steps, self.imagination_rollouts, ) + self.state_shape).to(self.model_actor.device)
        actions_b   = torch.zeros((self.imagination_steps, self.imagination_rollouts, self.actions_count)).to(self.model_actor.device)
        values_b    = torch.zeros((self.imagination_steps, self.imagination_rollouts, 1)).to(self.model_actor.device)

        states_t    = state_initial.repeat(self.imagination_rollouts, 1).clone().detach()

        for n in range(self.imagination_steps):                       
            actions_t = self._sample_action(states_t)

            states_next_t, _  = self.imagination_module.eval(states_t, actions_t)
            value_t           = self.model_critic(states_t, actions_t)
           
            states_t = states_next_t.clone()

            states_b[n]     = states_t.clone()
            actions_b[n]    = actions_t.clone()
            values_b[n]     = value_t.clone()

        
        values_b = values_b.squeeze(2)

        return states_b, actions_b, values_b

    def _sample_imagination(self, actions_b, values_b):
        values_mean  = torch.mean(values_b, dim = 1)
        best_idx     = torch.argmax(values_mean)

        return actions_b[0][best_idx].detach().to("cpu").numpy()
           
    def _sample_action(self, state_t):
        
        if self.enabled_training:
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
       
        action_t    = self.model_actor(state_t)

        noise       = torch.randn(state_t.shape[0], self.actions_count).to(self.model_actor.device)
        action_t    = action_t + epsilon*noise
        action_t    = torch.clamp(action_t, -1.0, 1.0)
    

        return action_t

    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        action_next_t   = self.model_actor_target.forward(state_next_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, action_next_t).detach()

        #critic loss
        value_target    = reward_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

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
        self.imagination_module.save(save_path) 

    def load(self, save_path):
        self.model_critic.load(save_path)
        self.model_actor.load(save_path)
        self.imagination_module.load(save_path)     


   