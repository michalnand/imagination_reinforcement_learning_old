import numpy
import torch
from .ExperienceBufferContinuous import *

from .ImaginationModule import *


class AgentDDPGImaginationMetaActor():
    def __init__(self, env, ModelCritic, ModelActor, ModelImagination, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration

        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        
        

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)

        self.entropy_beta           = config.entropy_beta
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

        _, actions_b, rewards_b = self._process_imagination(state_t)
        action                  = self._sample_imagination(actions_b, rewards_b)

        state_new, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            self.experience_replay.add(self.state, action, self.reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size) and self.iterations%self.update_frequency == 0:
            self.imagination_module.train()
            self.train_model()

        self.state = state_new
            
        if done:
            self.env.reset()

        self.iterations+= 1

        return self.reward, done

   
    def _process_imagination(self, state_initial):
        states_b    = torch.zeros((self.imagination_steps, self.imagination_rollouts, ) + self.state_shape).to(self.model_actor.device)
        actions_b   = torch.zeros((self.imagination_steps, self.imagination_rollouts, self.actions_count)).to(self.model_actor.device)
        values_b    = torch.zeros((self.imagination_steps, self.imagination_rollouts, 1)).to(self.model_actor.device)

        states_t    = state_initial.repeat(self.imagination_rollouts, 1).clone().detach()

        for n in range(self.imagination_steps):                       
            actions_t = self._sample_action(states_t.detach())

            states_next_t, _  = self.imagination_module.eval(states_t.detach(), actions_t)
            value_t           = self.model_critic(states_t.detach(), actions_t)
           
            states_t = states_next_t.clone()

            states_b[n]     = states_next_t.clone()
            actions_b[n]    = actions_t.clone()
            values_b[n]     = value_t.clone()

        
        values_b = values_b.squeeze(2)

        return states_b, actions_b, values_b

    def _sample_imagination(self, actions_b, values_b):

        values_sum   = torch.sum(values_b, dim = 0).detach().to("cpu").numpy()

        probs        = numpy.exp(values_sum - numpy.max(values_sum))
        probs        = probs/numpy.sum(probs)
        selected     = numpy.random.choice(range(len(probs)), p = probs)

        return actions_b[0][selected].detach().to("cpu").numpy()
    
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

    def _imagination_exploration_entropy_loss(self, initial_states_b):
        batch_size = initial_states_b.shape[0]

        entropy_t = torch.zeros((batch_size, 1))

        for n in range(batch_size):
            states_b, _, _  = self._process_imagination(initial_states_b[n])

            #take ending states in each rollout
            ending_states_b = states_b[:][self.imagination_steps - 1]

            #compute variance
            variance        = torch.var(ending_states_b)
            
            #compute entropy, considering states distribution is gaussian
            entropy_t[n]    = 0.5*torch.log(2.0*numpy.pi*numpy.e*variance)
        
        exploration_entropy_loss = entropy_t.mean()        
        return exploration_entropy_loss
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        action_next_t      = self.model_actor_target.forward(state_next_t).detach()
        value_next_t       = self.model_critic_target.forward(state_next_t, action_next_t).detach()

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

        #maximize exploration entropy loss
        entropy_loss = -self.entropy_beta*self._imagination_exploration_entropy_loss(state_t)
        actor_loss+=  entropy_loss
        
        #print("entropy_loss = ", entropy_loss)


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
