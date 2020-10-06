import numpy
import torch
from .ExperienceBufferContinuous import *


class AgentDDPGEntropy():
    def __init__(self, env, ModelCritic, ModelActor, ModelEntropy, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration
    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.entropy_samples        = config.entropy_samples
        self.entropy_beta           = config.entropy_beta
        self.entropy_learning_rate  = config.entropy_learning_rate

        self.experience_replay      = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        self.model_entropy          = ModelEntropy.Model(self.state_shape, self.actions_count)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)
        self.optimizer_entropy  = torch.optim.Adam(self.model_entropy.parameters(), lr= config.entropy_learning_rate)

        self.state          = env.reset()

        self.iterations     = 0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
       
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        action_t    = self.model_actor(state_t)
        action      = action_t.squeeze(0).detach().to("cpu").numpy()

        noise  = numpy.random.normal(loc = 0.0, scale = epsilon, size = self.actions_count)
        action = action + epsilon*noise

        action = numpy.clip(action, -1.0, 1.0)

        state_new, self.reward, done, self.info = self.env.step(action)

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
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device, self.entropy_samples)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        #entropy bonus reward
        entropy_predicted_t    = self.model_entropy(state_t, state_next_t, action_t)
        entropy_target_t       = self._compute_entropy(self.experience_replay.sample_next_states(self.entropy_samples, self.model_entropy.device))

        action_next_t       = self.model_actor_target.forward(state_next_t).detach()
        value_next_t        = self.model_critic_target.forward(state_next_t, action_next_t).detach()

        #entropy loss
        entropy_loss = (entropy_target_t - entropy_predicted_t)**2
        entropy_loss = entropy_loss.mean()
        
        #update entropy
        self.optimizer_entropy.zero_grad()
        entropy_loss.backward() 
        self.optimizer_entropy.step()

        #critic loss
        entropy_t       = self.entropy_beta*torch.tanh(entropy_predicted_t).detach()
        value_target    = reward_t + entropy_t + self.gamma*done_t*value_next_t
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
        self.model_entropy.save(save_path)

    def load(self, save_path):
        self.model_critic.load(save_path)
        self.model_actor.load(save_path)
        self.model_entropy.load(save_path)

    def _compute_entropy(self, states):
        batch_size  = states.shape[0]

        result      = torch.zeros(batch_size).to(self.model_entropy.device)

        for b in range(batch_size):
            v           = torch.var(states[b], dim = 0)
            result[b]   = v.mean()

        return result
    