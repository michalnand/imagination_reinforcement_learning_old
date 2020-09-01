import torch
import numpy

class PolicyBufferContinuous:
    def __init__(self, envs_count, buffer_size, state_shape, actions_size, device):
        self.envs_count     = envs_count
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size  = actions_size
        self.device         = device

        self.clear()
  
    def clear(self):
        self.states_prev_b      = torch.zeros((self.envs_count, self.buffer_size, ) + self.state_shape).to(self.device)
        self.states_b           = torch.zeros((self.envs_count, self.buffer_size, ) + self.state_shape).to(self.device)

        self.mu_b               = torch.zeros((self.envs_count, self.buffer_size, self.actions_size)).to(self.device)
        self.var_b              = torch.zeros((self.envs_count, self.buffer_size, self.actions_size)).to(self.device)
        self.values_b           = torch.zeros((self.envs_count, self.buffer_size, 1)).to(self.device)
        
        self.actions_b          = torch.zeros((self.envs_count, self.buffer_size, self.actions_size))

        self.rewards_b          = numpy.zeros((self.envs_count, self.buffer_size)) 
        self.dones_b            = numpy.zeros((self.envs_count, self.buffer_size), dtype=bool)

        self.discounted_rewards_b = numpy.zeros((self.envs_count, self.buffer_size, 1))
        
        self.idxs = numpy.zeros(self.envs_count, dtype=int)

    def add(self, env_id, state, mu, var, value, action, reward, done):
        idx = int(self.idxs[env_id])

        self.states_b[env_id][idx]    = state

        self.states_prev_b[env_id][idx] = self.states_b[env_id][idx].clone()
        self.states_b[env_id][idx]      = state.clone()
        
        self.mu_b[env_id][idx]    = mu.clone()
        self.var_b[env_id][idx]   = var.clone()

        self.values_b[env_id][idx]    = value
        self.actions_b[env_id][idx]   = action
        self.rewards_b[env_id][idx]   = reward
        self.dones_b[env_id][idx]     = done

        self.idxs[env_id] = int(self.idxs[env_id] + 1)

    def size(self):
        return self.idxs[0]

    
    def calc_discounted_reward(self, gamma):

        self.discounted_rewards = numpy.zeros((self.envs_count, self.buffer_size, 1))

        for env_id in range(self.envs_count):
            q = 0.0
            for n in reversed(range(self.buffer_size)):
                if self.dones_b[env_id][n]:
                    gamma_ = 0.0
                else:
                    gamma_ = gamma

                q = self.rewards_b[env_id][n] + gamma_*q
                self.discounted_rewards[env_id][n][0] = q
        

            


    def sample(self, env_id, batch_size, device):
       
        
        states_b           = torch.zeros((batch_size, ) + self.state_shape).to(self.device)
        mu_b               = torch.zeros((batch_size, self.actions_size)).to(self.device)
        var_b              = torch.zeros((batch_size, self.actions_size)).to(self.device)
        values_b           = torch.zeros((batch_size, 1)).to(self.device)
        actions_b          = torch.zeros((batch_size), dtype= self.actions_b[0][0][0].dtype)

        self.actions_b          = torch.zeros((self.batch_size, self.actions_size))


        rewards_b          = numpy.zeros((batch_size)) 
        dones_b            = numpy.zeros((batch_size), dtype=bool)

        discounted_rewards_b = torch.zeros((batch_size, 1)).to(self.device)
        

        for i in range(batch_size):
            idx = numpy.random.randint(self.size())

            states_b[i] = self.states_b[env_id][idx].clone()
            mu_b[i]     = self.mu_b[env_id][idx].clone()
            var_b[i]    = self.var_b[env_id][idx].clone()
            values_b[i] = self.values_b[env_id][idx].clone()
            actions_b[i] = self.actions_b[env_id][idx].clone()
            rewards_b[i] = self.rewards_b[env_id][idx]
            dones_b[i] = self.dones_b[env_id][idx]

            discounted_rewards_b[i][0] = self.discounted_rewards_b[env_id][idx][0]



        return states_b.detach(), mu_b.detach(), var_b.detach(), values_b.detach(), actions_b, rewards_b, dones_b, discounted_rewards_b.detach()

    