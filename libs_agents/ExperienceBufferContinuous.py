import numpy
import torch


class ExperienceBufferContinuous():

    def __init__(self, size):
        self.size   = size
       
        self.ptr    = 0 
        self.state_b  = []
        self.action_b = []
        self.reward_b = []
        self.done_b   = []

    def length(self):
        return len(self.state_b)

    def is_full(self):
        if self.length() == self.size:
            return True
            
        return False

    def add(self, state, action, reward, done):

        if done != 0:
            done_ = 1.0
        else:
            done_ = 0.0

        if self.length() < self.size:
            self.state_b.append(state.copy())
            self.action_b.append(action)
            self.reward_b.append(reward)
            self.done_b.append(done_)
        else:
            self.state_b[self.ptr]  = state.copy()
            self.action_b[self.ptr] = action.copy()
            self.reward_b[self.ptr] = reward
            self.done_b[self.ptr]   = done_

            self.ptr = (self.ptr + 1)%self.length()

    def _print(self):
        for i in range(self.length()):
            #print(self.state_b[i], end = " ")
            print(self.action_b[i], end = " ")
            print(self.reward_b[i], end = " ")
            print(self.done_b[i], end = " ")
            print("\n")

   
    def sample(self, batch_size, device):
        
        state_shape     = (batch_size, ) + self.state_b[0].shape[0:]
        action_shape    = (batch_size, ) + self.action_b[0].shape[0:]
        reward_shape    = (batch_size, )
        done_shape      = (batch_size, )


        state_t         = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        action_t        = torch.zeros(action_shape,  dtype=torch.float32).to(device)
        reward_t        = torch.zeros(reward_shape,  dtype=torch.float32).to(device)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        done_t          = torch.zeros(done_shape,  dtype=torch.float32).to(device)

        
        for i in range(0, batch_size):
            n  = numpy.random.randint(self.length() - 1)
            state_t[i]      = torch.from_numpy(self.state_b[n]).to(device)
            action_t[i]     = torch.from_numpy(self.action_b[n]).to(device).to(device)
            reward_t[i]     = torch.from_numpy(numpy.asarray(self.reward_b[n])).to(device)
            state_next_t[i] = torch.from_numpy(self.state_b[n+1]).to(device)
            done_t[i]       = torch.from_numpy(numpy.asarray(self.done_b[n])).to(device)

        return state_t.detach(), action_t.detach(), reward_t.detach(), state_next_t.detach(), done_t.detach()


if __name__ == "__main__":
    state_shape     = (3, 13, 17)
    action_shape    = (7,)


    replay_buffer = ExperienceBuffer(107)

    for i in range(1000):
        state   = numpy.random.randn(state_shape[0], state_shape[1], state_shape[2])
        action  = numpy.random.randn(action_shape[0])[0]
        reward  = numpy.random.rand(1)
        done    = numpy.random.randint(2)

        replay_buffer.add(state, action, reward, done)

        if i > 1:
            state_t, action_t, reward_t, state_next_t, done_t = replay_buffer.sample(5, device="cpu")


    print(state_t.shape)
    print(action_t.shape)
    print(reward_t.shape)
    print(state_next_t.shape)
    print(done_t.shape)

