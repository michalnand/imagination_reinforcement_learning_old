import numpy
import torch


class ExperienceBufferPrioritized():

    def __init__(self, size, n_steps = 1):
        self.size   = size
       
        self.ptr    = 0 
        self.state_b  = []
        self.action_b = []
        self.reward_b = []
        self.done_b   = []
        self.priority = []

        self.n_steps = n_steps

        self.loss = numpy.ones(size)

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
            self.action_b.append(int(action))
            self.reward_b.append(reward)
            self.done_b.append(done_)
            self.priority.append(1.0)
        else:
            self.state_b[self.ptr]  = state.copy()
            self.action_b[self.ptr] = int(action)
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
        action_shape    = (batch_size, )
        reward_shape    = (self.n_steps, batch_size, )
        done_shape      = (self.n_steps, batch_size, )
      

        state_t         = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        action_t        = torch.zeros(action_shape,  dtype=int)
        reward_t        = torch.zeros(reward_shape,  dtype=torch.float32).to(device)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        done_t          = torch.zeros(done_shape,  dtype=torch.float32).to(device)

        #self.indices = self.find_indices_random(self.batch_size)
        self.indices = self.find_indices_priority(batch_size, self.loss)

        for i in range(0, batch_size):
            n  = self.indices[i]

            state_t[i]      = torch.from_numpy(self.state_b[n]).to(device)
            action_t[i]     = self.action_b[n]
            state_next_t[i] = torch.from_numpy(self.state_b[n + self.n_steps]).to(device)

            for j in range(self.n_steps):
                reward_t[j][i]     = torch.from_numpy(numpy.asarray(self.reward_b[n + j])).to(device)
                done_t[j][i]       = torch.from_numpy(numpy.asarray(self.done_b[n + j])).to(device)
        
        return state_t.detach(), action_t, reward_t.detach(), state_next_t.detach(), done_t.detach()

    def update_loss(self, loss_new):
        for i in range(len(self.indices)):
            n  = self.indices[i] 
            self.loss[n] = loss[i]

    def find_indices_random(self, count):
        indices = numpy.zeros(count, dtype=int)
        for i in range(count):
            indices[i]  = numpy.random.randint(self.length() - 1 - self.n_steps)
        
        return indices


    def find_indices_priority(self, count, loss):
        #probs sum have to be one
        #higher loss leads to higher probability

        loss_aligned = loss[0:loss.size - self.n_steps]
        probs = loss_aligned/numpy.sum(loss_aligned)

        indices = numpy.random.choice(len(probs), count, p=probs )

        return indices
        

if __name__ == "__main__":
    state_shape     = (3, 13, 17)
    action_shape    = (7,)

    buffer_size = 107

    n_steps = 3

    batch_size = 8


    replay_buffer = ExperienceBufferPrioritized(buffer_size, n_steps)

    for i in range(1000):
        state   = numpy.random.randn(state_shape[0], state_shape[1], state_shape[2])
        action  = numpy.random.randn(action_shape[0])[0]
        reward  = numpy.random.rand(1)
        done    = numpy.random.randint(2)

        replay_buffer.add(state, action, reward, done)

        if i > buffer_size:
            state_t, action_t, reward_t, state_next_t, done_t = replay_buffer.sample(batch_size, device="cpu")


    print(state_t.shape)
    print(action_t.shape)
    print(reward_t.shape)
    print(state_next_t.shape)
    print(done_t.shape)

