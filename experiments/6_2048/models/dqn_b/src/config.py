import common.decay

class Config(): 

    def __init__(self):
        self.gamma = 0.99
        self.tau   = 0.001
        
        self.update_frequency = 8

        self.batch_size     = 64 
        self.learning_rate  = 0.0001
        self.bellman_steps  = 4

        self.exploration = common.decay.Exponential(0.99999988, 1.0, 0.01, 0.01)
        
        self.experience_replay_size = 16384 
 

