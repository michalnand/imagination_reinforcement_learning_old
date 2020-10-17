import libs_common.decay

class Config(): 

    def __init__(self):
        self.gamma              = 0.99
        self.update_frequency   = 4
        self.tau                = 0.0002

        self.batch_size     = 32 
        self.learning_rate  = 0.0001
        self.bellman_steps  = 4
         
        self.experience_replay_size = 32768
 

