import libs_common.decay

class Config(): 

    def __init__(self):
        self.batch_size             = 32 
        
        self.gamma                  = 0.99
        self.learning_rate          = 0.0001
        self.tau                    = 0.0002
        self.update_frequency       = 4
        self.prioritized_buffer     = True
        self.bellman_steps          = 4
                
        self.experience_replay_size = 32768
        self.exploration            = libs_common.decay.Linear(1000000, 1.0, 0.05, 0.05)
      