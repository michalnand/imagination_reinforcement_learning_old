import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.learning_rate          = 0.002
        self.entropy_beta           = 0.01
        self.bellman_steps          = 4

        self.batch_size             = 256
        self.experience_replay_size = 8192
