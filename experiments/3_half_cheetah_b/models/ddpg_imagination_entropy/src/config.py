import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.critic_learning_rate   = 0.0005
        self.actor_learning_rate    = 0.0001
        self.tau                    = 0.01

        self.batch_size          = 64
        self.update_frequency    = 1

        self.experience_replay_size = 100000
        self.exploration   = libs_common.decay.Linear(1000000, 0.5, 0.1, 0.1)
        
 
        self.imagination_rollouts       = 16
        
        self.entropy_beta               = 10.0
        self.curiosity_beta             = 0.0

        self.env_learning_rate          = 0.0002
