import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.critic_learning_rate   = 0.0002
        self.actor_learning_rate    = 0.0001
        self.tau                    = 0.001

        self.batch_size          = 64
        self.update_frequency    = 4

        self.exploration   = libs_common.decay.Linear(2000000, 1.0, 0.3, 0.3)

        self.experience_replay_size = 16384

        self.imagination_beta             = 0.01
        self.entropy_beta                 = 0.1
        self.imagination_rollouts         = 4
        self.imagination_steps            = 4
        self.imagination_learning_rate    = 0.0002