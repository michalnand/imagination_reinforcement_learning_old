class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.learning_rate          = 0.0002
        self.entropy_beta           = 0.01
        self.eps_clip               = 0.2

        self.batch_size             = 512
        self.episodes_to_train      = 8
        self.training_epochs        = 8
