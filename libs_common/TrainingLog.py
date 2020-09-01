import numpy
import time

class TrainingLog:

    def __init__(self, file_name, episode_skip_log = 10, iterations_skip_mode = False):

        self.iterations         = 0
        self.episodes           = 0
        self.episode_score_sum = 0.0
        self.episode_score_sum_raw = 0.0
        self.episode_iterations= 0.0
        self.episode_iterations_filtered= 0.0
      
        self.total_score        = 0.0


        self.episode_score_sum_filtered = 0.0
        self.episode_score_sum_raw_filtered = 0.0

        self.episode_score_best = -10**6

        self.episode_time_prev = time.time()
        self.episode_time_now  = time.time()
        self.episode_time_filtered = 0.0

        self.is_best = False

        self.episode_skip_log   = episode_skip_log
        self.iterations_skip_mode = iterations_skip_mode

        self.file_name = file_name

        if self.file_name != None:
            f = open(self.file_name,"w+")
            f.close()

    def add(self, reward, done, raw_reward = 0):

        self.total_score+= reward
        self.episode_score_sum+= reward
        self.episode_score_sum_raw+= raw_reward
        self.episode_iterations+= 1

        self.iterations+= 1

        self.is_best = False

        if done:
            self.episodes+= 1

            k = 0.02
            self.episode_iterations_filtered    = (1.0 - k)*self.episode_iterations_filtered + k*self.episode_iterations
            self.episode_score_sum_filtered     = (1.0 - k)*self.episode_score_sum_filtered + k*self.episode_score_sum
            
            self.episode_score_sum_raw_filtered = (1.0 - k)*self.episode_score_sum_raw_filtered + k*self.episode_score_sum_raw

            self.episode_time_prev = self.episode_time_now
            self.episode_time_now  = time.time()
            self.episode_time_filtered = (1.0 - k)*self.episode_time_filtered + k*(self.episode_time_now - self.episode_time_prev)

            if self.episodes > 20:
                if self.episode_score_sum_filtered > self.episode_score_best:
                    self.episode_score_best = self.episode_score_sum_filtered
                    self.is_best = True


            self.episode_score_sum = 0
            self.episode_score_sum_raw = 0
            self.episode_iterations = 0
        
        if self.iterations_skip_mode:
            tmp = self.iterations
        else:
            tmp = self.episodes

        if tmp%self.episode_skip_log == 0:
            dp = 3
            log_str = ""
            log_str+= str(self.iterations) + " "
            log_str+= str(self.episodes) + " "
            log_str+= str(round(self.episode_iterations_filtered, dp)) + " "
            log_str+= str(round(self.total_score, dp)) + " "
            log_str+= str(round(self.episode_score_sum_filtered, dp)) + " "
            log_str+= str(round(self.episode_time_filtered, 4)) + " "
            log_str+= str(round(self.episode_score_sum_raw_filtered, 4)) + " "

            print(log_str)

            f = open(self.file_name,"a+")
            f.write(log_str+"\n")
            f.close()




