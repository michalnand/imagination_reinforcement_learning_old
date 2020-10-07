import numpy
from scipy import stats

class RLStatsCompute:
    def __init__(self, files_list, destination_file = "result_stats.log"):
        iterations, games, total_score, episode_score = self.load_files(files_list)
        self.process_stats(iterations, games, total_score, episode_score, destination_file)

    def load_file(self, file_name):
        data                = numpy.loadtxt(file_name, unpack = True)

        iteration           = data[0]
        games               = data[1]
        total_score         = data[3]
        episode_score       = data[4]

        return iteration, games, total_score, episode_score

    def load_files(self, files_list):
        iterations      = []
        games           = []
        total_score     = []
        episode_score   = []

        for f in files_list:
            iterations_, games_, total_score_, episode_score_ = self.load_file(f)

            iterations.append(iterations_)
            games.append(games_)
            total_score.append(total_score_)
            episode_score.append(episode_score_)

        iterations      = numpy.array(iterations)
        games           = numpy.array(games)
        total_score     = numpy.array(total_score)
        episode_score   = numpy.array(episode_score)

        return iterations, games, total_score, episode_score
        

    def compute_stats(self, data, interval = 0.95):
        count   = data.shape[0]
        alpha   = 1.0 - interval
        df      = count - 1
        t       = stats.t.ppf(1.0 - alpha/2.0, df)  
    
        mean = numpy.mean(data, axis = 0)
        std  = numpy.std(data, ddof=1, axis = 0)

        lower = mean - (t * std/ numpy.sqrt(count))
        upper = mean + (t * std/ numpy.sqrt(count))

        return mean, std, lower, upper

    def process_stats(self, iterations, games, total_score, episode_score, file_name):
        per_iteration_score = total_score/iterations[0]

        games_mean, games_std, games_lower, games_upper         = self.compute_stats(games)

        total_mean, total_std, total_lower, total_upper         = self.compute_stats(total_score)
        per_iteration_mean, per_iteration_std, per_iteration_lower, per_iteration_upper         = self.compute_stats(per_iteration_score)
        episode_mean, episode_std, episode_lower, episode_upper = self.compute_stats(episode_score)

        decimal_places = 4
        f = open(file_name, "w")
        for i in range(len(iterations[0])):
            result_str = ""
            result_str+= str(iterations[0][i]) + " "
            result_str+= str(games_mean[i])      + " "
            
            result_str+= str(round(total_mean[i], decimal_places)) + " "
            result_str+= str(round(total_lower[i], decimal_places)) + " "
            result_str+= str(round(total_upper[i], decimal_places)) + " "

            result_str+= str(round(per_iteration_mean[i], decimal_places)) + " "
            result_str+= str(round(per_iteration_lower[i], decimal_places)) + " "
            result_str+= str(round(per_iteration_upper[i], decimal_places)) + " "
    
            result_str+= str(round(episode_mean[i], decimal_places)) + " "
            result_str+= str(round(episode_lower[i], decimal_places)) + " "
            result_str+= str(round(episode_upper[i], decimal_places)) + " "
            result_str+= "\n" 

            f.write(result_str)

        f.close()
