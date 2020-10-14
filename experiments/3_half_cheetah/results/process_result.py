import sys
sys.path.insert(0, '../../..')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt


files = []
files.append("../models_0/ddpg/result/result.log")
files.append("../models_1/ddpg/result/result.log")
files.append("../models_2/ddpg/result/result.log")
files.append("../models_3/ddpg/result/result.log")
files.append("../models_4/ddpg/result/result.log")
rl_stats_compute_ddpg = RLStatsCompute(files, "ddpg_result_stats.log")


files = []
files.append("../models_0/ddpg_imagination_entropy/result/result.log")
files.append("../models_1/ddpg_imagination_entropy/result/result.log")
files.append("../models_2/ddpg_imagination_entropy/result/result.log")
files.append("../models_3/ddpg_imagination_entropy/result/result.log")
files.append("../models_4/ddpg_imagination_entropy/result/result.log")
rl_stats_compute_imagination = RLStatsCompute(files, "ddpg_imagination_entropy_result_stats.log") 


plt.cla()
plt.ylabel("score per episode")
plt.xlabel("episode")

plt.plot(rl_stats_compute_ddpg.games_mean, rl_stats_compute_ddpg.episode_mean, label="ddpg", color='blue')
plt.fill_between(rl_stats_compute_ddpg.games_mean, rl_stats_compute_ddpg.episode_lower, rl_stats_compute_ddpg.episode_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_imagination.games_mean, rl_stats_compute_imagination.episode_mean, label="ddpg imagination entropy", color='red')
plt.fill_between(rl_stats_compute_imagination.games_mean, rl_stats_compute_imagination.episode_lower, rl_stats_compute_imagination.episode_upper, color='red', alpha=0.2)


plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig("score_per_episode.png", dpi = 600)

 
plt.cla()
plt.ylabel("score per iteration")
plt.xlabel("iteration")

plt.plot(rl_stats_compute_ddpg.iterations, rl_stats_compute_ddpg.per_iteration_mean, label="ddpg", color='blue')
plt.fill_between(rl_stats_compute_ddpg.iterations, rl_stats_compute_ddpg.per_iteration_lower, rl_stats_compute_ddpg.per_iteration_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_imagination.iterations, rl_stats_compute_imagination.per_iteration_mean, label="ddpg imagination entropy", color='red')
plt.fill_between(rl_stats_compute_imagination.iterations, rl_stats_compute_imagination.per_iteration_lower, rl_stats_compute_imagination.per_iteration_upper, color='red', alpha=0.2)
 

plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig("score_per_iteration.png", dpi = 600)
