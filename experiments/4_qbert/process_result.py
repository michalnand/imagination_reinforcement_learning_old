import sys
sys.path.insert(0, '../../..')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt


files = []
files.append("../models/dqn/result/result.log")
files.append("../models/dqn/result/result.log")
rl_stats_compute_dqn = RLStatsCompute(files, "dqn_result_stats.log")


files = []
files.append("../models/dqn_imagination_entropy/result/result.log")
files.append("../models/dqn_imagination_entropy/result/result.log")
rl_stats_compute_dqn_imagination = RLStatsCompute(files, "dqn_imagination_entropy.log")


plt.cla()
plt.ylabel("score per episode")
plt.xlabel("episode")

plt.plot(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_mean, label="dqn", color='blue')
plt.fill_between(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_lower, rl_stats_compute_dqn.episode_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_dqn_imagination.games_mean, rl_stats_compute_dqn_imagination.episode_mean, label="dqn imagination entropy", color='red')
plt.fill_between(rl_stats_compute_dqn_imagination.games_mean, rl_stats_compute_dqn_imagination.episode_lower, rl_stats_compute_dqn_imagination.episode_upper, color='red', alpha=0.2)


plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig("score_per_episode.png", dpi = 600)

 
plt.cla()
plt.ylabel("score per iteration")
plt.xlabel("iteration")

plt.plot(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.per_iteration_mean, label="dqn", color='blue')
plt.fill_between(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.per_iteration_lower, rl_stats_compute_dqn.per_iteration_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_dqn_imagination.iterations, rl_stats_compute_dqn_imagination.per_iteration_mean, label="dqn imagination entropy", color='red')
plt.fill_between(rl_stats_compute_dqn_imagination.iterations, rl_stats_compute_dqn_imagination.per_iteration_lower, rl_stats_compute_dqn_imagination.per_iteration_upper, color='red', alpha=0.2)
 

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig("score_per_iteration.png", dpi = 600)
