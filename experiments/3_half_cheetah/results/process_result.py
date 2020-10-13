import sys
sys.path.insert(0, '../../..')

from libs_common.RLStatsCompute import *

files = []
files.append("../models_0/ddpg/result/result.log")
files.append("../models_1/ddpg/result/result.log")
files.append("../models_2/ddpg/result/result.log")
rl_stats_comput = RLStatsCompute(files, "ddpg_result_stats.log")


files = []
files.append("../models_0/ddpg_imagination_entropy/result/result.log")
files.append("../models_1/ddpg_imagination_entropy/result/result.log")
files.append("../models_2/ddpg_imagination_entropy/result/result.log")
rl_stats_comput = RLStatsCompute(files, "ddpg_imagination_entropy_result_stats.log") 