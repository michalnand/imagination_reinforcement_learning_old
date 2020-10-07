import sys
sys.path.insert(0, '../../..')

from libs_common.RLStatsCompute import *

files = []
'''
files.append("result_0.log")
files.append("result_1.log")
files.append("result_2.log")
files.append("result_3.log")
files.append("result_4.log")
files.append("result_5.log")
files.append("result_6.log")
files.append("result_7.log")
'''

files.append("result.log")
files.append("result.log")
files.append("result.log")
files.append("result.log")
files.append("result.log")
files.append("result.log")
files.append("result.log")
files.append("result.log")


rl_stats_comput = RLStatsCompute(files, "ddpg_result_stats.log")