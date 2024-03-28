import torch
import numpy as np   
import yaml 
import itertools
import matplotlib.pyplot as plt
import sys
import time

# load the RL platform
from POMDP_model import initialize_model, initialize_policy, initialize_reward, sample_trajectory, get_random_dist, sample_from

from func import negative_func, positive_func, log_output_param_error, log_output_tested_rewards, load_hyper_param, init_value_representation, init_history_space, init_occurrence_counters

from func import test_policy_normalized, test_output_log_file, current_time_str, Logger

from func import save_model_rewards, load_model_rewards, save_model_policy, load_model_policy

from func import load_hyper_param


prt_progress=True
prt_policy_normalization=True

T_true=torch.stack([torch.tensor([[0,0,1],[1,0,0],[0,1,0]]).unsqueeze(-1).repeat(1,1,2) for _ in range(4)])
O_true=torch.stack([torch.tensor([[0.4,0.2,0.4],[0.3,0.5,0.2],[0.2,0.7,0.1]]).transpose(0,1).repeat(1,1) for _ in range(4)])
mu_true=torch.tensor([1,0,0])
R_true=torch.tensor([[1,0],[0,1],[1,0]]).unsqueeze(0).repeat(3,1,1)

model_true_load=(mu_true, T_true, O_true)
reward_true_load=R_true






policy_star=[None for _ in range(3)]
policy_star[0]=torch.tensor([1,0]).unsqueeze(0).repeat(3,1)
policy_star[1]=torch.tensor([0,1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3,2,3,1)
policy_star[2]=torch.tensor([1,0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3,2,3,2,3,1)
policy_test=policy_star




from func import short_test


nS,nO,nA,H,K,nF,delta,gamma,iota = load_hyper_param("config\hyper_param_naive.yaml")


short_test(policy_star,mu_true,T_true,O_true,R_true,only_reward=False)
sampled_reward=short_test(policy_star,mu_true,T_true,O_true,R_true,only_reward=True)

num_samples=10
tested_risk_measure=(1/gamma)*np.array([np.exp(gamma*sum(short_test(policy_star,mu_true,T_true,O_true,R_true,only_reward=True))) for _ in range(num_samples)]).mean()


mu_err=np.zeros([K])
T_err=np.zeros([K])
O_err=np.zeros([K])
tested_returns=np.zeros([K])
evaluation_metrics=(mu_err, T_err, O_err, tested_returns)

from DP import dynamic_programing as dp
with open('log\log_episode_naive.txt',mode='r+') as log_episode_file:
    policy_at_each_episode=dp(\
                model_true=model_true_load,\
                    reward=reward_true_load,\
                        model_load=model_true_load,\
                            prt_progress=True,\
                                prt_policy_normalization=True,\
                                    evaluation_metrics=evaluation_metrics,\
                                            config_filename='hyper_param_naive',\
                                                policy_load=None)

# unpack
num_samples=10
tested_risk_measure=np.zeros([K])
for k in range(K):
    tested_risk_measure[k]=(1/gamma)*np.array([np.exp(gamma*sum(short_test(policy_at_each_episode[k], True))) for _ in range(num_samples)]).mean()

# plot planning result.
log_output_tested_rewards(tested_risk_measure,H)