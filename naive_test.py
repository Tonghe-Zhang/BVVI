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

# T_true=torch.tensor([[0,0,1],[1,0,0],[0,1,0]]).unsqueeze(0).unsqueeze(-1).repeat(4,1,1,2).to(torch.float64)
# O_true=torch.tensor([[0.4,0.2,0.4],[0.3,0.5,0.2],[0.2,0.7,0.1]]).unsqueeze(0).transpose(0,1).repeat(4,1,1).to(torch.float64)
# mu_true=torch.tensor([1,0,0]).to(torch.float32)
# R_true=torch.tensor([[1,0],[0,1],[1,0]]).unsqueeze(0).repeat(3,1,1).to(torch.float64)


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



def short_test(policy=policy_star,only_reward=False):
    from POMDP_model import sample_from, action_from_policy

    horizon=3
    model=(mu_true,T_true,O_true)
    reward=R_true
    output_reward=True

    print("\n")
    init_dist, trans_kernel, emit_kernel =model 

    full_traj=np.ones((3,horizon+1), dtype=int)*(-1)   
    if output_reward:
        sampled_reward=np.ones(horizon, dtype=np.float64)*(-1)

    # S_0
    full_traj[0][0]=sample_from(init_dist)
    # A single step of interactions
    for h in range(horizon+1):
        # S_h
        state=full_traj[0][h]
        # O_h \sim \mathbb{O}_h(\cdot|s_h)
        observation=sample_from(emit_kernel[h][:,state])
        full_traj[1][h]=observation
        # A_h \sim \pi_h(\cdot |f_h). We do not collect A_{H+1}, which is a[H]. We set a[H] as 0
        if h<horizon:
            action=action_from_policy(full_traj[1:3,:],h,policy)
            full_traj[2][h]=action
            # R_h = r_h(s_h,a_h)
            if output_reward:
                sampled_reward[h]=reward[h][state][action]
                print(f"(h,s,a,r)={h,state,action,sampled_reward[h]}")
        # S_h+1 \sim \mathbb{T}_{h}(\cdot|s_{h},a_{h})
        if h<horizon:  #do not record s_{H+1}
            new_state=sample_from(trans_kernel[h][:,state,action])
            full_traj[0][h+1]=new_state

    print(f"sampled_reward={sampled_reward}")
    print(f"full_traj=\n{full_traj}")
    if only_reward==True:
        return sampled_reward

nS,nO,nA,H,K,nF,delta,gamma,iota = load_hyper_param("config\hyper_param_naive.yaml")


short_test(policy_star,False)
sampled_reward=short_test(policy_star,True)

num_samples=10
tested_risk_measure=(1/gamma)*np.array([np.exp(gamma*sum(short_test(policy_star,True))) for _ in range(num_samples)]).mean()


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