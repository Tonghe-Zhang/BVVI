import torch
import numpy as np   
import yaml 
import itertools
import matplotlib.pyplot as plt
import sys
import time

# load the RL platform
from POMDP_model import initialize_model, initialize_policy, initialize_reward, sample_trajectory, get_random_dist, sample_from

from func import load_hyper_param, current_time_str, log_output_tested_rewards

with open('log\log_episode.txt',mode='r+') as log_episode_file:
    data=np.loadtxt('log\log_episode.txt')
    print(data.shape)
    tested_returns=data[:,0]
    loss_curve=data[:,1:4]
    
    nS,nO,nA,H,K,nF,delta,gamma,iota = load_hyper_param("config\hyper_param.yaml")
    # plot planning result.
    log_output_tested_rewards(tested_returns,H)

    # plot parameter learning results
    indices=np.arange(loss_curve.shape[0])*H
    labels_plt=['Initial distribution $\mu(\cdot)$',\
                'Transition matrices $\{\mathbb{T}_h(\cdot|s,a)\}_{h=1}^{H+1}$',\
                    'Emission matrices $\{\mathbb{O}_h(\cdot|s)\}_{h=1}^{H+1}$']
    for id in range(3):
        plt.plot((indices),loss_curve[:,id],label=labels_plt[id])
    plt.title(f'Average 2-norm Error of Monte-Carlo Simulation. Horizon H={H}')
    plt.xlabel(f'Samples N (=iteration $k$ * {H})')    # H transitions per iteration.
    plt.ylabel(r'$\frac{1}{d} \| \widehat{p}^k(\cdot)-p(\cdot) \|_2$')
    plt.legend(loc='upper right', labels=labels_plt)
    plt.savefig('plots/MCErr'+current_time_str()+'.jpg')
    plt.show()
    

