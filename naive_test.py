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

T_true=torch.stack([torch.tensor([[0,0,1],[1,0,0],[0,1,0]]).unsqueeze(-1).repeat(1,1,2) for _ in range(3)])
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

def test_dp():
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


# test Monte Carlo
    




from func import test_normalization_T,test_normalization_O
K=5000
mu_err=np.zeros([K])
T_err=np.zeros([K])
O_err=np.zeros([K])
mu_hat, T_hat, O_hat=initialize_model(nS,nO,nA,H,init_type='uniform')
Ns_init, Nssa, Nssa_ones, Nsa, Nos, Nos_ones, Ns =init_occurrence_counters(H,nS,nO,nA)
def test_monte_carlo(K,H,policy_load,mu,T,O,reward_true):
    for k in range(K):
        print(f"k={k}/{K}, {k/K*100:2f}\%")
        traj=sample_trajectory(H,policy_load,model=(mu,T,O),reward=reward_true,output_reward=False)
        ## update s0 count
        s0=traj[0][0]
        Ns_init[s0]+=1
        ## update s,a ->s' pairs count.
        for h in range(H):
            s,a,ss=traj[0,h], traj[2,h], traj[0,h+1]
            Nssa[h][ss][s][a]+=1
        ## update s->o pairs count.
        for h in range(H+1):
            s,o=traj[0,h], traj[1,h]
            Nos[h][o][s]+=1

        # line 35 in the orignal paper.
        ## update empirical initial distribution. \widehat{\mu}_1
        mu_hat=Ns_init/sum(Ns_init)
        ## update empirical transition kernels.   \widehat{\mathbb{T}}^k_{h}: h=1,2,...H
        for h in range(H):
            Nsa[h]=(torch.max(Nssa_ones, torch.sum(Nssa[h],dim=0,keepdim=True)))
            T_hat[h]=Nssa[h]/Nsa[h]
            # force normalization in case some (s,a) is not visited i.e. if some Nssa[h][s,a]==0
            for s in range(nS):
                for a in range(nA):
                    #print(f"Before: T_hat[{h}][:,{s},{a}]= { T_hat[h][:,s,a]}")
                    #print(f"sum(T_hat[h][:,s,a])==0{sum(T_hat[h][:,s,a])==0}")
                    normalize_sum=sum(T_hat[h][:,s,a])
                    if normalize_sum==0:
                        T_hat[h][:,s,a]=torch.ones_like(T_hat[h][:,s,a])/nS 
                    else:
                        T_hat[h][:,s,a]=T_hat[h][:,s,a]/normalize_sum
                    #print(f"After: T_hat[{h}][:,{s},{a}]= { T_hat[h][:,s,a]}")
        test_normalization_T(T_hat,nS,nA,H)
        # print(f"Check normalization for T: {test_normalization_T(T_hat,nS,nA,H)}")
        ## update empirical observation matrix.   \widehat{\mathbb{O}}^k_{h}: h=1,2,...H,H+1
        for h in range(H+1):
            Ns[h]=(torch.max(Nos_ones, torch.sum(Nos[h],dim=0,keepdim=True)))
            O_hat[h]=Nos[h]/Ns[h]
            # force normalization in case some (s) is not yet visited. i.e. some Nos[h][s]==0
            for s in range(nS):
                #print(f"O_hat[{h}][:,{s}]={O_hat[h][:,s]}, sum=={sum(O_hat[h][:,s])}")
                normalize_sum=sum(O_hat[h][:,s])
                if normalize_sum==0:
                    O_hat[h][:,s]=torch.ones_like(O_hat[h][:,s])/nO
                    #print(f"is zero,  change to {O_hat[h][:,s]}")
                else:
                    O_hat[h][:,s]=O_hat[h][:,s]/normalize_sum
                    #print(f"not zero,  change to {O_hat[h][:,s]}")
        test_normalization_O(O_hat,nS,H)
        # print(f"Check normalization for O: {test_normalization_O(O_hat,nS,H)}")
        mu_err[k]=torch.linalg.norm(mu-mu_hat)/mu.numel()
        T_err[k]=torch.linalg.norm(T-T_hat)/T.numel()
        O_err[k]=torch.linalg.norm(O-O_hat)/O.numel()
    log_output_param_error(mu_err,T_err,O_err,H)

test_monte_carlo(K,H=H,
                 policy_load=policy_star,
                 mu=mu_true,
                 T=T_true,
                 O=O_true,
                 reward_true=R_true)
