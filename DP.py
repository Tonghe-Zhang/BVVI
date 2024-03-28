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


def dynamic_programing(model_true, reward, model_load, policy_load,config_filename, prt_progress, prt_policy_normalization, evaluation_metrics)->list:
    
    nS,nO,nA,H,K,nF,delta,gamma,iota = load_hyper_param('config\\'+config_filename+'.yaml')
    
    
    policy_at_every_episode=[None for _ in range(K)]
    
    '''
    inputs: as the name suggests.
    output: policy, model_learnt, evaluation results
    '''
    # used only during sampling.
    mu,T,O=model_true

    # used only during evaluation
    mu_err, T_err, O_err, tested_risk_measure=evaluation_metrics

    # %%%%%%% Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if model_load==None:
        # Initialize the empiricl kernels with uniform distributions.
        mu_hat, T_hat, O_hat=initialize_model(nS,nO,nA,H,init_type='uniform')
    else:
        mu_hat, T_hat, O_hat=model_load
    if policy_load==None:
        # initialize the policy
        policy_learnt=initialize_policy(nO,nA,H)
    else:
        policy_learnt=policy_load

    if prt_policy_normalization:
        print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_policy_normalized(policy_test=policy_learnt,size_act=nA,size_obs=nO)}")

    # Bonus residues, correspond to \mathsf{t}_h^k(\cdot,\cdot)  and  \mathsf{o}_{h+1}^k(s_{h+1})
    bonus_res_t=torch.ones([H,nS,nA]).to(torch.float64)
    bonus_res_o=torch.ones([H+1,nS]).to(torch.float64)   # in fact there is no h=0 for residue o. we shift everything right.

    # Bonus
    ''''we should be aware that in the implementation, s_{H+1} is set to be absorbed to 0 so T(s_{H+1}|s_H) = \delta(S_{H+1}=0) and o_{H+1}^k(s_{H+1})= 1 if s_{H+1} \ne 0   else 3\sqrt{\frac{OH \iota}{k}} 
    so bonus_{H} is needs special care. '''
    bonus=torch.ones([H,nS,nA]).to(torch.float64)

    # create the history spaces \{\mathcal{F}_h\}_{h=1}^{H}
    history_space=init_history_space(H,nO,nA)

    # initialize empirical risk beliefs, beta vectors, Q-functions and value functions
    sigma_hat, beta_hat, Q_function, value_function=init_value_representation(H,nS, nO, nA)

    # initialize the (s,o,a) occurrence counters
    Ns_init, Nssa, Nssa_ones, Nsa, Nos, Nos_ones, Ns =init_occurrence_counters(H,nS,nO,nA)
    
    for k in range(K):
        if prt_progress:
            print(f"\n\n\tInto episode {k}/{K}={(k/K*100):.2f}%")
        # %%%%%%% Belief propagation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if prt_progress:
            print(f"\t\t belief propagation starts...")
        # line 6 in the original paper.
        sigma_hat[0]=mu_hat.unsqueeze(0).repeat(nO,1)
        # line 7 to 9 in the original paper.
        for h in range(1,H+1,1):
            #print(f"progress:{[torch.min(sigma_hat[t]).item() for t in range(H)]}")   # view how each layer evolves.
            history_coordinates=list(itertools.product(*history_space[h]))
            for hist in history_coordinates:
                #previous history f_{h-1}, act: a_{h-1}, obs: o_h
                prev_hist, act, obs=hist[0:-2], hist[-2], hist[-1]   
                # use Eqs.~\eqref{40} in the original paper to simplify the update rule.
                # be aware that we should use @ but not * !!!   * is Hadamard product while @ is matrix/vector product.
                sigma_hat[h][hist]=np.float64(nO)*torch.diag(O_hat[h][obs,:]).to(dtype=torch.float64) @ T_hat[h-1,:,:,act].to(dtype=torch.float64)  @  torch.diag(torch.exp(gamma* reward[h-1,:,act])).to(dtype=torch.float64) @ sigma_hat[h-1][prev_hist].to(dtype=torch.float64)
        # line 11 of the original paper
        bonus_res_t=torch.min(torch.ones([H,nS,nA]), 3*torch.sqrt(nS*H*iota / Nsa))
        bonus_res_o=torch.min(torch.ones([H+1,nS]), 3*torch.sqrt(nO*H*iota/Ns))
        
        # line 12 of the original paper. Notice that h starts from 0 in pytorch it's different from the original paper.
        for h in range(H):
            bonus[h]=np.fabs(np.exp(gamma*(H-h))-1)*\
                torch.min(torch.ones([nS,nA]), \
                    bonus_res_t[h]+torch.tensordot(bonus_res_o[h+1].to(torch.float64), T_hat[h].to(torch.float64), dims=1))
        if prt_progress:
            print(f"\t\t belief propagation ends...") 

        # %%%%%% Dynamic programming %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if prt_progress:
            print(f"\t\t dynamic programming starts...")
        # re-initialize
        beta_hat=[torch.ones_like(sigma_hat[h]) for h in range(H+1)] 
        for q_func in Q_function:
            q_func.zero_()
        for v_func in value_function:
            v_func.zero_()
        # line 16 in the original paper.
        for h in range (H-1,-1,-1):
            '''
            iter h from H-1 to 0
            policy is defined as stochastic, so its last dimension is nA and it shares the same size as Q-functions.
            only through dynamic programming our policy is deterministic (one-hot)
            the value function is 1 dimension lower than q function.
            '''
            # Invoke Bellman equation (49) under beta vector representation
            print(f"\t\t\t update Q function...")
            history_coordinates=list(itertools.product(*history_space[h]))
            for hist in history_coordinates:     # here hist represents f_h
                for act in range(nA):         # here action represents a_h, here obs is for o_{h+1}
                    # line 19 in the original paper.
                    Q_function[h][hist][act]=\
                        gamma* np.log(1/nO * \
                                    sum([torch.inner(sigma_hat[h+1][(hist)+(act,obs)] , beta_hat[h+1][(hist)+(act,obs)]) for obs in range(nO)] ))
            
            # line 22 in the original paper.
            print(f"\t\t\t update value function...")
            value_function[h]=torch.max(Q_function[h],dim=-1,keepdim=False).values
            # line 23 in the original paper.

            # select greedy action for the policy. The policy is one-hot in the last dimension.
            print(f"\t\t\t update greedy policy...")
            max_indices=torch.argmax(Q_function[h],dim=-1,keepdim=True)   # good thing about argmax: only return 1 value when there are multiple maxes. 
            policy_shape=policy_learnt[h].shape
            policy_learnt[h]=torch.zeros(policy_shape).scatter(dim=-1,index=max_indices,src=torch.ones(policy_shape))
            if prt_policy_normalization:
                print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_policy_normalized(policy_test=policy_learnt,size_act=nA,size_obs=nO)}")

            # action_greedy is \widehat{\pi}_h^k(f_h)
            action_greedy=torch.argmax(policy_learnt[h][hist]).item()
            
            # line 23 in the original paper.
            if prt_progress:
                print(f"\t\t\t update beta vector...")
            for state in range(nS):
                beta_hat[h][hist][state]=np.exp(gamma*reward[h][state][action_greedy])*\
                    sum([ T_hat[h][next_state][state][action_greedy]*
                        (
                            sum([ O_hat[h+1][next_obs][next_state]* beta_hat[h+1][hist+(action_greedy,next_obs,)][next_state] for next_obs in range(nO)])
                        )
                    for next_state in range(nS)
                    ])\
                    + np.sign(gamma)*bonus[h][state][action_greedy]
                
                # line 24: Control the range of beta vector
                gamma_plus=positive_func(gamma)
                gamma_minus=negative_func(gamma)
                beta_hat[h][hist][state]=np.clip(beta_hat[h][hist][state], \
                                                np.exp(gamma_minus*(H-h)), \
                                                    np.exp(gamma_plus*(H-h)))
            if prt_progress:
                print(f"\t\t\t Horizon remains: {h}/{H}")

        policy_at_every_episode[k]=policy_learnt
    return policy_at_every_episode