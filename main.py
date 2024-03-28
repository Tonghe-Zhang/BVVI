
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

from func import test_normalization, test_output_log_file, current_time_str, Logger

from func import save_model_rewards, load_model_rewards, save_model_policy, load_model_policy

# load hyper parameters
nS,nO,nA,H,K,nF,delta,gamma,iota = load_hyper_param("hyper_param.yaml")

# obtain the true environment. invisible for the agent. Immutable. Only used during sampling.
real_env_kernels=initialize_model(nS,nO,nA,H,init_type='random')

# initiliaze the reward
reward_fix=initialize_reward(nS,nA,H,'random')

# record the generated kernels and rewards.
save_model_rewards(real_env_kernels, reward_fix, 'real_env')

# Training 
prt_progress=True
prt_policy_normalization=True

# [Evaluation] Reset the parameter errors and accumulated returns tested in the true envirnoemt of each iteration.
mu_err=np.zeros([K])
T_err=np.zeros([K])
O_err=np.zeros([K])
tested_returns=np.zeros([K])
evaluation_metrics=(mu_err, T_err, O_err, tested_returns)


def beta_vector_value_iteration(model_true, reward, evaluation_metrics,log_episode_file, model_load=None, policy_load=None)->tuple:
    '''
    inputs: as the name suggests.
    output: policy, model_learnt, evaluation results
    '''
    # used only during sampling.
    mu,T,O=model_true

    # used only during evaluation
    mu_err, T_err, O_err, tested_returns=evaluation_metrics

    # %%%%%%% Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if model_load==None:
        # Initialize the empiricl kernels with uniform distributions.
        mu_hat, T_hat, O_hat=initialize_model(nS,nO,nA,H,init_type='uniform')
    else:
        mu_hat, T_hat, O_hat=model_load
    if policy_load==None:
        # initialize the policy
        policy_load=initialize_policy(nO,nA,H)
    else:
        policy_load=policy_load

    if prt_policy_normalization:
        print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_normalization(policy_test=policy_load,size_act=nA,size_obs=nO)}")

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
    
    # %%%%%% Start of training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            policy_shape=policy_load[h].shape
            policy_load[h]=torch.zeros(policy_shape).scatter(dim=-1,index=max_indices,src=torch.ones(policy_shape))
            if prt_policy_normalization:
                print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_normalization(policy_test=policy_load,size_act=nA,size_obs=nO)}")
            '''      
            '''

            # action_greedy is \widehat{\pi}_h^k(f_h)
            action_greedy=torch.argmax(policy_load[h][hist]).item()
            
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

        # %%%%%% Parameter Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if prt_progress:
            print(f"\t\tEnter parameter learning")
        # line 29-30 in the original paper. Interact with the environment and sample a trajectory.
        if prt_policy_normalization:
            print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_normalization(policy_test=policy_load,size_act=nA,size_obs=nO)}")
        traj=sample_trajectory(H,policy_load,model=(mu,T,O),reward=reward,output_reward=False)

        # line 34 in the orignal paper.
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
        ## update empirical observation matrix.   \widehat{\mathbb{O}}^k_{h}: h=1,2,...H,H+1
        for h in range(H+1):
            Ns[h]=(torch.max(Nos_ones, torch.sum(Nos[h],dim=0,keepdim=True)))
            O_hat[h]=Nos[h]/Ns[h]

        # %%%%%% Performance evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # [Evaluation] test policy learnt in this episode against the true environment, collect the average accumulated rewards of 10 i.i.d. tests.
        reward_tests=[np.sum(sample_trajectory(H,policy_load,model_true,reward,output_reward=True)) for _ in range(10) ]
        tested_returns[k]=np.mean(reward_tests)
        # [Evaluation] compute the average Frobenius error between the true and learnt parameters until this iter.
        mu_err[k]=torch.linalg.norm(mu-mu_hat)/mu.numel()
        T_err[k]=torch.linalg.norm(T-T_hat)/T.numel()
        O_err[k]=torch.linalg.norm(O-O_hat)/O.numel()
        # [Logging]logging into log_episode_file after each episode.
        if prt_progress:
            print(f"\tEnd of episode {k}. policy's tested_returns[{k}]={tested_returns[k]}, mu_err[{k}]={mu_err[k]}, T_err[{k}]={T_err[k]}, O_err[{k}]={O_err[k]}")
        write_str=str(tested_returns[k])+'\t'+str(mu_err[k])+'\t'+str(T_err[k])+'\t'+str(O_err[k])+'\t'
        log_episode_file.write(write_str+ "\n")
        # [Save weights] record the latest learnt parameters and policy:
        save_model_policy((mu_hat, T_hat, O_hat), policy_load, 'learnt')
        if prt_progress:
            print(f"\tSuccessfuly saved newest kernels and policies to folder: {'./learnt'}")
    # %%%%%% End of training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if prt_progress:
        print(f"End of training. number of iters K={K}")
    model_load=(mu_hat, T_hat, O_hat)
    evaluation_results=(mu_err,T_err,O_err,tested_returns)
    return (policy_load, model_load, evaluation_results)

def visualize_performance(evaluation_results):
    # unpack
    mu_err,T_err,O_err, tested_returns=evaluation_results

    # plot planning result.
    log_output_tested_rewards(tested_returns,H)

    # plot parameter learning results
    log_output_param_error(mu_err,T_err,O_err, H)

def main(output_to_log_file=False, train_from_scratch=True, model_true_load=None, reward_true_load=None, model_load=None, policy_load=None):
    if output_to_log_file:
        print(f"Will output log information to both the file:{'console_output.log'} and the console.")
        old_stdout = sys.stdout
        log_file = open("console_output.log","w")
        sys.stdout = Logger() #sys.stdout = log_file
        print(f"Start BVVI test. Current time={current_time_str()}")
        time.sleep(3)

    print('%'*100)
    print('test Beta Vector Value Iteration.')
    print('%'*100)
    print('hyper parameters:{}')
    with open('hyper_param.yaml') as hyp_file:
        content=hyp_file.read()
    print(content)
    print('%'*100)
    print('Call function \'  beta_vector_value_iteration...\' ')

    with open('log_episode.txt',mode='r+') as log_episode_file:
        log_episode_file.write(f"\n\nTest BVVI. Current time={current_time_str()}")
        if train_from_scratch:
            (policy, model_learnt, evaluation_results)=beta_vector_value_iteration(\
                model_true=real_env_kernels,\
                    reward=reward_fix,\
                        evaluation_metrics=evaluation_metrics,\
                            log_episode_file=log_episode_file)
        else:
            (policy, model_learnt, evaluation_results)=beta_vector_value_iteration(\
                model_true=model_true_load,\
                    reward=reward_true_load,\
                        model_load=model_load,\
                            policy_load=policy_load,
                                evaluation_metrics=evaluation_metrics,\
                                    log_episode_file=log_episode_file)
        log_episode_file.write(f"\n\nEnd Testing BVVI. Current time={current_time_str()}")
        log_episode_file.close()
    episode_data=np.loadtxt('log_episode.txt', dtype=np.float64)
    print('\'  beta_vector_value_iteration...\' returned.')
    print(f"End BVVI test. Current time={current_time_str()}")
    print('%'*100)
    print('Call function \'  visualize_performance...\' ')

    visualize_performance(evaluation_results)

    print('\'  visualize_performance...\' returned.')
    print('%'*100)
    print('Beta Vector Value Iteration test complete.')
    print('%'*100)
    
    if output_to_log_file is True:
        sys.stdout = old_stdout
        log_file.close()



'''
1. To load previously saved models from file, run these commands:

model_true_load, reward_true_load=load_model_rewards('real_env')

model_load, policy_load=load_model_policy('learnt')

.....and then run this command:

main(output_to_log_file=True, train_from_scratch=False, model_true_load=model_true_load, reward_true_load=reward_true_load, model_load=model_load, policy_load=policy_load)


2. To do the job without logging too much, run:

with open('log_episode_2.txt',mode='r+') as log_episode_file:
    (policy, model_learnt, evaluation_results)=beta_vector_value_iteration(\
                model_true=model_true_load,\
                    reward=reward_true_load,\
                        model_load=model_load,\
                            policy_load=policy_load,
                                evaluation_metrics=evaluation_metrics,\
                                    log_episode_file=log_episode_file)


3. To print not to the logging file but only to the console, run:

output_to_log_file=False, then print logging info to console

'''
        
main(output_to_log_file=True, train_from_scratch=True)



'''''
# check recent update in console log file:

while True:
    with open("console_output.log",mode='r') as log_console_file:
        print(log_console_file.read())
    time.sleep(10)
'''



''''
# Test the file

test_output_log_file()

'''