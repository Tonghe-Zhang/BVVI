import torch
import numpy as np   
import yaml 
import itertools
import matplotlib.pyplot as plt
import sys
import time
from tqdm import tqdm

from POMDP_model import initialize_model_reward, initialize_model, initialize_policy, sample_trajectory, get_random_dist, sample_from

from utils import negative_func, positive_func, log_output_param_error, log_output_tested_rewards, log_output_test_reward_pretty, load_hyper_param, init_value_representation, init_history_space, init_occurrence_counters

from utils import test_policy_normalized, test_output_log_file, current_time_str, Logger, short_test

from utils import save_model_rewards, load_model_rewards, save_model_policy, load_model_policy, test_normalization_O,test_normalization_T

def BVVI(hyper_param:tuple,
         num_episodes:int,
         model_true:tuple,
         reward_true:torch.Tensor,
         model_load:tuple,
         policy_load,
         evaluation_metrics,
         log_episode_file,
         true_weight_output_parent_directory='real_env',
         weight_output_parent_directory='learnt',
         prt_progress=True,
         prt_policy_normalization=True
         )->tuple:
    '''
    inputs: 
        hyper_param: tuple. (nS,nO,nA,H,K,nF,delta,gamma,iota). 
        model_true:  (mu, T, O) ternary tensor tuple.   the kernels of the true environment.
        reward_true: 3D tensor. the reward matrices of the true environmet. 
        log_episode_file:string, the filename to which logging information of each episode is output. 
                        the format of log_episode_file:      [tested risk-measure of policy k    mu_err[k]   T_err[k]   O_err[k] ]
                                                                                        ...         ...         ...         ... 
    output: ternary tensor tuple 
        (policy_learnt, model_learnt, evaluation_results)
    '''
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if model_true==None:
        raise(ValueError)
    # move true model to gpu.
    model_true=tuple(tensor.to(device) for tensor in model_true)
    
    if reward_true==None:
        raise(ValueError)
    reward_true.to(device)
    
    # unpack hyper parameters
    nS,nO,nA,H,K,nF,delta,gamma,iota =hyper_param
    # we can change number of episodes K from the console.
    K=num_episodes
    
    # used only during sampling.
    mu_true,T_true,O_true=model_true
    # save the true models and rewards used in this experiment.
    save_model_rewards(kernels=model_true, reward_table=reward_true, parent_directory=true_weight_output_parent_directory)
    # used only during evaluation
    mu_err, T_err, O_err, tested_risk_measure=evaluation_metrics

    # %%%%%%% Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    Check whether we will train from existing parameters (.pt, .pth) or train from scratch. 
    if mode_load is None, then we will train the model from scratch. We will initialize it.
    the same for the policy.
    '''
    if model_load==None:
        # train kernels from scratch. Initialize the empirical kernels with uniform distributions.
        model_load=initialize_model(nS,nO,nA,H,init_type='uniform')
    # move model_load to gpus.
    model_load=tuple(tensor.to(device) for tensor in model_load)
    mu_hat, T_hat, O_hat=model_load
    
    if policy_load==None:
        # train polcy from scratch. Initialize the policy
        policy_learnt=initialize_policy(nO,nA,H)
    else:
        policy_learnt=policy_load
    
    # move to gpus.
    policy_learnt=[item.to(device) for item in policy_learnt]

    if prt_policy_normalization:
        print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_policy_normalized(policy_test=policy_learnt,size_act=nA,size_obs=nO)}")

    # Bonus residues, correspond to \mathsf{t}_h^k(\cdot,\cdot)  and  \mathsf{o}_{h+1}^k(s_{h+1})
    bonus_res_t=torch.ones([H,nS,nA]).to(torch.float64).to(device)
    bonus_res_o=torch.ones([H+1,nS]).to(torch.float64).to(device)   # in fact there is no h=0 for residue o. we shift everything right.

    # Bonus
    ''''we should be aware that in the implementation, s_{H+1} is set to be absorbed to 0 so T(s_{H+1}|s_H) = \delta(S_{H+1}=0) and o_{H+1}^k(s_{H+1})= 1 if s_{H+1} \ne 0   else 3\sqrt{\frac{OH \iota}{k}} 
    so bonus_{H} is needs special care. '''
    bonus=torch.ones([H,nS,nA]).to(torch.float64).to(device)

    # create the history spaces \{\mathcal{F}_h\}_{h=1}^{H}
    history_space=init_history_space(H,nO,nA)

    # initialize empirical risk beliefs, beta vectors, Q-functions and value functions
    sigma_hat, beta_hat, Q_function, value_function=init_value_representation(H,nS, nO, nA)
    # move to gpus
    sigma_hat=[item.to(device) for item in sigma_hat]
    beta_hat=[item.to(device) for item in beta_hat]
    Q_function=[item.to(device) for item in Q_function]
    value_function=[item.to(device) for item in value_function]

    # initialize the (s,o,a) occurrence counters
    counters=init_occurrence_counters(H,nS,nO,nA)
    counters=tuple(cnt.to(device) for cnt in counters)        
    Ns_init, Nssa, Nssa_ones, Nsa, Nos, Nos_ones, Ns=counters
    
    # %%%%%% Start of training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k in tqdm(range(K)):
        # print(f"\n\n\tInto episode {k}/{K}={(k/K*100):.2f}%")
        if prt_progress:
            print(f"\n\n\tInto episode {k}/{K}={(k/K*100):.2f}%")
        # %%%%%%% Belief propagation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                '''
                print(f"O_hat[h][obs,:].shape={O_hat[h][obs,:].shape}")
                print(f"T_hat[h-1,:,:,act].shape={T_hat[h-1,:,:,act].shape}")
                print(f"reward_true[h-1,:,act].shape={reward_true[h-1,:,act].shape}")
                '''
                sigma_hat[h][hist]=np.float64(nO)*\
                    torch.diag(O_hat[h][obs,:]).to(dtype=torch.float64)\
                        @ T_hat[h-1,:,:,act].to(dtype=torch.float64)  \
                            @  torch.diag(torch.exp(gamma* reward_true[h-1,:,act])).to(dtype=torch.float64).to(device) \
                                @ sigma_hat[h-1][prev_hist].to(dtype=torch.float64)
        # line 11 of the original paper
        bonus_res_t=torch.min(torch.ones([H,nS,nA]).to(device), 3*torch.sqrt(nS*H*iota / Nsa)).to(device)
        bonus_res_o=torch.min(torch.ones([H+1,nS]).to(device), 3*torch.sqrt(nO*H*iota/Ns)).to(device)
        
        # line 12 of the original paper. Notice that h starts from 0 in pytorch it's different from the original paper.
        for h in range(H):
            bonus[h]=np.fabs(np.exp(gamma*(H-h))-1)*\
                torch.min(torch.ones([nS,nA]).to(device), \
                    bonus_res_t[h]+torch.tensordot(bonus_res_o[h+1].to(torch.float64), T_hat[h].to(torch.float64), dims=1))
        if prt_progress:
            print(f"\t\t belief propagation ends...") 

        # %%%%%% Dynamic programming %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if prt_progress:
            print(f"\t\t dynamic programming starts...")
        # re-initialize
        beta_hat=[torch.ones_like(sigma_hat[h]).to(device) for h in range(H+1)] 
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
            if prt_progress:
                print(f"\t\t\t update Q function...")
            history_coordinates=list(itertools.product(*history_space[h]))
            for hist in history_coordinates:     # here hist represents f_h
                for act in range(nA):         # here action represents a_h, here obs is for o_{h+1}
                    # line 19 in the original paper.
                    Q_function[h][hist][act]=\
                        gamma* torch.log(1e-7+1/nO * \
                                    sum([torch.inner(sigma_hat[h+1][(hist)+(act,obs)] , beta_hat[h+1][(hist)+(act,obs)]) for obs in range(nO)] ))
            
            # line 22 in the original paper.
            if prt_progress:
                print(f"\t\t\t update value function...")
            value_function[h]=torch.max(Q_function[h],dim=-1,keepdim=False).values
            # line 23 in the original paper.

            # select greedy action for the policy. The policy is one-hot in the last dimension.
            if prt_progress:
                print(f"\t\t\t update greedy policy...")
            max_indices=torch.argmax(Q_function[h],dim=-1,keepdim=True).to(device)   # good thing about argmax: only return 1 value when there are multiple maxes. 
            policy_shape=policy_learnt[h].shape
            policy_learnt[h]=torch.zeros(policy_shape).to(device).scatter(dim=-1,index=max_indices,src=torch.ones(policy_shape).to(device))
            if prt_policy_normalization:
                print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_policy_normalized(policy_test=policy_learnt,size_act=nA,size_obs=nO)}")

            # action_greedy is \widehat{\pi}_h^k(f_h)
            action_greedy=torch.argmax(policy_learnt[h][hist]).item()
            
            # line 23 in the original paper.
            if prt_progress:
                print(f"\t\t\t update beta vector...")
            for state in range(nS):
                beta_hat[h][hist][state]=np.exp(gamma*reward_true[h][state][action_greedy])*\
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
                beta_hat[h][hist][state]=torch.clamp(beta_hat[h][hist][state], \
                                                torch.exp(torch.tensor(gamma_minus*(H-h))), \
                                                    torch.exp(torch.tensor(gamma_plus*(H-h))))
            if prt_progress:
                print(f"\t\t\t Horizon remains: {h}/{H}")

        # %%%%%% Parameter Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if prt_progress:
            print(f"\t\tEnter parameter learning")
        # line 29-30 in the original paper. Interact with the environment and sample a trajectory.
        if prt_policy_normalization:
            print(f"\t\t\t\tPOLICY NORMALIZATION TEST:{test_policy_normalized(policy_test=policy_learnt,size_act=nA,size_obs=nO)}")
        traj=sample_trajectory(H,policy_learnt,model=model_true,reward=reward_true,output_reward=False)

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
            # force normalization in case some (s,a) is not visited i.e. if some Nssa[h][s,a]==0. It accelerated convergence of parameters.
            for s in range(nS):
                for a in range(nA):
                    #print(f"Before: T_hat[{h}][:,{s},{a}]= { T_hat[h][:,s,a]}")
                    #print(f"sum(T_hat[h][:,s,a])==0{sum(T_hat[h][:,s,a])==0}")
                    normalize_sum=sum(T_hat[h][:,s,a])
                    if normalize_sum==0:
                        T_hat[h][:,s,a]=torch.ones_like(T_hat[h][:,s,a]).to(device)/nS 
                    else:
                        T_hat[h][:,s,a]=T_hat[h][:,s,a]/normalize_sum
                    #print(f"After: T_hat[{h}][:,{s},{a}]= { T_hat[h][:,s,a]}")
        # test_normalization_T(T_hat,nS,nA,H)
        # print(f"Check normalization for T: {test_normalization_T(T_hat,nS,nA,H)}")
        
        ## update empirical observation matrix.   \widehat{\mathbb{O}}^k_{h}: h=1,2,...H,H+1
        for h in range(H+1):
            Ns[h]=(torch.max(Nos_ones, torch.sum(Nos[h],dim=0,keepdim=True)))
            O_hat[h]=Nos[h]/Ns[h]
            # force normalization in case some (s) is not yet visited. i.e. some Nos[h][s]==0 It accelerated convergence of parameters.
            for s in range(nS):
                #print(f"O_hat[{h}][:,{s}]={O_hat[h][:,s]}, sum=={sum(O_hat[h][:,s])}")
                normalize_sum=sum(O_hat[h][:,s])
                if normalize_sum==0:
                    O_hat[h][:,s]=torch.ones_like(O_hat[h][:,s]).to(device)/nO
                    #print(f"is zero,  change to {O_hat[h][:,s]}")
                else:
                    O_hat[h][:,s]=O_hat[h][:,s]/normalize_sum
                    #print(f"not zero,  change to {O_hat[h][:,s]}")
        # test_normalization_O(O_hat,nS,H)
        # print(f"Check normalization for O: {test_normalization_O(O_hat,nS,H)}")
        
        # %%%%%% Performance evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # [Evaluation] test policy learnt in this episode against the true environment, collect the average accumulated rewards of 10 i.i.d. tests.
        ''''
        tested_return = \frac{1}{\gamm} \mathbb{E}^{\widehat{\pi}^k} e^{\gamma \sum_{h=1}^H r_h(s_h,a_h)} 
        '''
        num_samples=20
        # Fixed a small mistake: forgot to use np.log()
        tested_risk_measure[k]=(1/gamma)*torch.log(1e-7+torch.tensor([torch.exp(gamma*sum(sample_trajectory(H,policy_learnt,model_true,reward_true,output_reward=True))) for _ in range(num_samples)]).mean())
        # [Evaluation] compute the average Frobenius error between the true and learnt parameters until this iter.
        mu_err[k]=torch.linalg.norm(mu_true-mu_hat)/mu_true.numel()
        T_err[k]=torch.linalg.norm(T_true-T_hat)/T_true.numel()
        O_err[k]=torch.linalg.norm(O_true-O_hat)/O_true.numel()
        # [Logging]logging into log_episode_file after each episode.
        if prt_progress:
            print(f"\tEnd of episode {k}. policy's tested_returns[{k}]={tested_risk_measure[k]}, mu_err[{k}]={mu_err[k]}, T_err[{k}]={T_err[k]}, O_err[{k}]={O_err[k]}")
        write_str=str(tested_risk_measure[k])+'\t'+str(mu_err[k])+'\t'+str(T_err[k])+'\t'+str(O_err[k])+'\t'
        log_episode_file.write(write_str+ "\n")
        # [Save weights] record the latest learnt parameters and policy every 200 episodes
        if (k % 200==0):
            save_model_policy((mu_hat, T_hat, O_hat), policy_learnt, weight_output_parent_directory)
        if prt_progress:
            print(f"\tSuccessfuly saved the newest kernels and policies to folder: {'./learnt'}")
    # %%%%%% End of training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if prt_progress:
        print(f"End of training. Number of iters K={K}")
    model_learnt=(mu_hat, T_hat, O_hat)
    evaluation_results=(mu_err,T_err,O_err,tested_risk_measure)
    return (policy_learnt, model_learnt, evaluation_results)
