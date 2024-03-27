import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
import yaml

from POMDP_model import get_random_dist, sample_from

def test_MC_1D():
    num_samples=1000

    size_of_space=10

    dist_true=get_random_dist(dim=size_of_space,dist_type='random')

    dist_empirical=get_random_dist(dim=size_of_space,dist_type='uniform')

    freq_table=np.zeros(size_of_space)

    with open('log.txt',mode='w') as log_file:
        loss=np.zeros(num_samples)
        for k in range(num_samples):
            sample=sample_from(dist=dist_true)
            freq_table[sample]+=1
            dist_empirical=freq_table/sum(freq_table)
            loss[k]=np.linalg.norm(dist_empirical-dist_true)
            if (k % 100 == 99):
                print(f"k={k} finish logging.")
        np.savetxt('log.txt',loss)
        log_file.close()

    with open('log.txt',mode='r') as log_file:
        loss_curve=np.loadtxt('log.txt')
        print(f"read in {len(loss_curve)} items from File:{'log.txt'}" )
        indices=np.arange(num_samples)
        plt.plot(indices,loss_curve)
        plt.title('2-norm loss of Monte-Carlo Simulation')
        plt.xlabel('iteration')
        plt.ylabel(r'$\| \widehat{p}-p \|_2$')
        plt.show()


from POMDP_model import initialize_model, initialize_policy, initialize_reward, sample_trajectory

def test_MC_high_dimensional():
    # load hyper parameters from a yaml file.
    with open("hyper_param.yaml", 'r') as file:
        hyper_param = yaml.safe_load(file)
    nA=hyper_param['sizes']['size_of_action_space']
    nS=hyper_param['sizes']['size_of_state_space']
    nO=hyper_param['sizes']['size_of_observation_space']
    H=hyper_param['sizes']['horizon_len']
    K=hyper_param['sizes']['num_episode']
    nF=pow((nO*nA),H) #size_of_history_space
    delta=hyper_param['sizes']['confidence_level']
    gamma=hyper_param['sizes']['discount_factor']
    iota =np.log(K*H*nS*nO*nA/delta)
    reward=initialize_reward(nS,nA,H,'random')

    # start the algorithm
    num_iter=K

    # obtain the true environment. invisible for the agent. Immutable. Only used during sampling.
    model_true=initialize_model(nS,nO,nA,H,init_type='random')
    mu,T,O=model_true

    # Initialize the empiricl kernels with uniform distributions.
    mu_hat, T_hat, O_hat=initialize_model(nS,nO,nA,H,init_type='uniform')
    
    policy=initialize_policy(nO,nA,H)

    # %%%%%% Parameter Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # (s,o,a) occurrence counters
    ## for initial state estimation
    Ns_init=torch.zeros([nS])      # frequency of s0
    ## for transition estimation
    Nssa=torch.zeros([H,nS,nS,nA]) # frequency of s' given (s,a)
    Nssa_ones=torch.ones([1,nS,nA])# matrix of 1 of size N(s,a) 
    Nsa=torch.ones([H,nS,nA])      # frequency of (s,a) :           \widehat{N}_{h}(s_h,a_h) \vee 1  h=1,2,3,...H
    ## for emission estimation
    Nos=torch.zeros([H+1,nO,nS])   # frequency of o  given s
    Nos_ones=torch.ones([1,nS])    # matrix of 1 of size N(s)
    Ns=torch.ones([H+1,nS])        # frequency of s:                \widehat{N}_{h}(s_{h}) \vee 1    h=1,2,3,...H+1
    
    # [Evaluation] Reset the errors before each iteration.
    mu_err=np.zeros([num_iter])
    T_err=np.zeros([num_iter])
    O_err=np.zeros([num_iter])

    # start to learn the dynamics.
    for k in range(num_iter):
        traj=sample_trajectory(H,policy,model=(mu,T,O))

        # update s0 count
        s0=traj[0][0]
        Ns_init[s0]+=1

        # update s,a ->s' pairs count.
        for h in range(H):
            s,a,ss=traj[0,h], traj[2,h], traj[0,h+1]
            Nssa[h][ss][s][a]+=1

        # update s->o pairs count.
        for h in range(H+1):
            s,o=traj[0,h], traj[1,h]
            Nos[h][o][s]+=1
        
        # update empirical initial distribution. \widehat{\mu}_1
        mu_hat=Ns_init/sum(Ns_init)

        # update empirical transition kernels.   \widehat{\mathbb{T}}^k_{h}: h=1,2,...H
        for h in range(H):
            #print(f"size: Nssa[h]={Nssa[h].shape}, ones={Nssa_ones.shape}, sum={torch.sum(Nssa[h],dim=0,keepdim=True).shape}")
            Nsa[h]=(torch.max(Nssa_ones, torch.sum(Nssa[h],dim=0,keepdim=True)))
            T_hat[h]=Nssa[h]/Nsa[h]

        # update empirical observation matrix.   \widehat{\mathbb{O}}^k_{h}: h=1,2,...H,H+1
        for h in range(H+1):
            #print(f"size: Nos[h]={Nos[h].shape}, ones={Nos_ones.shape}, sum={torch.sum(Nos[h],dim=0,keepdim=True).shape}")
            Ns[h]=(torch.max(Nos_ones, torch.sum(Nos[h],dim=0,keepdim=True)))
            O_hat[h]=Nos[h]/Ns[h]
        
        # [Evaluation] compute the average Frobenius error until this iter.
        mu_err[k]=torch.linalg.norm(mu-mu_hat)/mu.numel()
        T_err[k]=torch.linalg.norm(T-T_hat)/T.numel()
        O_err[k]=torch.linalg.norm(O-O_hat)/O.numel()
    # [Evaluation] output the loss curves of parameter learning.
    log_output(mu_err,T_err,O_err, H)
    
def log_output(mu_err,T_err,O_err, H:int)->None:
    '''
    write and read Monte-Carlo erros and plot three curves on a graph. 
    '''
    with open('log.txt',mode='w') as log_file:
        param_error=np.column_stack((mu_err,T_err,O_err))
        np.savetxt('log.txt',param_error)
        log_file.close()

    with open('log.txt',mode='r') as log_file:
        loss_curve=np.loadtxt('log.txt')
        print(f"read in {loss_curve.shape[0]} items from File:{'log.txt'}" )
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
        plt.savefig('plots/Monte-Carlo-Estimation-Error.jpg')
        plt.show()






# test_MC_high_dimensional()
