
# mischellaneous functions
import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch

# (x)^+ and (x)^- functions.
def negative_func(x:np.double)->np.double:
    return np.min(x,0)
def positive_func(x:np.double)->np.double:
    return np.max(x,0)

# load hyper parameters from a yaml file.
def load_param(hyper_param_file_name:str)->tuple:
    ''''
    given hyper parameter file name, return the params.   
    
    '''
    with open(hyper_param_file_name, 'r') as file:
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
    return (nS,nO,nA,H,K,nF,delta,gamma,iota)

# output loss curve.
def log_output_param_error(mu_err,T_err,O_err, H:int)->None:
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
        plt.savefig('plots/MCErr'+str(datetime.datetime.now())[:-7]+'.jpg')
        plt.show()

def log_output_tested_rewards(accumulated_rewards_of_each_episode,H:int)->None:
    loss_curve=accumulated_rewards_of_each_episode
    indices=np.arange(loss_curve.shape[0])*H
    labels_plt=['Average Accumulated Rewards']
    for id in range(3):
        plt.plot((indices),loss_curve[:,id],label=labels_plt[id])
    plt.title(f'Average Accumulated Rewards of Output Policies. Horizon H={H}')
    plt.xlabel(f'Samples N (=iteration $k$ * {H})')    # H transitions per iteration.
    plt.ylabel(r'$sum_{h=1}^{H}r_h(\boldsymbol{S}_h,\boldsymbol{A}_h)$')
    plt.legend(loc='upper right', labels=labels_plt)
    plt.savefig('plots/Reward'+str(datetime.datetime.now())[:-7]+'.jpg')
    plt.show()





def init_history_space(H:int, nO:int, nA:int)->list:
    '''
    inputs: horizon length H, sizes of observation space nO and action space nA.
    outputs: list of tensors. 
    '''
    observation_space=tuple(list(np.arange(nO)))
    action_space=tuple(list(np.arange(nA)))
    history_space=[None for _ in range(H+1)]
    for h in range(H+1):
        # Create the space of \mathcal{F}_h = (\mathcal{O}\times \mathcal{A})^{h-1}\times \mathcal{O}
        history_space[h]=[observation_space if i%2==0 else action_space for i in range(2*(h))]+[observation_space]
    return history_space

def init_value_representation(horizon_length:int,size_of_state_space:int, size_of_observation_space:int, size_of_action_space:int)->tuple:
    '''
    inputs: as the name suggests.
    output: a list of tensors, the series of (empirical) risk-sensitive beliefs   sigma :  \vec{\sigma}_{h,f_h} \in \R^{S}
    In the following loop, 
        "hist_coords" is the coordinates of all the possible histories in the history space of order h.
        This iterator traverses the history space of \mathcal{F}_h, which recordes all possible 
        observable histories up to step h. The order of traversal is identical to a binary search tree.
        Each element in the coordinate list "hist" is an OAOA...O tuple of size 2h-1.
        "hist" can be viewed as an encoding of f_h
        
        sigma[h][hist] is of shape torch.Size([nS])  is the vector \vec{sigma}_{h,f_h} \in \R^S        
        
        sigma[h-1][hist[0:-2]] is still of shape torch.Size([nS]) is the belief of previous history: \vec{sigma}_{h-1,f_{h-1}} \in \R^S   

        run these lines to check the shapes of the tensors:
            print(f"h={h}":)
            print(f"\thistory{hist} correspond to belief {sigma[h][hist].shape}")
            if h >=1:
                print(f"\t\twhose preivous history is {hist[0:-2]}, with previous belief {sigma[h-1][hist[0:-2]].shape}")
    '''
    risk_belief=[None for _ in range(horizon_length+1)]
    for h in range(horizon_length+1):
        risk_belief[h]=torch.zeros([size_of_observation_space if i%2==0 else size_of_action_space for i in range(2*(h))]+[size_of_observation_space] +[size_of_state_space], dtype=torch.float64)
    
    '''
    Create beta vectors, Q-values and value functions
    beta_vector:       tensor list of length H+1
        beta_vector[h][hist][s] is \widehat{\beta}_{h, f_h}^k(s_h)
    Q_function:     tensor list of length H
        each element Q_function[h].shape    torch.Size([nO, nA, nO, nA, nO, nA])
            is the Q function at step h. The last dimension is the action a_h, the rest are history coordinates.

        Q_function[h][history].shape: torch.Size([nA])
            is the Q function vector at step h, conditioned on history: Q_f(\cdot;f_h), with different actions

        Q_function[h][history][a] is Q_h(a;f_h)

    value_function: tensor list of length H
        each element value_function[h].shape :  torch.Size([4, 2, 4, 2, 4]) is the value function at step h.
    '''
    beta_vector=[torch.ones_like(risk_belief[h],dtype=torch.float64) for h in range(horizon_length+1)] 

    Q_function=[torch.zeros(risk_belief[h].shape[:-1]+(size_of_action_space,),dtype=torch.float64) for h in range(horizon_length)]

    value_function=[torch.zeros(risk_belief[h].shape[:-1],dtype=torch.float64) for h in range(horizon_length)]

    return (risk_belief, beta_vector, Q_function, value_function)


def init_occurrence_counters(H:int, nS:int, nO:int, nA:int)->tuple:
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
    return (Ns_init, Nssa, Nssa_ones, Nsa, Nos, Nos_ones, Ns)
