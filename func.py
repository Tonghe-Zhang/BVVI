
# mischellaneous functions
import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import sys

def test_normalization_T(T_hat,nS,nA,H):
            normalized=True
            for h in range(H):
                for s in range(nS):
                    for a in range(nA):
                        if (np.fabs(sum(T_hat[h][:,s,a])-1)>1e-6):
                            print(f"T_hat[{h}][:,{s},{a}]={T_hat[h][:,s,a]}, sum={sum(T_hat[h][:,s,a]):.10f}")
                            raise(ValueError)
                            return False
            return True

def test_normalization_O(O_hat,nS,H):
            normalized=True
            for h in range(H+1):
                for s in range(nS):
                        if (np.fabs(sum(O_hat[h][:,s])-1)>1e-6):
                            print(f"O_hat[{h}][:,{s}]={O_hat[h][:,s]}, sum={sum(O_hat[h][:,s]):.10f}")
                            raise(ValueError)
                            return False
            return True

# (x)^+ and (x)^- functions.
def negative_func(x:np.float64)->np.float64:
    return np.min(x,0)
def positive_func(x:np.float64)->np.float64:
    return np.max(x,0)

# obtain current time as a python string
def current_time_str()->str:
    import datetime
    return str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
# load hyper parameters from a yaml file.
def load_hyper_param(hyper_param_file_name:str)->tuple:
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
    with open('log\log.txt',mode='w') as log_file:
        # log_file.write(f"\n\nTest BVVI. Current time={current_time_str()}")
        param_error=np.column_stack((mu_err,T_err,O_err))
        np.savetxt('log\log.txt',param_error)
        log_file.close()

    with open('log\log.txt',mode='r') as log_file:
        loss_curve=np.loadtxt('log\log.txt')
        # print(f"read in {loss_curve.shape[0]} items from File:{'log\log.txt'}" )
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

def log_output_tested_rewards(averge_risk_measure_of_each_episode:np.array,H:int)->None:
    loss_curve=averge_risk_measure_of_each_episode
    indices=np.arange(loss_curve.shape[0])  #*H
    labels_plt=['BVVI(ours)']
    # replace with these lines when we have multiple curves.
    # for id in range(3):
    #     plt.plot((indices),loss_curve[id],label=labels_plt[id])
    plt.plot((indices), loss_curve) #, labels_plt

    plt.title(f'Average Risk Measure of Policies. Horizon H={H}')
    plt.xlabel(f'Episode $k$')    # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Average Risk Measure')         # $\sum_{h=1}^{H}r_h(\mathbf{S}_h,\mathbf{A}_h)$
    
    plt.legend(loc='upper right', labels=labels_plt)
    plt.savefig('plots/Reward'+current_time_str()+'.jpg')
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


def test_policy_normalized(policy_test:list, size_obs:int, size_act:int)->bool:
    '''
    input policy
    output whether this policy is normalized
    '''
    import itertools
    normalized_flag=True
    horizon_len=len(policy_test)
    history_space=init_history_space(horizon_len,nO=size_obs,nA=size_act)
    for h in range(horizon_len):
        # retrieve the policy tensor at step h
        policy_step=policy_test[h]
        # traverse all history coordinates
        history_coordinates=list(itertools.product(*history_space[h]))
        for hist in history_coordinates:
            action_distribution=policy_step[hist]
            if torch.sum(action_distribution).item()!=1:
                normalized_flag=False
                raise(ValueError)
                return normalized_flag
    return True


def test_log_output():
    log_output_tested_rewards(averge_risk_measure_of_each_episode=np.array([1,3,2,4,7]), H=5)

def test_output_log_file(output_to_log_file=True):
    import sys
    if output_to_log_file:
        old_stdout = sys.stdout
        log_file = open("log\console_output.log","w")
        sys.stdout = log_file
    print('%'*100)
    print('test Beta Vector Value Iteration.')
    print('%'*100)
    print('hyper parameters:{}')
    with open('config\hyper_param.yaml') as hyp_file:
        content=hyp_file.read()
    print(content)
    print('%'*100)
    print('Call function \'  beta_vector_value_iteration...\' ')
    
    print('\'  beta_vector_value_iteration...\' returned.')
    print('%'*100)
    print('Call function \'  visualize_performance...\' ')
    
    print('\'  visualize_performance...\' returned.')
    print('%'*100)
    print('Beta Vector Value Iteration test complete.')
    print('%'*100)
    if output_to_log_file is True:
        sys.stdout = old_stdout
        log_file.close()


class Logger(object):
    
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log\console_output.log", "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

def save_model_rewards(kernels, reward_table, parent_directory):
    mu, T, O =kernels
    torch.save(mu,parent_directory+'\mu.pt')
    torch.save(T,parent_directory+'\T.pt')
    torch.save(O,parent_directory+'\O.pt')
    torch.save(reward_table,parent_directory+'\R.pt')

def load_model_rewards(parent_directory)->tuple:
    mu=torch.load(parent_directory+'\mu.pt')
    T=torch.load(parent_directory+'\T.pt')
    O=torch.load(parent_directory+'\O.pt')
    reward_table=torch.load(parent_directory+'\R.pt')
    kernels=(mu, T, O)
    return (kernels, reward_table)

def save_model_policy(kernels, policy, parent_directory):
    mu, T, O =kernels
    torch.save(mu,parent_directory+'\mu.pt')
    torch.save(T,parent_directory+'\T.pt')
    torch.save(O,parent_directory+'\O.pt')

    policy_dict={id:len for id, len in enumerate(policy)}
    torch.save(policy_dict,parent_directory+'\Policy.pt')

def load_model_policy(parent_directory):
    mu=torch.load(parent_directory+'\mu.pt')
    T=torch.load(parent_directory+'\T.pt')
    O=torch.load(parent_directory+'\O.pt')
    kernels=(mu, T, O)

    policy_dict = torch.load(parent_directory+'\Policy.pt')
    policy=[policy_dict[id] for id in range(len(policy_dict))]
    return (kernels, policy)

# test_log_output()


def short_test(policy,mu_true,T_true,O_true,R_true,only_reward=False):
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
        return np.cumsum(sampled_reward) / np.arange(1, sampled_reward.shape[0]+1)
    #sampled_reward     accumulated_mean = 
    else:
        return full_traj
    