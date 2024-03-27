import torch
import numpy as np 
import pandas
import torch.nn.functional as F
import yaml 
import itertools

# load the RL platform
from POMDP_model import initialize_model, initialize_policy, initialize_reward, sample_trajectory

def negative_func(x:np.double)->np.double:
    return np.min(x,0)
def positive_func(x:np.double)->np.double:
    return np.max(x,0)


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

# obtain the true environment. invisible for the agent. Immutable. Only used during sampling.
model_true=initialize_model(nS,nO,nA,H,init_type='random')
mu,T,O=model_true

# Initialize the empiricl kernels with uniform distributions.
mu_hat, T_hat, O_hat=initialize_model(nS,nO,nA,H,init_type='uniform')

policy=initialize_policy(nO,nA,H)


################################################################
##### borrowed from test_monte_carlo.py file####################
Nsa=torch.ones([H,nS,nA])
Ns=torch.ones([H+1,nS])
################################################################
################################################################

# Bonus residues, correspond to \mathsf{t}_h^k(\cdot,\cdot)  and  \mathsf{o}_{h+1}^k(s_{h+1})
bonus_res_t=torch.ones([H,nS,nA]).to(torch.float64)
bonus_res_o=torch.ones([H+1,nS]).to(torch.float64)   # in fact there is no h=0 for residue o. we shift everything right.

# Bonus
''''
we should be aware that in the implementation, s_{H+1} is set to be absorbed to 0
so T(s_{H+1}|s_H) = \delta(S_{H+1}=0)
and o_{H+1}^k(s_{H+1})= 1 if s_{H+1} \ne 0   else 3\sqrt{\frac{OH \iota}{k}} 
so bonus_{H} is needs special care.
'''
bonus=torch.ones([H,nS,nA]).to(torch.float64)


# create the history spaces \{\mathcal{F}_h\}_{h=1}^{H}
observation_space=tuple(list(np.arange(nO)))
action_space=tuple(list(np.arange(nA)))
history_space=[None for _ in range(H+1)]
for h in range(H+1):
    # Create the space of \mathcal{F}_h = (\mathcal{O}\times \mathcal{A})^{h-1}\times \mathcal{O}
    history_space[h]=[observation_space if i%2==0 else action_space for i in range(2*(h))]+[observation_space]

'''
Create the series of (empirical) risk-sensitive beliefs
sigma :  \vec{\sigma}_{h,f_h} \in \R^{S}
'''
sigma_hat=[None for _ in range(H+1)]
for h in range(H+1):
    sigma_hat[h]=torch.zeros([nO if i%2==0 else nA for i in range(2*(h))]+[nO] +[nS], dtype=torch.float64)
'''
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


'''
Create beta vectors, Q-values and value functions
beta_hat:       tensor list of length H+1
    beta_hat[h][hist][s] is \widehat{\beta}_{h, f_h}^k(s_h)
Q_function:     tensor list of length H
    each element Q_function[h].shape    torch.Size([nO, nA, nO, nA, nO, nA])
        is the Q function at step h. The last dimension is the action a_h, the rest are history coordinates.

    Q_function[h][history].shape: torch.Size([nA])
        is the Q function vector at step h, conditioned on history: Q_f(\cdot;f_h), with different actions

    Q_function[h][history][a] is Q_h(a;f_h)

value_function: tensor list of length H
    each element value_function[h].shape :  torch.Size([4, 2, 4, 2, 4]) is the value function at step h.
'''

beta_hat=[torch.ones_like(sigma_hat[h],dtype=torch.float64) for h in range(H+1)] 

Q_function=[torch.zeros(sigma_hat[h].shape[:-1]+(nA,),dtype=torch.float64) for h in range(H)]

value_function=[torch.zeros(sigma_hat[h].shape[:-1],dtype=torch.float64) for h in range(H)]


'''
To view how each layer evolves, run:
# for test only
for h in range(H):
    sigma_hat[h]=torch.ones_like(sigma_hat[h])*(-114514.00)
'''
for k in range(K):
    print(f"Into episode {k}/{K}={k/K}")

    # %%%%%%% Belief propagation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            sigma_hat[h][hist]=np.double(nO)*torch.diag(O_hat[h][obs,:]).to(dtype=torch.float64) @ T_hat[h-1,:,:,act]  @  torch.diag(torch.exp(gamma* reward[h-1,:,act])).to(dtype=torch.float64) @ sigma_hat[h-1][prev_hist]
    # line 11 of the original paper
    bonus_res_t=torch.min(torch.ones([H,nS,nA]), 3*torch.sqrt(nS*H*iota / Nsa))
    bonus_res_o=torch.min(torch.ones([H+1,nS]), 3*torch.sqrt(nO*H*iota/Ns))
    
    # line 12 of the original paper. Notice that h starts from 0 in pytorch it's different from the original paper.
    for h in range(H):
        bonus[h]=np.fabs(np.exp(gamma*(H-h))-1)*\
            torch.min(torch.ones([nS,nA]), \
                bonus_res_t[h]+torch.tensordot(bonus_res_o[h+1].to(torch.float64), T_hat[h], dims=1))
    print(f"\t\t belief propagation ends...")
    # %%%%%% Dynamic programming %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        max_indices=torch.argmax(Q_function[h],dim=-1,keepdim=True)
        policy[h]=torch.zeros_like(policy[h]).scatter(dim=-1,index=max_indices,src=Q_function[h])

        # action_greedy is \widehat{\pi}_h^k(f_h)
        action_greedy=torch.argmax(policy[h][hist]).item()
        
        # line 23 in the original paper.
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
        print(f"\t\tfinish horizon {h}/{H}")



'''
next_obs=0
            next_state=0
            print(f"hist+(action_greedy,next_obs,)={hist+(action_greedy,next_obs,)}")
            print(f"beta_hat[h+1][hist+(action_greedy,next_obs,)][next_state]={beta_hat[h+1][hist+(action_greedy,next_obs,)][next_state]}")
            print(f"O_hat[h+1][next_obs][next_state]={O_hat[h+1][next_obs][next_state]}")
            print(f"list={}")


            [ O_hat[h+1][next_obs][next_state]*beta_hat[h+1][hist+(action_greedy,next_obs,)][next_state] for next_obs in range(nO) ]


            print(f"sum={sum([O_hat[h+1][next_obs][next_state]*beta_hat[h+1][hist+(action_greedy,next_obs,)][next_state]] for next_obs in range(nO))}")


'''
