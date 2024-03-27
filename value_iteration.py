import torch
import numpy as np 
import pandas
import torch.nn.functional as F
import yaml 
import itertools

# load the RL platform
from POMDP_model import initialize_model, initialize_policy, initialize_reward, sample_trajectory

from POMDP_model import Nsa, Ns



'''

for o1 in range(nO):
    f=(o1,)
    print(f"\t f={f}")
    for a in range(nA):
        for o in range(nO):
            ff=f+(a,)+(o,)
            print(f"\t ff={ff}")
output:
    
for o1 in range(nO):
    f=(o1,)
    
    # assign value to all g[o1]

for h in range(H):
    for f in (F[h]):
        # assign value to g[f]
        for a in range(nA):
            for o in range(nO):
                ff=f+(a,)+(o,)
                # assign value to each g[(f,a,o)]=G(g_1(a,o), g_2(f))

'''

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


# Bonus residues, correspond to \mathsf{t}_h^k(\cdot,\cdot)  and  \mathsf{o}_{h+1}^k(s_{h+1})
bonus_res_t=torch.ones([H,nS,nA]).to(torch.float64)
bonus_res_o=torch.ones([H,nS]).to(torch.float64)

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
history_space=[None for _ in range(H)]
for h in range(H):
    # Create the space of \mathcal{F}_h = (\mathcal{O}\times \mathcal{A})^{h-1}\times \mathcal{O}
    history_space[h]=[observation_space if i%2==0 else action_space for i in range(2*(h))]+[observation_space]

'''
Create the series of (empirical) risk-sensitive beliefs
sigma :  \vec{\sigma}_{h,f_h} \in \R^{S}
'''
sigma_hat=[None for _ in range(H)]
for h in range(H):
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

beta_hat=torch.ones_like(sigma_hat)

'''
To view how each layer evolves, run:
# for test only
for h in range(H):
    sigma_hat[h]=torch.ones_like(sigma_hat[h])*(-114514.00)

'''
for k in range(K):
    # line 6 in the original paper.
    sigma_hat[0]=mu_hat.unsqueeze(0).repeat(nO,1)
    # line 7 to 9 in the original paper.
    for h in range(1,H,1):
        #print(f"progress:{[torch.min(sigma_hat[t]).item() for t in range(H)]}")   # view how each layer evolves.
        history_coordinates=list(itertools.product(*history_space[h]))
        for hist in history_coordinates:
            #previous history f_{h-1}, act: a_{h-1}, obs: o_h
            prev_hist, act, obs=hist[0:-2], hist[-2], hist[-1]   
            # use Eqs.~\eqref{40} in the original paper to simplify the update rule.
            # be aware that we should use @ but not * !!!   * is Hadamard product while @ is matrix/vector product.
            sigma_hat[h][hist]=np.double(nO)*torch.diag(O_hat[h][obs,:]).to(dtype=torch.float64) @ T_hat[h,:,:,act]  @  torch.diag(torch.exp(gamma* reward[h,:,act])).to(dtype=torch.float64) @ sigma_hat[h-1][prev_hist]
    # line 11 of the original paper
    bonus_res_t=torch.min(torch.ones([H,nS,nA]), 3*torch.sqrt(nS*H*iota / Nsa))
    bonus_res_o=torch.min(torch.ones([H,nS]), 3*torch.sqrt(nO*H*iota/Ns))
    
    # line 12 of the original paper. Notice that h starts from 0 in pytorch it's different from the original paper.
    for h in range(H-1):
        bonus[h]=np.fabs(np.exp(gamma*(H-h))-1)*\
            torch.min(torch.ones([H,nS,nA]), \
                    bonus_res_t[h]+torch.tensordot(bonus_res_o[h+1].to(torch.float64), T_hat[h], dims=1))
    # pay special attenstion to the terminal state: s_{H+1} is absorbed to the same state 0.
    bonus[H-1]=np.fabs(np.exp(gamma)-1)*torch.min(torch.ones([H,nS,nA]), bonus_res_t[h]+\
                                                torch.ones([H,nS,nA])*np.min(1,3*np.sqrt(nO*H*iota/k)))
    
