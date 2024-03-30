'''

In this module we define the algorithms to calculate the optimal policy of 
acting in a POMDP. 
We will use naive dynamic programming to solve this problem.
We will use the true model parameters. 
No bonus will be added.  
No measure-change, just simply viewing the history space as the states. 
Reference: 

'''

'''
history_coordinates=list(itertools.product(*history_space[h]))
            for hist in history_coordinates:
        
observation_space=tuple(list(np.arange(nO)))
    action_space=tuple(list(np.arange(nA)))
    history_space=[None for _ in range(H+1)]
    for h in range(H+1):
        # Create the space of \mathcal{F}_h = (\mathcal{O}\times \mathcal{A})^{h-1}\times \mathcal{O}
        history_space[h]=[observation_space if i%2==0 else action_space for i in range(2*(h))]+[observation_space]
    return history_space
'''

import numpy as np
from func import load_hyper_param

config_filename='log_episode_naive_long'
nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param('config\\'+config_filename+'.yaml')
for h in range(H):
    history_space_size=np.pow(nO,h)*np.pow(nA,h-1)
    single_step_policy_space=