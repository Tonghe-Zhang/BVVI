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
'''
import numpy as np
from utils import load_hyper_param

config_filename='log_episode_naive_long'
nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param('config\\'+config_filename+'.yaml')
for h in range(H):
    history_space_size=np.pow(nO,h)*np.pow(nA,h-1)
    
import itertools
import numpy as np
action_space=tuple(list(np.arange(nA)))
pi_s=[None for _ in range(H)]
pii_s=[None for _ in range(H)]
for h in range(H):
    pi_s[h]=[action_space for _ in range(int(pow(nO,h+1)*pow(nA,h)))]
    pii_s[h]=list(itertools.product(*pi_s[h]))
    print('%'*100)
    print(f"h={h}")
    for policy in pii_s[h]:
        print(policy)
'''




import matplotlib.pyplot as plt
import numpy as np
from utils import moving_average
K_end=1000
log_file_directory='log\\'+'log_episode_naive_long'+'.txt'
with open(log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()
        
POMDP_value_smooth=moving_average(POMDP_single_episode_rewards,1)
POMDP_value_mixture=np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards)))
POMDP_optimal_value=max(POMDP_value_mixture)
POMDP_mixture_regret=POMDP_optimal_value-POMDP_value_mixture
indices=np.arange(POMDP_mixture_regret.shape[0])

plt.subplot(4,1)
plt.plot(indices, POMDP_mixture_regret,label='POMDP, Value')

plt.subplt(4,2)
plt.plot(indices, POMDP_mixture_regret,label='POMDP, PAC')

plt.show()

