import torch
import numpy as np 
import pandas
import torch.nn.functional as F
import yaml 
import itertools

# load the RL platform
from POMDP_model import initialize_model, initialize_policy, sample_trajectory


def value_iteration():
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
    reward=torch.tensor([H,nS,nA])


    # obtain the true environment. invisible for the agent. Immutable. Only used during sampling.
    model_true=initialize_model(nS,nO,nA,H,init_type='random')
    mu,T,O=model_true

    # Initialize the empiricl kernels with uniform distributions.
    mu_hat, T_hat, O_hat=initialize_model(nS,nO,nA,H,init_type='uniform')

    policy=initialize_policy(nO,nA,H)






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



def his_iter():
    nA=2
    nO=4
    H=3

    obs_space=tuple(list(np.arange(nO)))
    act_space=tuple(list(np.arange(nA)))
    his_space=[None for _ in range(H)]
    for h in range(H):
        # Create the space of \mathcal{F}_h = (\mathcal{O}\times \mathcal{A})^{h-1}\times \mathcal{O}
        his_space[h]=[ obs_space if i%2==0 else act_space for i in range(2*(h))]+[obs_space]

        all_coordinates=list(itertools.product(*his_space[h]))
        print(f"At h={h}: possible coordinates are")
        for coord in all_coordinates:
            print(f"\t\t{coord}")
his_iter()