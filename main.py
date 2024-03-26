import torch
import numpy as np 
import pandas
import torch.nn.functional as F
import yaml 

# load the RL platform
from POMDP_model import initialize_model, initialize_policy, sample_trajectory

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

# the annotated line numbers correspond to the lines on page 22 of the original paper. 
model_true=initialize_model(nS,nO,nA,init_type='random')







model_empirical=initialize_model(nS,nO,nA,init_type='uniform')

mu_hat, T_hat, O_hat=model_empirical

policy=initialize_policy(nO,nA,H)

# test Monte-Carlo Learning
for k in range(1,K+1,1):
    # line 29-30    
    full_trajectory=sample_trajectory(H,policy,model_true)

    # line 31 to 35 
    for h in range(0,H,1):
        # line 33
        [s,o,a,ss]=torch.cat((full_trajectory[0:3,h],full_trajectory[0:1,h+1])).to(torch.int64)    # overflow alert!H+1
        # line 34
        N_hatS[h][s][a]=N_hatS[h][s][a]+1
        N_hatO[h][s]   =N_hatO[h][s]+1
        # line 35: update T, nO, \mu estimates.
        OS_cnt[h][o][s]=OS_cnt[h][o][s]+1
        SSA_cnt[h][ss][s][a]=SSA_cnt[h][ss][s][a]+1
        T_hat[h,:,:,:]=SSA_cnt[h]/(torch.max(torch.ones([nS,nS,nA]),N_hatS[h,:,:].repeat(nS,1,1)))
        O_hat[h,:,:]=OS_cnt[h]/(torch.max(torch.ones([nO,nS]),N_hatO[h,:].repeat(nO,1)))
    s_1_hat=full_trajectory[0][1].to(torch.int64)
    mu_hat=1/k*(F.one_hot(s_1_hat,nS))+(1-1/k)*mu_hat


