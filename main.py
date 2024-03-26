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
model_true=initialize_model(nS,nO,nA,H,init_type='random')

model_empirical=initialize_model(nS,nO,nA,H,init_type='uniform')

mu_hat, T_hat, O_hat=model_empirical

policy=initialize_policy(nO,nA,H)

