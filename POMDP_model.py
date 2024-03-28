import numpy as np
import torch
import yaml

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

def get_random_dist(dim:int,dist_type:str)->np.array:
    '''
    dist_type='random':   generate a random distribution that normalize to one
    dist_type='uniform':   generate a uniform distribution
    '''
    if dist_type=='uniform':
        return 1/dim*np.ones([dim])
    elif dist_type=='random':
        dist=np.abs(np.random.randn(dim))
        return dist/sum(dist)
    else:
        raise(NotImplementedError)

def sample_from(dist)->int:
    '''
    generate one sample from a given distribution 'dist'.
    '''
    return int(np.random.choice(a=list(np.arange(len(dist))), size=1, p=(dist)))

def initialize_reward(nS:int, nA:int, H:int, init_type:str)->torch.Tensor:
    '''
    inputs:
        sizes of the (s,a) spaces, horizon length H, and initiazation type(see below)
        dist_type='random':    the rewards of each (h,s,a) are not necessarily the same and are randomly picked, however different at each h.
        dist_type='uniform':   rewards on the entire (s,a) space are identical and randomly chosen
        dist_type='ergodic':   rewards of all (h,s,a) are identically chosen random value.
    returns:    
        reward.shape: torch.Size([H,nS,nA])
    description:
        reward[h][s][a] stands for r_h(s,a), which picks value in [0,1]
    '''
    if init_type=='uniform':
        r_unif=np.random.rand()
        reward=torch.ones([H,nS,nA])*r_unif
        return reward
    elif init_type=='random':
        reward=torch.zeros([H,nS,nA])
        for h in range(H):
            reward[h]=torch.rand([nS,nA])
        return reward
    elif init_type=='ergodic':
        reward_layer=torch.rand([nS,nA]).unsqueeze(0)
        reward = reward_layer.repeat(H,1,1)
        return reward
    else:
        raise(NotImplementedError)


def initialize_model(nS:int,nO:int,nA:int,horizon:int,init_type:str)->tuple:
    '''
    inputs
        nS, nO, nA: integers. 
            sizes of the three spaces
        init_type:  string
            dist_type='random':   generate a random distribution that normalize to one
            dist_type='uniform':   generate a uniform distribution
    returns
        a tuple of three tensors. 
        the POMDP model  (mu, T, O), or (init state distributinon,  transition kernels,  emition kernels). 
        the shapes of the kernels:
        mu: torch.Size([nS])
        T : torch.Size([horizon,nS,nS,nA])
        O : torch.Size([horizon+1,nO,nS])
    remark
        The distributions are initialized as a randomly picked distribution, with the only constraint is that the distribution add up to 1.00
    '''

    '''
    initial distribution
    name: init_dist
    shape: torch.Size([nS])
    normalization: sum(mu) = tensor(1.0000, dtype=torch.float64)
    initialization: random distribution. 
    '''
    init_dist=torch.tensor(get_random_dist(dim=nS,dist_type=init_type))
    
    
    '''
    transition matrices
    name:trans_kernel
    shape: torch.Size([horizon,nS,nS,nA])
    access: trans_kernel[h][:,s,a] is the distribution of \mathbb{T}_{h}(\cdot|s,a) \in \Delta(\mathcal{S})    for h =1,2,...H
    normalization: sum(trans_kernel[h][:,s,a])=tensor(1.0000, dtype=torch.float64)
    initialization: random distribution.
    notice: we will not collect s_{H+1}, but set T_{H}(\cdot|sH,aH) as \delta(s_{H+1}-0), i.e. the H+1-th state is an absorbing state 0.
    '''
    #trans_kernel=torch.transpose(torch.tensor( [ [ [get_random_dist(dim=nS) for s in range(10)] for a in range(20) ] for h in range(30) ]  ),1,3)
    trans_kernel=torch.transpose(torch.tensor( np.array([ np.array([ np.array([get_random_dist(dim=nS,dist_type=init_type) for s in range(nS)]) for a in range(nA) ]) for h in range(horizon) ])  ),1,3)

    '''
    emission matrices
    name:emit_kernel
    shape: torch.Size([horizon+1,nO,nS])
    access: emit_kernel[h][:,s] is the distribution of \mathbb{O}_{h}(\cdot|s) \in \Delta(\mathcal{O})
    normalization: sum(emit_kernel[h][:,s])=tensor(1.0000, dtype=torch.float64)
    initialization: random distribution. 
    '''
    #emit_kernel=torch.transpose(torch.tensor([[get_random_dist(nO) for _ in range(nS)] for _ in range(horizon+1)]),1,2)
    emit_kernel=torch.transpose(torch.tensor(np.array([np.array([get_random_dist(dim=nO,dist_type=init_type) for _ in range(nS)])for _ in range(horizon+1)])),1,2)
    
    return (init_dist.to(torch.float64),trans_kernel.to(torch.float64),emit_kernel.to(torch.float64))

def sample_trajectory(horizon:int, policy, model, reward, output_reward=False):
    '''
    inputs:
        horizo length "horizon"
        policy list "policy"
        probabiliy kernels of the environment "model"
        whether this sampler returns the rewards at each step: record_reward
    returns:
        1. only when "output_reward" is false:
            a 3x(H+1) integer list 'full_traj' of the form: 
            s0 s1 s2 ... sH    sh=full_traj[0][h]
            o0 o1 o2 ... oH    oh=full_traj[1][h]
            a0 a1 a2 ... -1    ah=full_traj[2][h]  
        2. only when "output_reward" is true:
            a (H)-shape np.float64 list, named 'sampled_reward'
            r0 r1 ...rH-1      rh=full_traj[3][h]  
    description:
        sample a trajectory from a policy and a POMDP model with given horizon length.
        we will not sample the last action which is in fact out of horizon. AH+1 or a[H] will be denoted as -1.
    '''

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
                # print(f"sample a reward of {sampled_reward[h]} at (h,s,a)={h,state,action}")
        # S_h+1 \sim \mathbb{T}_{h}(\cdot|s_{h},a_{h})
        if h<horizon:  #do not record s_{H+1}
            new_state=sample_from(trans_kernel[h][:,state,action])
            full_traj[0][h+1]=new_state

    if output_reward:
        return sampled_reward
    return full_traj



def initialize_policy(nO:int,nA:int,H:int):
    '''
    function:
        Initialize a stochastic, history-dependent policy that receives observable history of [O_1,A_1,\cdots, O_H] 
        and returns a policy table  \pi = \{ \pi_h(\cdot|\cdot)  \} as a high-dimensional tensor.

    inputs:
        sizes of the observation and action space (nO,nA), and the horizon length H

    returns:
        length H tensor list.
        total dimension: 2H+1
        .shape =torch.Size([horizon,        nO, nA, nO, nA, nO, nA, ... nO,        nA])

        dim[0]: 
            horizon index h
            policy_table[h] indicates \pi_h(\cdot|\cdot)
        dim[1]~dim[2H-1]: 
            o1 a1 o2 a2...oH history
        dim[2H]:
            the distribution of ah, initialized as uniform distribution.

    How to access the policy_table:
    policy[h][f] is a tensor of shape torch.Size([nA]), which corresponds to \pi_h(\cdot|f_h).   policy[h][history].shape=torch.Size([nA])
    policy[h][f][a] is \pi_h(a|f_h)
    To see the shapes:
        In [219]: for h in range(H):
        ...:     print(policy[h].shape)
        ...:
        torch.Size([4, 2])
        torch.Size([4, 2, 4, 2])
        torch.Size([4, 2, 4, 2, 4, 2])
        torch.Size([4, 2, 4, 2, 4, 2, 4, 2])
        torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2])

    How to sample from the policy:
    input some tuple fh of length 2H-1 
        fh=(1, 0, 1, 0, 1, 0, 1, 0, 1)
    input the horizon number h  (so that only the first 2h-1 elements of fh will be valid)
        h=2
    then the distribution of a_h \sim \pi_h(\cdot|fh) is 
        dist=policy[h][f].unsqueeze(0)
    which is of shape
        torch.Size([1, nA])
    To sample from an action from the policy at step h with history f_h, run this line:
        ah=sample_from(dist)
    '''

    policy=[None for _ in range(H)]
    for h in range(H):
        policy[h]=(torch.ones([nO if i%2==0 else nA for i in range(2*h)]+[nO] +[nA])*(1/nA)).to(torch.float64)
    return policy


def action_from_policy(raw_history:tuple,h:int,policy)->int:
    '''
    sample an action from policy_table at step h, with previous observable history indicated by a tuple 'history'.
    '''
    # convert the two-row observable history "raw_history" to a one-row oaoaoa tuple.
    history=[-1 for _ in range(2*h+1)]
    for t in range(h):
        history[2*t]=int(raw_history[0][t])
        history[2*t+1]=int(raw_history[1][t])
    history[2*h]=int(raw_history[0][h])

    history=tuple(history)

    # retrieve \pi_h(\cdot|f_h)   policy[h][history] .shape=torch.Size([nA])
    action_distribution=policy[h][history] 

    '''console debug output:
    #print(f"h={h}, action_distribution={action_distribution}, shape={action_distribution.shape}")
    '''

    # a_h \sim \pi_h(\cdot|f_h)
    action = sample_from(action_distribution)
    return action

'''
a=A.mean(-1,keepdim=True)
c=a.expand((-1,)*2+(3,))
'''

def show_trajectory(*args,record_reward=False):
    if record_reward:
        _, rewards=args
        print(f"rewards at the first H steps:")
        print(f"reward={rewards}")
    else:
        traj=args
        hor=len(traj[0])-1
        print(f"sampled a full trajectory with horizon length={hor}. We did not record the action and rewards at H+1-th step.")
        print(f"horizn={np.array([i+1 for i in range(hor)])}")
        print(f"states={traj[0]}")
        print(f"obsers={traj[1]}")
        print(f"action={traj[2]}")


def test_sampling():
    model=initialize_model(nS,nO,nA,H,init_type='uniform')
    
    policy=initialize_policy(nO,nA,H)

    reward=initialize_reward(nS,nA,H,"random")
    
    with_reward=True  # will also record the rewards obtained during the interaction.

    if with_reward:
        rewards=sample_trajectory(H,policy,model,reward,output_reward=with_reward)
        show_trajectory(None, rewards, record_reward=with_reward)
    else:
        traj=sample_trajectory(H,policy,model,reward,output_reward=with_reward)
        show_trajectory(traj, record_reward=with_reward)

def test_policy():
    '''
    view the shapes of the policies. 
    run this function we should obtain:
    with (nO,nA)=(4,2)
    @ h=0, policy[0].shape=torch.Size([4, 2])
    @ h=1, policy[1].shape=torch.Size([4, 2, 4, 2])
    @ h=2, policy[2].shape=torch.Size([4, 2, 4, 2, 4, 2])
    @ h=3, policy[3].shape=torch.Size([4, 2, 4, 2, 4, 2, 4, 2])      
    @ h=4, policy[4].shape=torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2])
    '''
    policy=initialize_policy(nO,nA,H)
    normalized=True
    for h in range(H):
        print(f"@ h={h}, policy[{h}].shape={policy[h].shape}")

# test_sampling()

# test_policy()