# Under the setting of:
  size_of_action_space: 2
  size_of_state_space: 3
  size_of_observation_space: 3
  horizon_len: 3
  num_episode: 10
  confidence_level: 0.2
  risk_sensitivity_factor: 1.0

# Set the kernels: Run these commands:
H=3
T_true=T=[torch.tensor([[0,0,1],[1,0,0],[0,1,0]]).unsqueeze(-1).repeat(1,1,2) for _ in range(4)]
O_true=[torch.tensor([[0.4,0.2,0.4],[0.3,0.5,0.2],[0.2,0.7,0.1]]).transpose(0,1).repeat(1,1) for _ in range(4)]
mu_true=torch.tensor([1,0,0])
R_true=torch.tensor([[1,0],[0,1],[1,0]]).unsqueeze(0).repeat(3,1,1)


T_true=torch.tensor([[0,0,1],[1,0,0],[0,1,0]]).unsqueeze(0).unsqueeze(-1).repeat(4,1,1,2).to(torch.float32)
O_true=torch.tensor([[0.4,0.2,0.4],[0.3,0.5,0.2],[0.2,0.7,0.1]]).unsqueeze(0).transpose(0,1).repeat(4,1,1).to(torch.float32)
mu_true=torch.tensor([1,0,0]).to(torch.float32)
R_true=torch.tensor([[1,0],[0,1],[1,0]]).unsqueeze(0).repeat(3,1,1).to(torch.float32)


T[h][:,:,a]=
tensor([[0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])
In [134]: T[0][:,:,0]
Out[134]:
tensor([[0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])

In [135]: T[0][:,:,1]
Out[135]:
tensor([[0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])

O_true[h]=
tensor([[0.4000, 0.3000, 0.2000],
        [0.2000, 0.5000, 0.7000],
        [0.4000, 0.2000, 0.1000]])
In [129]: O_true[0][:,1]
Out[129]: tensor([0.3000, 0.5000, 0.2000])

In [130]: O_true[0][:,0]
Out[130]: tensor([0.4000, 0.2000, 0.4000])

In [131]: O_true[0][:,2]
Out[131]: tensor([0.2000, 0.7000, 0.1000])

In [131]: O_true[0][:,3]
Out[131]: tensor([0.2000, 0.7000, 0.1000])

mu_true=
torch.tensor([1,0,0])

R=tensor([[[1, 0],
         [0, 1],
         [1, 0]],

        [[1, 0],
         [0, 1],
         [1, 0]],

        [[1, 0],
         [0, 1],
         [1, 0]]])
In [148]: R_true[0][:,:]
Out[148]:
tensor([[1, 0],
        [0, 1],
        [1, 0]])

# Maximum risk = 1/1 e^{1*(1+1+1)}=e^3=20.085536923187668

# Optimal policy is deterministi:
h=0, a=0
h=1, a=1
h-2, a=0
# run these lines to implement such policy:

policy_star=[None for _ in range(3)]
policy_star[0]=torch.tensor([1,0]).unsqueeze(0).repeat(3,1)
policy_star[1]=torch.tensor([0,1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3,2,3,1)
policy_star[2]=torch.tensor([1,0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3,2,3,2,3,1)
policy_test=policy_star

# run everything to test the optimal policy:

def short_test(only_reward=False):
    from POMDP_model import sample_from, action_from_policy

    horizon=3
    policy=policy_star
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
        return sampled_reward


short_test(False)
sampled_reward=short_test(True)


num_samples=10
tested_risk_measure=(1/gamma)*np.array([np.exp(gamma*sum(short_test(True))) for _ in range(num_samples)]).mean()

# Max reward should be:
20.085536923187668


# Only test the dynamic programming of BVVI

from utils import load_hyper_param

nS,nO,nA,H,K,nF,delta,gamma,iota = load_hyper_param("config\hyper_param_naive.yaml")

prt_progress=True
prt_policy_normalization=True

model_true_load=(mu_true, T_true, O_true)
reward_true_load=R_true

mu_err=np.zeros([K])
T_err=np.zeros([K])
O_err=np.zeros([K])
tested_returns=np.zeros([K])
evaluation_metrics=(mu_err, T_err, O_err, tested_returns)

with open('log\log_episode_naive.txt',mode='r+') as log_episode_file:
    (policy, model_learnt, evaluation_results)=beta_vector_value_iteration(\
                model_true=model_true_load,\
                    reward=reward_true_load,\
                        model_load=model_true_load,\
                            policy_load=reward_true_load,\
                                evaluation_metrics=evaluation_metrics,\
                                    log_episode_file=log_episode_file)

# unpack
mu_err,T_err,O_err, tested_risk_measure=evaluation_results

# plot planning result.
log_output_tested_rewards(tested_risk_measure,H)

# plot parameter learning results
log_output_param_error(mu_err,T_err,O_err, H)