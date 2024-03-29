import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import itertools
import sys
import time
from tqdm import tqdm

from func import load_hyper_param, short_test, visualize_performance, log_output_test_reward_pretty, current_time_str, Logger
from POMDP_model import initialize_model_reward
from BVVI import BVVI

def main(config_filename='hyper_param_naive',
         model_true=None,
         reward_true=None,
         model_load=None,
         policy_load=None,
         output_to_log_file=False,
         log_episode_filename='log_episode_naive',
         prt_progress=True,
         prt_policy_normalization=True,
         weight_output_parent_directory='learnt\\naive'
         ):
    '''
    model_true if None, then randomly initialize one.
    '''
    # load hyper parameters
    hyper_param= load_hyper_param('config\\'+config_filename+'.yaml')    # can delete the naive
    nS,nO,nA,H,K,nF,delta,gamma,iota =hyper_param

    # initialize true model and rewards if necessary
    if model_true ==None:
        model_true,reward_true=initialize_model_reward(nS,nO,nA,H,model_init_type='random_homogeneous', reward_init_type='random_homogeneous')

    # [Evaluation] Reset the parameter errors and accumulated returns tested in the true envirnoemt of each iteration.
    mu_err=np.zeros([K])
    T_err=np.zeros([K])
    O_err=np.zeros([K])
    tested_returns=np.zeros([K])
    evaluation_metrics=(mu_err, T_err, O_err, tested_returns)

    # start the algorithm and logging instructions.
    if output_to_log_file:
        print(f"Will output log information to both the file:{'console_output.log'} and the console.")
        old_stdout = sys.stdout
        log_file = open("log\console_output.log","w")
        sys.stdout = Logger() #sys.stdout = log_file
        print(f"Start BVVI test. Current time={current_time_str()}")
        time.sleep(3)

    print('%'*100)
    print('test Beta Vector Value Iteration.')
    print('%'*100)
    print('hyper parameters:{}')
    with open('config\\'+config_filename+'.yaml') as hyp_file:  # can remove naive.
        content=hyp_file.read()
    print(content)
    print('%'*100)
    print('Call function beta_vector_value_iteration...')

    with open('log\\'+log_episode_filename+'.txt',mode='w') as log_episode_file:
        # log_episode_file.write(f"\n\nTest BVVI. Current time={current_time_str()}")   #real_env_kernels reward_fix
        (policy_learnt, model_learnt, evaluation_results)=BVVI(\
            hyper_param=hyper_param,\
                model_true=model_true,\
                    reward_true=reward_true,\
                        model_load=model_load,\
                            policy_load=policy_load,\
                                evaluation_metrics=evaluation_metrics,\
                                    log_episode_file=log_episode_file,\
                                        weight_output_parent_directory=weight_output_parent_directory,\
                                            prt_progress=prt_progress,\
                                                prt_policy_normalization=prt_policy_normalization)
        # log_episode_file.write(f"\n\nEnd Testing BVVI. Current time={current_time_str()}")
        log_episode_file.close()
    episode_data=np.loadtxt('log\\'+log_episode_filename+'.txt', dtype=np.float64)
    print('beta_vector_value_iteration...returned.')
    print(f"End BVVI test. Current time={current_time_str()}")
    print('%'*100)
    print('Call function visualize_performance...')

    
    visualize_performance(evaluation_results,H)

    print('visualize_performance...returned.')
    print('%'*100)
    print('Beta Vector Value Iteration test complete.')
    print('%'*100)
    
    if output_to_log_file is True:
        sys.stdout = old_stdout
        log_file.close()
    return policy_learnt


def test_with_naive_env():
    mu_true=torch.tensor([1,0,0])
    T_true=torch.stack([torch.tensor([[0,0,1],[1,0,0],[0,1,0]]).unsqueeze(-1).repeat(1,1,2) for _ in range(3)])
    O_true=torch.stack([torch.tensor([[0.4,0.2,0.4],[0.3,0.5,0.2],[0.2,0.7,0.1]]).transpose(0,1).repeat(1,1) for _ in range(4)])
    R_true=torch.tensor([[1,0],[0,1],[1,0]]).unsqueeze(0).repeat(3,1,1)

    model_true=(mu_true, T_true, O_true)
    reward_true=R_true

    policy_learnt=main(config_filename= 'hyper_param_naive',
                        model_true=model_true,
                        reward_true=reward_true,
                        model_load=None,
                        policy_load=None,
                        output_to_log_file=True,
                        log_episode_filename= 'log_episode_naive',
                        prt_progress=False,
                        prt_policy_normalization=False,
                        weight_output_parent_directory='learnt\\naive'
                        )
    
    # if we are training from naive params, also run this line:
    log_output_test_reward_pretty(H=T_true.shape[0],gamma=1.0, plot_optimal_policy=True, optimal_value=np.exp(T_true.shape[0]), log_episode_file_name='log_episode_naive')

    print('%'*100)
    print("short test of policy")
    short_test(policy_learnt,mu_true,T_true,O_true,R_true,only_reward=False)
    # print(policy_learnt)

def test_with_medium_random_env():

    model_true,reward_true=initialize_model_reward()

    policy_learnt=main(config_filename= 'hyper_param_naive',
                        model_true=model_true,
                        reward_true=reward_true,
                        model_load=None,
                        policy_load=None,
                        output_to_log_file=True,
                        log_episode_filename= 'log_episode_naive',
                        prt_progress=False,
                        prt_policy_normalization=False,
                        weight_output_parent_directory='learnt\\naive'
                        )


if __name__ == "__main__":
    test_with_naive_env()


