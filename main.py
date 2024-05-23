import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import time
from tqdm import tqdm
import os
from utils import make_all_dirs,init_manual,load_hyper_param, short_test, visualize_performance, log_output_test_reward_pretty, current_time_str, Logger, write_current_time_str
from POMDP_model import initialize_model_reward
from BVVI import BVVI
from RSVI2 import RSVI2
from plot import BVVI_plot, multi_risk_level_plot,plot_pac,plot_regret, plot_cum_reward,plot_cum_reward_various_risk

def main(Alg:str,
         num_episodes:int,
         config_filename:str,
         log_episode_filename:str,
         model_true=None,
         reward_true=None,
         model_load=None,
         policy_load=None,
         output_to_log_file=False,
         prt_progress=True,
         prt_policy_normalization=True,
         true_weight_output_parent_directory='real_env',
         weight_output_parent_directory=os.path.join('learnt','naive')
         ):
    '''
    model_true if None, then randomly initialize one.
    '''
    # load hyper parameters
    hyper_param= load_hyper_param(os.path.join('config',config_filename+'.yaml'))    # can delete the naive
    nS,nO,nA,H,K,nF,delta,gamma,iota =hyper_param

    # we can change the number of episodes K from the console.
    K=num_episodes

    # initialize true model and/or rewards if necessary
    if model_true ==None:
        print(f"no model_true input, initialize a new one")
        model_true,_=initialize_model_reward(nS,nO,nA,H,model_init_type='random_homogeneous', reward_init_type='random_homogeneous')
    if model_true==None:
        raise(ValueError)
    
    if reward_true ==None:
        print(f"no reward_true input, initialize a new one")
        _, reward_true=initialize_model_reward(nS,nO,nA,H,model_init_type='random_homogeneous', reward_init_type='random_homogeneous')
    if reward_true==None:
        raise(ValueError)
    
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
        log_file = open(os.path.join('./log',current_time_str(),"console_output.log"),"w")
        sys.stdout = Logger() #sys.stdout = log_file
        print(f"Start {Alg }test. Current time={current_time_str()}")
        time.sleep(3)

    print('%'*100)
    print('Test Algorithm.')
    print('%'*100)
    print('hyper parameters==')
    with open(os.path.join('config',config_filename+'.yaml')) as hyp_file:  # can remove naive.
        content=hyp_file.read()
    print(content)
    print(f"[number of episodes K is changed to {K}]")
    print('%'*100)
    print('Call function beta_vector_value_iteration...')

    with open(os.path.join('./log',current_time_str(),log_episode_filename+'.txt'),mode='w') as log_episode_file:
        if Alg=='BVVI':
            (policy_learnt, model_learnt, evaluation_results)=BVVI(\
                hyper_param=hyper_param,\
                    num_episodes=num_episodes,\
                        model_true=model_true,\
                            reward_true=reward_true,\
                                model_load=model_load,\
                                    policy_load=policy_load,\
                                        evaluation_metrics=evaluation_metrics,\
                                            log_episode_file=log_episode_file,\
                                                true_weight_output_parent_directory=true_weight_output_parent_directory,\
                                                    weight_output_parent_directory=weight_output_parent_directory,\
                                                        prt_progress=prt_progress,\
                                                            prt_policy_normalization=prt_policy_normalization)
        elif Alg=='RSVI2':
            (policy_learnt, model_learnt, evaluation_results)=RSVI2(\
                hyper_param=hyper_param,\
                    model_true=model_true,\
                        reward_true=reward_true)
        else:
            raise(NotImplementedError)
        log_episode_file.close()
    episode_data=np.loadtxt(os.path.join('./log',current_time_str(),log_episode_filename+'.txt'), dtype=np.float64)
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

def train_naive_env(Alg:str,
                        num_episodes:int,
                        config_filename:str,
                        log_episode_filename:str,
                        stochastic_transition,
                        identity_emission,
                        peaky_reward:bool):
    
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))# can delete the naive

    mu_true, T_true, O_true,R_true=init_manual(H,
                          stochastic_transition=stochastic_transition,
                          identity_emission=identity_emission,
                          peaky_reward=peaky_reward)
    
    model_true=(mu_true, T_true, O_true)
    reward_true=R_true

    policy_learnt=main(Alg=Alg,
                       config_filename= config_filename,
                        log_episode_filename= log_episode_filename,
                       num_episodes=num_episodes,
                        model_true=model_true,
                        reward_true=reward_true,
                        model_load=None,
                        policy_load=None,
                        output_to_log_file=True,
                        prt_progress=False,
                        prt_policy_normalization=False,
                        true_weight_output_parent_directory=os.path.join('real_env','naive_real_id'),      #'real_env/naive_real'
                        weight_output_parent_directory=os.path.join('learnt','naive_id')                      #'learnt/naive' 
                        )
    
    # if we are training from naive params, also run this line:
    # Note: this only works for gamma=1. otherwise change
    log_output_test_reward_pretty(H=H,K_end=num_episodes,gamma=1.0, plot_optimal_policy=True,
                                  optimal_value=1/gamma*np.log(np.exp(gamma*H)),
                                  log_episode_file_name=log_episode_filename)

    print('%'*100)
    print("short test of policy")
    short_test(policy_learnt,mu_true,T_true,O_true,R_true,only_reward=False)
    # print(policy_learnt)

def naive_train_and_plot(Alg:str,
                         num_episodes:int,
                         config_filename:str,
                         log_filename:str,
                         train_from_scratch,
                         stochastic_transition,
                          identity_emission,
                          peaky_reward:bool, 
                          instant_plot=False):
    # print(f"@@@{config_filename}, {os.path.join('config',config_filename+'.yaml')}")
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join("config",config_filename+'.yaml'))

    # train again on the naive dataset.
    if train_from_scratch:
        train_naive_env(Alg,
                        num_episodes=num_episodes,
                        config_filename=config_filename,
                        log_episode_filename=log_filename,
                        stochastic_transition=stochastic_transition,
                        identity_emission=identity_emission,
                        peaky_reward=peaky_reward)

    if instant_plot:
        optimal_value=1/gamma*np.log(np.exp(gamma*H))
        log_file_directory=os.path.join('./log',current_time_str(),log_filename+'.txt').replace('\\', '/')

        with open(log_file_directory,mode='r') as log_episode_file:
            averge_risk_measure_of_each_episode=np.loadtxt(log_file_directory)[0:num_episodes+1,0]
            '''
            Plot regret, suppose that we know the optimal value funciton.
            '''
            plot_type='regret'   # other options: 'risk_average', 'risk_each'

            risk_measure_smooth=np.cumsum(averge_risk_measure_of_each_episode)/(1+np.arange(len(averge_risk_measure_of_each_episode)))

            regret_curve=optimal_value-averge_risk_measure_of_each_episode

            regret_curve_smooth=np.cumsum(regret_curve)/(1+np.arange(len(regret_curve)))
            
            indices=np.arange(regret_curve_smooth.shape[0])

            if plot_type=='regret':
                plot_curve=regret_curve_smooth
            elif plot_type=='risk_average':
                plot_curve=risk_measure_smooth
            elif plot_type=='risk_each':
                plot_curve=averge_risk_measure_of_each_episode
            plt.plot(indices, plot_curve,label='BVVI(ours)')
            
            # upper and lower bounds of the accumulated risk measure.

            plt.ylim((min(plot_curve)*0.4,max(plot_curve)*1.2))

            plt.title(f'Performance of Output Policies')   # . Horizon H={H}
            
            plt.xlabel(f'Episode $k$')    # H transitions per iteration.   Samples N (=iteration $K$ * {H})
            
            if plot_type=='regret':
                plt.ylabel(f'Average Regret')
            elif plot_type=='risk_average':
                plt.ylabel(f'Average Risk Measure')
            elif plot_type=='risk_each':
                plt.ylabel(f'Risk Measure of Each Episode')
        
            # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join('plots',current_time_str(),'single-Reward.jpg'))

            plt.show()

def train_single_risk(train_from_scratch:bool,
                      config_filename:str,
                      pomdp_log_filename:str,
                      mdp_log_filename:str,
                      plot_all:bool,
                      num_episodes:int)->None:
    if train_from_scratch:
        naive_train_and_plot(Alg='BVVI',
                             num_episodes=num_episodes,
                         config_filename=config_filename,
                         train_from_scratch=True,
                         log_filename=pomdp_log_filename,
                         stochastic_transition=True,
                         identity_emission=False,
                         peaky_reward=False)
    
        naive_train_and_plot(Alg='BVVI',
                             num_episodes=num_episodes,
                         config_filename=config_filename,
                         train_from_scratch=True,
                         log_filename=mdp_log_filename,
                         stochastic_transition=True,
                         identity_emission=True,
                         peaky_reward=False)
    if plot_all:
        BVVI_plot(num_episodes=num_episodes, 
                    window_width_MDP=3,
                     window_width_POMDP=30,
                     config_filename=config_filename,
                    POMDP_log_filename=pomdp_log_filename,
                    MDP_log_filename=mdp_log_filename)


def train_multiple_risk(train_from_scratch, plot_all, num_episodes, gamma_range=[ -5.0,-3.0, -1.0, 0.01, 1.0, 3.0, 5.0]):
    """
    Use the config files from ./config/current_time/various_risk/gamma=... 
    to train the BVVI algorithm.
    Then output log information to  ./log/current_time/various_risk/gamma=... 
    """
    config_files=[ _ for _ in range(len(gamma_range))]
    log_files=config_files 
    for i,gamma in enumerate(gamma_range):
        config_files[i]=os.path.join("various_risk",f"gamma={gamma}")
        log_files[i]=os.path.join("various_risk",f"gamma={gamma}")
    if train_from_scratch:
        for i in range(len(gamma_range)):
            naive_train_and_plot(Alg='BVVI',
                                num_episodes=num_episodes,
                                config_filename=config_files[i],
                                train_from_scratch=True,
                                log_filename=log_files[i],
                                stochastic_transition=True,
                                identity_emission=False,
                                peaky_reward=False)
    if plot_all:
        multi_risk_level_plot(window_width_POMDP=30,
                                config_files=config_files,
                                POMDP_log_files=log_files,
                                num_episodes=num_episodes)

if __name__ == "__main__":
    """
    Run this command in console(only specify those arguments different from their default values):
    
    python main.py --train_single False --gamma_range -1.0 0.01 1.0 3.0 5.0
    
    or this line:
    
    > python main.py --gamma_range -5.0 -3.0 -1.0 0.01 1.0 3.0 5.0 
    
    or this line to avoid keyborad interrupt:
    
    > nohup python main.py --train_single False --gamma_range -1.0 0.01 1.0 3.0 5.0 &
    > nohup python main.py --train_single False --gamma_range -3.0 &
    after using nohup command, you can run 
    
    > job
    
    in the terminal to see whether your code is still running, even after you trun off the terminal console.
    """
    
    """
    train_from_scratch=True 
    plot_all=True
    num_episodes=2000
    gamma_range=[ -5.0,-3.0, -1.0, 0.01, 1.0, 3.0, 5.0]
    config_filename='naive' #'naive-medium'
    log_filename_pomdp='pomdp'
    log_filename_mdp='mdp'
    """

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Example of using argparse in Python")
    # Add arguments
    parser.add_argument('--plot_all', type=str2bool, nargs='?', const=True, default=True, help='Flag to plot all figures')
    parser.add_argument('--train_from_scratch', type=str2bool, nargs='?', const=True, default=True, help='Flag to train the model from scratch')
    parser.add_argument('--train_single', type=str2bool, nargs='?', const=True, default=True, help=r'Flag to train single risk measure $\gamma=1.0$')
    parser.add_argument('--train_various', type=str2bool, nargs='?', const=True, default=True,
                        help=r'Flag to train under multiple risk measure specified by gamma_range')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes for training')
    parser.add_argument('--gamma_range', nargs='+', type=float, default=[-5.0, -3.0, -1.0, 0.01, 1.0, 3.0, 5.0], help='Range of gamma values')
    parser.add_argument('--config_filename', type=str, default='naive', help='Configuration filename')
    parser.add_argument('--log_filename_pomdp', type=str, default='pomdp', help='Log filename for POMDP')
    parser.add_argument('--log_filename_mdp', type=str, default='mdp', help='Log filename for MDP')
    # Parse the arguments
    args = parser.parse_args()
    # Access arguments
    train_from_scratch = args.train_from_scratch
    train_single= args.train_single
    train_various= args.train_various
    plot_all = args.plot_all
    num_episodes = args.num_episodes
    gamma_range = args.gamma_range
    config_filename = args.config_filename
    log_filename_pomdp = args.log_filename_pomdp
    log_filename_mdp = args.log_filename_mdp
    # Print the values (for demonstration)
    print(f'Received the following parameters from console:')
    print(f'\ttrain_from_scratch: {train_from_scratch}')
    print(f'\ttrain_single: {train_single}')
    print(f'\ttrain_various: {train_various}')
    print(f'\tplot_all: {plot_all}')
    print(f'\tnum_episodes: {num_episodes}')
    print(f'\tgamma_range: {gamma_range}')
    print(f'\tconfig_filename: {config_filename}')
    print(f'\tlog_filename_pomdp: {log_filename_pomdp}')
    print(f'\tlog_filename_mdp: {log_filename_mdp}')  

    make_all_dirs()
    
    if train_single:
        train_single_risk(train_from_scratch=train_from_scratch,
                          plot_all=plot_all,
                          num_episodes=num_episodes,
                          config_filename=config_filename,
                          pomdp_log_filename=log_filename_pomdp,
                          mdp_log_filename=log_filename_mdp)
    
    if train_various:
        train_multiple_risk(train_from_scratch=train_from_scratch,
                            plot_all=plot_all,
                            num_episodes=num_episodes,
                            gamma_range=gamma_range)
    
    plot_pac(config_filename=config_filename,
                  POMDP_log_filename=log_filename_pomdp,
                  MDP_log_filename=log_filename_mdp,
                  K_end=num_episodes)
    
    plot_regret(window_width_MDP=3,
                     window_width_POMDP=30,
                     config_filename=config_filename,
                    POMDP_log_filename=log_filename_pomdp,
                    MDP_log_filename=log_filename_mdp,
                    K_end=num_episodes)
    
    plot_cum_reward(config_filename=config_filename,
             POMDP_log_filename=log_filename_pomdp,
             MDP_log_filename=log_filename_mdp,
             K_end=num_episodes)
    
    plot_cum_reward_various_risk(gamma_range=gamma_range,num_episodes=num_episodes)