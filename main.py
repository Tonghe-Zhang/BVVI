import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import itertools
import sys
import time
from tqdm import tqdm
from scipy.optimize import curve_fit
from func import smooth

from func import load_hyper_param, short_test, visualize_performance, log_output_test_reward_pretty, current_time_str, Logger, load_model_rewards, load_model_policy
from POMDP_model import initialize_model_reward
from BVVI import BVVI
from RSVI2 import RSVI2

def main(Alg:str,
         config_filename='hyper_param_naive',
         model_true=None,
         reward_true=None,
         model_load=None,
         policy_load=None,
         output_to_log_file=False,
         log_episode_filename='log_episode_naive',
         prt_progress=True,
         prt_policy_normalization=True,
         true_weight_output_parent_directory='real_env',
         weight_output_parent_directory='learnt\\naive'
         ):
    '''
    model_true if None, then randomly initialize one.
    '''
    # load hyper parameters
    hyper_param= load_hyper_param('config\\'+config_filename+'.yaml')    # can delete the naive
    nS,nO,nA,H,K,nF,delta,gamma,iota =hyper_param

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
        log_file = open("log\console_output.log","w")
        sys.stdout = Logger() #sys.stdout = log_file
        print(f"Start {Alg }test. Current time={current_time_str()}")
        time.sleep(3)

    print('%'*100)
    print('Test Algorithm.')
    print('%'*100)
    print('hyper parameters:{}')
    with open('config\\'+config_filename+'.yaml') as hyp_file:  # can remove naive.
        content=hyp_file.read()
    print(content)
    print('%'*100)
    print('Call function beta_vector_value_iteration...')

    with open('log\\'+log_episode_filename+'.txt',mode='w') as log_episode_file:
        # log_episode_file.write(f"\n\nTest BVVI. Current time={current_time_str()}")   #real_env_kernels reward_fix
        if Alg=='BVVI':
            (policy_learnt, model_learnt, evaluation_results)=BVVI(\
                hyper_param=hyper_param,\
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
            from RSVI2 import RSVI2
            (policy_learnt, model_learnt, evaluation_results)=RSVI2(\
                hyper_param=hyper_param,\
                    model_true=model_true,\
                        reward_true=reward_true)
        else:
            raise(NotImplementedError)
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

def test_with_medium_random_env(from_scratch=False):
    if from_scratch:
        model_true, reward_true=(None,None)
        model_load, policy_load=(None,None)
    else:
        model_true, reward_true=load_model_rewards(parent_directory='real_env\medium_real')
        model_load, policy_load=load_model_policy(parent_directory='learnt\medium')
    policy_learnt=main(config_filename= 'hyper_param_medium',
                        model_true=None,
                        reward_true=None,
                        model_load=None,
                        policy_load=None,
                        output_to_log_file=True,
                        log_episode_filename= 'log_episode_medium',
                        prt_progress=False,
                        prt_policy_normalization=False,
                        true_weight_output_parent_directory='real_env\medium_real',
                        weight_output_parent_directory='learnt\medium'
                        )
    
def manual_initialization(H:int,
                          stochastic_transition:bool,
                          identity_emission:bool,
                          peaky_reward:bool):
    from func import Normalize_T, Normalize_O

    # Initial Distribution
    mu_true=torch.tensor([1,0,0])

    # Transition
    if stochastic_transition==False:
        T_true=torch.stack([torch.tensor([[0,0,1],
                                        [1,0,0],
                                        [0,1,0]]).unsqueeze(-1).repeat(1,1,2) for _ in range(H)])
    else:
        # Stochastic Transition(much harder)
        T_true=torch.stack([torch.tensor([[0.03,0.04,0.89],
                                        [0.95,0.02,0.10],
                                        [0.02,0.94,0.01]]).to(torch.float64).unsqueeze(-1).repeat(1,1,2) for _ in range(H)])
        T_true=Normalize_T(T_true)

    # Emission
    if identity_emission==False:
        '''
        O_true=torch.stack([torch.tensor([[0.4,0.3,0.2],
                                          [0.2,0.5,0.7],
                                          [0.4,0.2,0.1]]).transpose(0,1).repeat(1,1) for _ in range(H+1)])
        '''
        O_true=torch.stack([torch.tensor([[0.83,0.05,0.02],
                                          [0.08,0.79,0.09],
                                          [0.09,0.06,0.89]]).to(torch.float64).transpose(0,1).repeat(1,1) for _ in range(H+1)])
        O_true=Normalize_O(O_true)
    else:
        O_true=torch.eye(3).unsqueeze(0).repeat(H+1,1,1)

    # Rewards
    # peacky rewards (much easier than 0/1)
    if peaky_reward==True:
        R_true=torch.tensor([[1,-10],[-10,1],[1,-10]]).unsqueeze(0).repeat(H,1,1)
    else:
        R_true=torch.tensor([[1,0],[0,1],[1,0]]).unsqueeze(0).repeat(H,1,1)

    return (mu_true,T_true,O_true,R_true)

def test_with_naive_env(Alg:str,
                        config_filename,
                        log_episode_filename,
                        stochastic_transition,
                        identity_emission,
                        peaky_reward=True):
    
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param('config\\'+config_filename+'.yaml')    # can delete the naive

    mu_true, T_true, O_true,R_true=manual_initialization(H,
                          stochastic_transition=stochastic_transition,
                          identity_emission=identity_emission,
                          peaky_reward=peaky_reward)
    
    model_true=(mu_true, T_true, O_true)
    reward_true=R_true

    policy_learnt=main(Alg=Alg,
                       config_filename= config_filename,
                        model_true=model_true,
                        reward_true=reward_true,
                        model_load=None,
                        policy_load=None,
                        output_to_log_file=True,
                        log_episode_filename= log_episode_filename,
                        prt_progress=False,
                        prt_policy_normalization=False,
                        true_weight_output_parent_directory='real_env\\naive_real_id',      #'real_env\\naive_real'
                        weight_output_parent_directory='learnt\\naive_id'                      #'learnt\\naive' 
                        )
    
    # if we are training from naive params, also run this line:
    # Note: this only works for gamma=1. otherwise change
    log_output_test_reward_pretty(H=H,K_end=1000,gamma=1.0, plot_optimal_policy=True, optimal_value=1/gamma*np.exp(gamma*H),
                                  log_episode_file_name='log_episode_naive')

    print('%'*100)
    print("short test of policy")
    short_test(policy_learnt,mu_true,T_true,O_true,R_true,only_reward=False)
    # print(policy_learnt)

def naive_train_and_plot(Alg:str,
                         K_end:int,
                         config_filename:str,
                         log_episode_file_name:str,
                         train_from_scratch,
                         stochastic_transition,
                          identity_emission,
                          peaky_reward):
    
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param('config\\'+config_filename+'.yaml')

    # train again on the naive dataset.
    if train_from_scratch:
        test_with_naive_env(Alg,
                            config_filename=config_filename,
                            log_episode_filename=log_episode_file_name,
                            stochastic_transition=stochastic_transition,
                            identity_emission=identity_emission,
                            peaky_reward=peaky_reward)

    
    optimal_value=1/gamma*np.log(np.exp(gamma*H))
    log_file_directory='log\\'+log_episode_file_name+'.txt'

    with open(log_file_directory,mode='r') as log_episode_file:
        averge_risk_measure_of_each_episode=np.loadtxt(log_file_directory)[0:K_end+1,0]
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

        plt.savefig('plots/Reward'+current_time_str()+'.jpg')

        plt.show()

def BVVI_plot(window_width_MDP:int,
            window_width_POMDP:int,
            config_filename='hyper_param_naive',
            POMDP_log_filename='log_episode_naive',
            MDP_log_filename='log_episode_naive_2',
            K_end=1000):
    # load hyper parameters
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param('config\\'+config_filename+'.yaml')

    # read POMDP file
    log_file_directory='log\\'+POMDP_log_filename+'.txt'
    with open(log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()
    # read MDP file
    log_file_directory='log\\'+MDP_log_filename+'.txt'
    with open(log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()    

    # optimal values
    # Todo:
    optimal_value_POMDP=H*0.98 #*1.002  1/gamma*(np.exp(gamma*H)) 
    # max(np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards))))
    optimal_value_MDP=H   # 1/gamma*np.exp(gamma*H)*1.002

    # Subplot 1: Episodic Return
    # POMDP: episodic return
    # plt.subplot(3,2,1)
    POMDP_episodic_smooth=smooth(POMDP_single_episode_rewards, window_len=2,window='max_pooling')
    POMDP_episodic_smooth=smooth(POMDP_episodic_smooth, window_len=3,window='hamming')
    indices=np.arange(POMDP_episodic_smooth.shape[0])
    plt.plot(indices,POMDP_episodic_smooth, c='cornflowerblue',  linestyle='dotted',
             label='Partially Observable')
    #plt.plot(np.arange(POMDP_single_episode_rewards.shape[0]),POMDP_single_episode_rewards, label='Partially Observable')

    # MDP: episodic return
    # plt.subplot(3,2,2)
    MDP_episodic_smooth=smooth(MDP_single_episode_rewards, window_len=30,window='hamming')
    indices=np.arange(MDP_episodic_smooth.shape[0])
    plt.plot(indices,MDP_episodic_smooth, c='darkorange', label='Fully Observable')

    # Optimal Policy
    plt.axhline(y =optimal_value_MDP, color = 'red', linestyle = 'dashdot',
                label='Optimal Policy') 
    # MDP and POMDP
    plt.ylim((min(min(POMDP_episodic_smooth),min(MDP_episodic_smooth))*0.95,
              (max(max(POMDP_episodic_smooth),max(MDP_episodic_smooth)))*1.10))
    plt.title(f'Episodic Return of BVVI')
    plt.xlabel(f'Episode $k$')                             # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Episodic Return')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    print(f"POMDP_max={max(POMDP_single_episode_rewards)}, smoothed max={max(POMDP_episodic_smooth)}")
    print(f"MDP_max={max(MDP_single_episode_rewards)}, smoothed max={max(MDP_episodic_smooth)}")
    print(f"Optimal Max={1/gamma*np.exp(gamma*H)}")
    plt.savefig('plots/FinalReturn'+current_time_str()+'.jpg')
    plt.show()

    # Subplot 2: Regret
    # plt.subplot(3,2,3)
    plt.subplot(1,2,1)
    # MDP: raw data
    # plt.subplot(3,2,4)
    MDP_regret=np.cumsum(optimal_value_MDP-MDP_single_episode_rewards)
    indices=np.arange(MDP_regret.shape[0])
    scatter_size=np.ones_like(indices)*1
    plt.scatter(indices, MDP_regret,linestyle='dotted', c='orange', s=scatter_size,
                label='Raw Data')    # plt.plot(indices, MDP_regret, label='Fully observable(Raw Data)')
    
    # # MDP: smoothing
    # MDP_regret_smooth=smooth(MDP_regret, window_len=30,window='hamming')
    # indices=np.arange(MDP_regret_smooth.shape[0])
    # plt.plot(indices[40:], MDP_regret_smooth[40:], linestyle='dashed', label='Fully observable(Smoothed)')
    # MDP: fitting
    def square_rt(x,a,b,d):
        return a*np.sqrt(b*x)+d
    indices=np.arange(MDP_regret.shape[0])
    fit_param, fit_curve = curve_fit(square_rt, indices, MDP_regret)
    MDP_regret_fit=square_rt(indices, *fit_param)
    plt.plot(indices, MDP_regret_fit,c='darkorange',
             label=r'Fitted with ${O}\left(\sqrt{K}\right)$') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)
    
    # plot MDP regret
    plt.ylim((min(min(MDP_regret),min(MDP_regret))*0.3,(max(max(MDP_regret),max(MDP_regret)))*1.2))
    plt.title(f'Fully Observable Environment')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    
    plt.subplot(1,2,2)
    # POMDP: raw data
    optimal_value_POMDP=H*0.986 #max((POMDP_single_episode_rewards))*0.91
    # 1/gamma*np.exp(gamma*H)*0.96
    # max(np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards))))
    POMDP_regret=np.cumsum(optimal_value_POMDP-POMDP_single_episode_rewards)
    indices=np.arange(POMDP_regret.shape[0])
    scatter_size=np.ones_like(indices)*0.02
    plt.scatter(indices, POMDP_regret,linestyle='dotted', s=scatter_size,
                label='Raw Dat)')    # plt.plot(indices, POMDP_regret, label='Partially Observable(Raw Data)')
    # # POMDP: smoothing
    # POMDP_regret_smooth=smooth(POMDP_regret, window_len=30,window='hamming')
    # indices=np.arange(POMDP_regret_smooth.shape[0])
    # plt.plot(indices[40:], POMDP_regret_smooth[40:], label='Partially Observable(Smoothed)')
    # POMDP: fitting
    def square_rt(x,a,b,d):
        return a*np.sqrt(b*x)+d
    indices=np.arange(POMDP_regret.shape[0])
    fit_param, fit_curve = curve_fit(square_rt, indices, POMDP_regret)
    POMDP_regret_fit=square_rt(indices, *fit_param)
    # Plot POMDP Regret
    plt.plot(indices, POMDP_regret_fit,c='royalblue', label='Partially Observable(Fitted)') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)
    plt.ylim((min(min(POMDP_regret),min(POMDP_regret))*0.3,(max(max(POMDP_regret),max(POMDP_regret)))*1.2))
    plt.title(f'Partially Observable Environment')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.suptitle('Regret of BVVI')
    plt.savefig('plots/FinalRegret'+current_time_str()+'.jpg')
    plt.show()
    
    # Subplot 3: PAC guantee
    # MDP PAC
    plt.subplot(1,2,1)
    MDP_PAC_raw=np.cumsum(optimal_value_MDP-MDP_single_episode_rewards)/(1+np.arange(len(MDP_single_episode_rewards)))
    indices=np.arange(MDP_PAC_raw.shape[0])
    # plt.plot(indices, MDP_PAC_smooth,label='Fully Observable')
    plt.semilogx(indices, MDP_PAC_raw,c='orange', linestyle='dotted',
                 label='Raw Data')
    # MDP fit
    def inverse_sqrt(x,a,b,c,d):
        return a*(1/np.sqrt(b*x+c))+d
    indices=np.arange(MDP_PAC_raw.shape[0])
    fit_param, fit_curve = curve_fit(inverse_sqrt, indices, MDP_PAC_raw)
    MDP_PAC_fit=inverse_sqrt(indices, *fit_param)
    plt.semilogx(indices, MDP_PAC_fit,c='darkorange', linestyle='solid', 
                 label=r'Fitted with ${O}\left(\frac{1}{\sqrt{K}}\right)$')

    #plot POMDP and MDP PAC
    plt.xlim(1,1000)
    plt.ylim((min(min(MDP_PAC_raw),min(MDP_PAC_raw))*0.4,(max(max(MDP_PAC_raw),max(MDP_PAC_raw)))*0.6))
    plt.title(f'Fully Observable Environment')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Average Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    # POMDP PAC raw
    POMDP_PAC_raw=optimal_value_POMDP-np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards)))
    indices=np.arange(POMDP_PAC_raw.shape[0])
    # plt.plot(indices, POMDP_PAC_smooth,label='Partially Observable')
    plt.semilogx(indices, POMDP_PAC_raw,linestyle='dotted',
                 label='Raw Data')
    # POMDP PAC fitting
    def inverse_sqrt(x,a,b,c,d):
        return a*(1/np.sqrt(b*x+c))+d
    indices=np.arange(POMDP_PAC_raw.shape[0])
    fit_param, fit_curve = curve_fit(inverse_sqrt, indices, POMDP_PAC_raw)
    POMDP_PAC_fit=inverse_sqrt(indices, *fit_param)
    plt.semilogx(indices, POMDP_PAC_fit,c='royalblue', linestyle='solid',
                 label=r'Fitted with $\tilde{O}\left(\frac{1}{\sqrt{K}}\right)$')
    # plot POMDP PAC
    plt.xlim(1,1000)
    plt.ylim((min(min(POMDP_PAC_raw),min(POMDP_PAC_raw))*0.4,(max(max(POMDP_PAC_raw),max(POMDP_PAC_raw)))*0.6))
    plt.title(f'Partially Observable Environment')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Average Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.suptitle('PAC Guarantee of BVVI')
    plt.savefig('plots/FinalPAC'+current_time_str()+'.jpg')
    plt.show()
    # raise ValueError(f"hellow")

if __name__ == "__main__":
    train_from_scratch=False #True
    plot_all=True
    K_end=2000  #1000 #
    if train_from_scratch:
        naive_train_and_plot(Alg='BVVI',
                             K_end=K_end,
                         config_filename='hyper_param_naive_long',
                         train_from_scratch=True,
                         log_episode_file_name='log_episode_naive_long_perturb_2000',
                         stochastic_transition=True,
                         identity_emission=False,
                         peaky_reward=False)
    
        naive_train_and_plot(Alg='BVVI',
                             K_end=K_end,
                         config_filename='hyper_param_naive_long',
                         train_from_scratch=True,
                         log_episode_file_name='log_episode_naive_long_id_2000',
                         stochastic_transition=True,
                         identity_emission=True,
                         peaky_reward=False)
    if plot_all:
        BVVI_plot(window_width_MDP=3,
                     window_width_POMDP=30,
                     config_filename='hyper_param_naive_long',
                    POMDP_log_filename='log_episode_naive_long_perturb_2000',
                    MDP_log_filename='log_episode_naive_long_id_2000',
                    K_end=K_end
                )

    