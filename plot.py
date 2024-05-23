import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_hyper_param, current_time_str
from scipy.optimize import curve_fit
from utils import smooth

def BVVI_plot(num_episodes:int,
              window_width_MDP:int,
            window_width_POMDP:int,
            config_filename:str,
            POMDP_log_filename:str,
            MDP_log_filename:str,
            ):
    plt.close()
    # load hyper parameters
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))

    # read POMDP file
    log_file_directory=os.path.join('./log',current_time_str(),POMDP_log_filename+'.txt').replace('\\', '/')
    with open(log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:num_episodes+1,0]    
        log_episode_file.close()
    # read MDP file
    log_file_directory=os.path.join('./log',current_time_str(),MDP_log_filename+'.txt').replace('\\', '/')
    with open(log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:num_episodes+1,0]    
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
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'single-Return.jpg'))
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
                label='Raw Data')    # plt.plot(indices, POMDP_regret, label='Partially Observable(Raw Data)')
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
    plt.plot(indices, POMDP_regret_fit,c='royalblue',
             label=r'Fitted with ${O}\left(\sqrt{K}\right)$') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)
    plt.ylim((min(min(POMDP_regret),min(POMDP_regret))*0.3,(max(max(POMDP_regret),max(POMDP_regret)))*1.2))
    plt.title(f'Partially Observable Environment')
   
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.suptitle('Regret of BVVI')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'single-Regret.jpg'))
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
    plt.xlim(1,num_episodes)
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
                 label=r'Fitted with ${O}\left(\frac{1}{\sqrt{K}}\right)$')
    # plot POMDP PAC
    plt.xlim(1,num_episodes)
    plt.ylim((min(min(POMDP_PAC_raw),min(POMDP_PAC_raw))*0.4,(max(max(POMDP_PAC_raw),max(POMDP_PAC_raw)))*0.6))
    plt.title(f'Partially Observable Environment')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Average Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.suptitle('PAC Guarantee of BVVI')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'single-PAC.jpg'))
    plt.show()
    # raise ValueError(f"hellow")
    plt.close()



def multi_risk_level_plot(config_files:list,
                          POMDP_log_files:list,
                          num_episodes:int):
    plt.close()
    # read risk_params
    num_params=len(config_files)
    print(f"num_params={num_params}")
    risk_params=np.zeros([num_params])
    for i in range(num_params):
        with open(os.path.join('config',config_files[0]+'.yaml'), 'r') as file:
            nS,nO,nA,H,K,nF,delta,risk_params[i],iota =load_hyper_param(os.path.join('config',config_files[i]+'.yaml'))
            K=num_episodes
        file.close()
    print(f"Risk Params={risk_params}")

    # read episodic rewards
    episodic_rewards=np.zeros([K, num_params])
    print(f"shape of container=={episodic_rewards.shape}")
    for i in range(num_params):
        log_file_directory=os.path.join('./log',current_time_str(),POMDP_log_files[i]+'.txt')
        print(f"will read file from {log_file_directory}")
        with open(log_file_directory, 'r') as file:
            a=np.loadtxt(log_file_directory)[:,0]
            print(f"{a.shape}")
            episodic_rewards[:,i]=np.loadtxt(log_file_directory)[:,0]
            print(f"{episodic_rewards.shape}")
            file.close()
    
    # save various output of risk level into a single file
    np.savetxt(os.path.join('./log',current_time_str(),'various_risk','all.txt'), episodic_rewards)

    # signal processing of the episodic rewards.

    # smooth the raw data.
    from utils import POMDP_smooth, POMDP_PAC, POMDP_regret
    
    optimal_values=[None,H*0.96, H*0.97,  H*0.975, H*0.98, H, H]
    '''
    i=0
    i=1:     H*0.96
    i=2:     H*0.97
    i=3:     H*0.975
    i=4:     H*0.98
    i=5:     H
    i=6:     H
    '''
    '''
    # plot Regrets
    indices, regret,regret_fit, scatter_size=POMDP_regret(optimal_value_POMDP,smoothed)
    plt.scatter(indices, regret,linestyle='dotted', s=scatter_size,
        label='Raw Data')
    plt.plot(indices,regret_fit)
    plt.show()
    '''
    # plot PACs
    label_text=['' for _ in range(num_params*2)]
    label_color=['red', 'darkorange', 'gold', 'skyblue', 'green', 'blue', 'purple']
    for i in range(1,num_params,1):
        '''
        print(f"Max episodic reward of {i}=={max(episodic_rewards[:,i])}")
        print(f"{max(np.cumsum(episodic_rewards[:,i])/(1+np.arange(len(episodic_rewards[:,i]))))}")
        '''
        label_text[i*2]=r'$\gamma$='+f'{risk_params[i]}'

        optimal_value_POMDP=optimal_values[i]

        # smoothed curves
        raw=episodic_rewards[:,i]
        smoothed_id, smoothed=POMDP_smooth(raw)
        # plt.plot(indicies,smoothed)
        # plt.show()
        # PAC
        indices, PAC, PAC_fit=POMDP_PAC(optimal_value_POMDP,smoothed)
        ax = plt.gca()
        smooth_PAC_id, smooth_PAC=POMDP_smooth(PAC)
        plt.scatter(smooth_PAC_id, smooth_PAC,c=label_color[i],alpha=0.3, s=np.ones_like(smooth_PAC_id))
        plt.xscale('log')
        plt.semilogx(indices, PAC_fit,linestyle='solid',c=label_color[i], label=label_text[i*2])
    plt.xlim(10,num_episodes)
    plt.title(f'PAC Guarantee of BVVI\nUnder Different Risk Sensitivity '+r' $\gamma$')
    plt.xlabel('Episode Number k')
    plt.ylabel('Average Regret')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'various-PAC.jpg'))
    plt.show()

def plot_pac(config_filename:str,
             POMDP_log_filename:str,
             MDP_log_filename:str,
             K_end:int):
    plt.close()
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))

    # read POMDP file
    pomdp_log_file_directory=os.path.join('log',current_time_str(),POMDP_log_filename+'.txt')
    with open(pomdp_log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(pomdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()

    # read MDP file
    mdp_log_file_directory=os.path.join('log',current_time_str(),MDP_log_filename+'.txt')
    with open(mdp_log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(mdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()    

    # optimal values
    # Todo:
    optimal_value_POMDP=max(np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards))))
    optimal_value_MDP=1/gamma*np.exp(gamma*H)
    
    # plot POMDP curve.
    POMDP_regret_smooth=optimal_value_POMDP-np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards)))
    indices=np.arange(POMDP_regret_smooth.shape[0])
    plt.plot(indices, POMDP_regret_smooth,label='Partially Observable')

    # plot MDP curve.
    MDP_regret=optimal_value_MDP-MDP_single_episode_rewards
    MDP_regret_smooth=np.cumsum(MDP_regret)/(1+np.arange(len(MDP_regret)))
    indices=np.arange(MDP_regret_smooth.shape[0])
    plt.plot(indices, MDP_regret_smooth,label='Fully observable')

    #plot POMDP and MDP together.
    plt.ylim((min(min(POMDP_regret_smooth),min(MDP_regret_smooth))*0.4,(max(max(POMDP_regret_smooth),max(MDP_regret_smooth)))*1.2))
    plt.title(f'BVVI in Different Environments ')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Average Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'single-PAC.jpg'))
    plt.show()

def plot_regret(window_width_MDP:int,
                                window_width_POMDP:int,
                                config_filename:str,
                                POMDP_log_filename:str,
                                MDP_log_filename:str,
                                K_end:int):
    plt.close()
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))

    from utils import moving_average
    # read POMDP file
    pomdp_log_file_directory=os.path.join('log',current_time_str(),POMDP_log_filename+'.txt')
    with open(pomdp_log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(pomdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()

    # read MDP file
    mdp_log_file_directory=os.path.join('log',current_time_str(),MDP_log_filename+'.txt')
    with open(mdp_log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(mdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()    

    # optimal values
    # Todo:
    optimal_value_POMDP=max(POMDP_single_episode_rewards[int(K_end//2):-1])
    # 1/gamma*np.exp(gamma*H)
    optimal_value_MDP=1/gamma*np.exp(gamma*H)
    
    '''
    # plot POMDP curve.
    POMDP_regret_smooth=POMDP_single_episode_rewards   
    # optimal_value_POMDP-POMDP_single_episode_rewards # 
    # POMDP_regret_smooth=np.cumsum(POMDP_regret_smooth)                                        #/(1+np.arange(len(POMDP_single_episode_rewards)))
    indices=np.arange(POMDP_regret_smooth.shape[0])
    plt.plot(indices, POMDP_regret_smooth,label='Partially Observable')    # *(1+np.arange(len(POMDP_single_episode_rewards)))
    plt.show()
    '''
    # plot MDP curve.
    from scipy.optimize import curve_fit
    from utils import smooth
    MDP_regret=np.cumsum(optimal_value_MDP-MDP_single_episode_rewards)
    indices=np.arange(MDP_regret.shape[0])
    scatter_size=np.ones_like(indices)*0.02
    
    # plot raw data in scatter or line segment
    plt.scatter(indices, MDP_regret,linestyle='dotted', s=scatter_size, label='Fully observable(Raw Data)')
    # plt.plot(indices, MDP_regret, label='Fully observable(Raw Data)')

    # smoothing
    smooth_curve=smooth(MDP_regret, window_len=30,window='hamming')
    smooth_indices=np.arange(smooth_curve.shape[0])
    plt.plot(smooth_indices[40:], smooth_curve[40:], label='Fully observable(Smoothed)')

    # fit MDP curve with sqrt function.
    def square_rt(x,a,b,d):
        return a*np.sqrt(b*x)+d
    fit_param, fit_curve = curve_fit(square_rt, indices, MDP_regret)
    MDP_regret_smooth=square_rt(indices, *fit_param)
    plt.plot(indices, MDP_regret_smooth,label='Fully observable(Fitted)') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)

    #plot POMDP and MDP together.
    plt.ylim((min(min(MDP_regret_smooth),min(MDP_regret_smooth))*0.3,(max(max(MDP_regret_smooth),max(MDP_regret_smooth)))*1.2))
    plt.title(f'BVVI in Different Environments ')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'single-PAC.jpg'))
    plt.show()

def plot_cum_reward(config_filename:str,
             POMDP_log_filename:str,
             MDP_log_filename:str,
             K_end:int):
    plt.close()
    # read POMDP file
    pomdp_log_file_directory=os.path.join('log',current_time_str(),POMDP_log_filename+'.txt')
    with open(pomdp_log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(pomdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()

    # read MDP file
    mdp_log_file_directory=os.path.join('log',current_time_str(),MDP_log_filename+'.txt')
    with open(mdp_log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(mdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()

    MDP_cumulative_rewards=np.cumsum(MDP_single_episode_rewards)
    POMDP_cumulative_rewards=np.cumsum(POMDP_single_episode_rewards)
    indices=np.arange(MDP_cumulative_rewards.shape[0])

    #plot POMDP and MDP together.
    plt.plot(indices, POMDP_cumulative_rewards, label="POMDP")
    plt.plot(indices, MDP_cumulative_rewards, label="MDP")
    plt.ylim((min(min(MDP_cumulative_rewards),min(POMDP_cumulative_rewards))*0.3,
              (max(max(MDP_cumulative_rewards),max(POMDP_cumulative_rewards)))*1.2))
    plt.title(f'BVVI in Different Environments ')
    plt.xlabel(f'Episode $k$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Cumulative Rewards')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'single-cum-rewards.jpg'))
    plt.show()

def plot_cum_reward_various_risk(gamma_range:list,num_episodes:int):
    plt.close()
    # determine the size of the plot.
    num_plots=len(gamma_range)
    rows = int(np.sqrt(num_plots))  # Calculate the number of rows
    cols = (num_plots + rows - 1) // rows  # Calculate the number of columns

    # read in episodic rewards of all the risk levels.
    episodic_rewards=np.zeros([num_episodes,num_plots])
    episodic_rewards=np.loadtxt(os.path.join('./log',current_time_str(),'various_risk','all.txt'))

    # calculate the cumulative rewards.
    cumulative_rewards=np.cumsum(episodic_rewards,axis=0)  # of shape [K,num_plots]
    indices=np.arange(episodic_rewards.shape[0])

    # Create a grid of subplots based on the number of elements in the array
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    # Plot data on each subplot
    for i, ax in enumerate(axs.flat):
        gamma=gamma_range[i]
        if i < num_plots:            
            ax.plot(indices,cumulative_rewards[i,:])
            ax.xlabel(f'Episode $k$')           
            ax.ylabel(f'Cumulative Rewards')    
            ax.set_title(f'$\gamma$={gamma}')
            ax.legend(loc='upper right')
    
    # add a super title to the entire plot.
    plt.suptitle(f'BVVI under Various Risk Levels')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'various-cum-sep.jpg'))
    plt.show()
    plt.close()

    plt.figure()
    # plot all the curves in a single graph:
    for i in range(num_plots):
        plt.plot(indices,cumulative_rewards[i,:],label=r"$\gamma=$"+f"{gamma}")
    plt.xlabel(f'Episode $k$')
    plt.ylabel(f'Cumulative Rewards')
    plt.suptitle(f'BVVI under Various Risk Levels')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',current_time_str(),'various-cum-all.jpg'))
    plt.show()
