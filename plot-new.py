import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_hyper_param
from scipy.optimize import curve_fit
from utils import smooth
crts="2024-05-22-16-57-58"


def plot_pac(config_filename:str,
             POMDP_log_filename:str,
             MDP_log_filename:str,
             K_end:int):
    plt.close()
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))

    # read POMDP file
    pomdp_log_file_directory=os.path.join('log',crts,POMDP_log_filename+'.txt')
    with open(pomdp_log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(pomdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()

    # read MDP file
    mdp_log_file_directory=os.path.join('log',crts,MDP_log_filename+'.txt')
    with open(mdp_log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(mdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()    

    # optimal values
    # Todo:
    optimal_value_POMDP=max(np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards))))
    optimal_value_MDP=1/gamma*np.log(np.exp(gamma*H))
    
    # plot POMDP curve.
    """ 
    J(\pi^\star)-\frac{1}{K}\sum_{k=1}^K J(\pi^k)
    """
    POMDP_pac_smooth=optimal_value_POMDP-\
    np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards)))
    indices=np.arange(POMDP_pac_smooth.shape[0])
    plt.plot(indices, POMDP_pac_smooth,label='Partially Observable')

    # plot MDP curve.
    MDP_regret=optimal_value_MDP-MDP_single_episode_rewards
    MDP_pac_smooth=np.cumsum(MDP_regret)/(1+np.arange(len(MDP_regret)))
    indices=np.arange(MDP_pac_smooth.shape[0])
    plt.plot(indices, MDP_pac_smooth,label='Fully observable')

    #plot POMDP and MDP together.
    plt.ylim((min(min(POMDP_pac_smooth),min(MDP_pac_smooth))*0.4,(max(max(POMDP_pac_smooth),max(MDP_pac_smooth)))*1.2))
    plt.xlim(-2,750)
    plt.title(f'BVVI in Different Environments ')
    plt.xlabel(f'Number of Episodes $K$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Suboptimality')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,'single-PAC.jpg'))
    plt.show()

# def plot_regret(
#         config_filename:str,
#         POMDP_log_filename:str,
#         MDP_log_filename:str,
#         K_end:int):
#     plt.close()
#     nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))

#     from utils import moving_average
#     # read POMDP file
#     pomdp_log_file_directory=os.path.join('log',crts,POMDP_log_filename+'.txt')
#     with open(pomdp_log_file_directory,mode='r') as log_episode_file:
#         POMDP_single_episode_rewards=np.loadtxt(pomdp_log_file_directory)[0:K_end+1,0]    
#         log_episode_file.close()

#     # read MDP file
#     mdp_log_file_directory=os.path.join('log',crts,MDP_log_filename+'.txt')
#     with open(mdp_log_file_directory,mode='r') as log_episode_file:
#         MDP_single_episode_rewards=np.loadtxt(mdp_log_file_directory)[0:K_end+1,0]    
#         log_episode_file.close()    

#     # optimal values
#     # Todo:
#     optimal_value_POMDP=max(POMDP_single_episode_rewards[int(K_end//2):-1])
#     # 1/gamma*np.exp(gamma*H)
#     optimal_value_MDP=H
    
#     plt.subplot(1,2,1)
#     # plot MDP curve.
#     MDP_regret=np.cumsum(optimal_value_MDP-MDP_single_episode_rewards)
#     indices=np.arange(MDP_regret.shape[0])
#     scatter_size=np.ones_like(indices)*0.02
#     # plot raw data in scatter or line segment
#     plt.scatter(indices, MDP_regret,linestyle='dotted', s=scatter_size, label='Fully observable(Raw Data)')
#     # plt.plot(indices, MDP_regret, label='Fully observable(Raw Data)')
#     # smoothing
#     smooth_curve=smooth(MDP_regret, window_len=30,window='hamming')
#     smooth_indices=np.arange(smooth_curve.shape[0])
#     # fit MDP curve with sqrt function.
#     def square_rt(x,a,b,d):
#         return a*np.sqrt(b*x)+d
#     fit_param, fit_curve = curve_fit(square_rt, indices, MDP_regret)
#     MDP_regret_smooth=square_rt(indices, *fit_param)
#     #plt.plot(smooth_indices[40:], smooth_curve[40:], label='Fully Observable(Smoothed)')
#     plt.plot(indices, MDP_regret_smooth,label='Fully observable(Fitted)',c='skyblue') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)
#     plt.legend(loc='upper left')
#     plt.ylim([0,max(MDP_regret)*1.2])
#     plt.show()
#     exit()
#     # plot POMDP curve.
#     plt.subplot(1,2,2)
#     POMDP_regret=np.cumsum(optimal_value_POMDP-POMDP_single_episode_rewards)
#     indices=np.arange(POMDP_regret.shape[0])
#     scatter_size=np.ones_like(indices)*0.02
#     # plot raw data in scatter or line segment
#     plt.scatter(indices, POMDP_regret,linestyle='dotted', s=scatter_size, label='Fully observable(Raw Data)')
#     # plt.plot(indices, POMDP_regret, label='Fully observable(Raw Data)')
#     # smoothing
#     smooth_curve=smooth(POMDP_regret, window_len=30,window='hamming')
#     smooth_indices=np.arange(smooth_curve.shape[0])
#     # fit POMDP curve with sqrt function.
#     def square_rt(x,a,b,d):
#         return a*np.sqrt(b*x)+d
#     fit_param, fit_curve = curve_fit(square_rt, indices, POMDP_regret)
#     POMDP_regret_smooth=square_rt(indices, *fit_param)
    
#     plt.plot(smooth_indices[40:], smooth_curve[40:], label='Partially Observable(Smoothed)')
#     plt.plot(indices, POMDP_regret_smooth,label='Fully observable(Fitted)') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)

#     #plot POMDP and MDP together.
#     plt.ylim((min(min(MDP_regret_smooth),min(POMDP_regret_smooth))*0.3,(max(max(MDP_regret_smooth),max(POMDP_regret_smooth)))*1.2))
#     plt.title(f'BVVI in Different Environments ')
#     plt.xlabel(f'Number of Episodes $K$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
#     plt.ylabel(f'Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.savefig(os.path.join('plots',crts,'single-PAC.jpg'))
#     plt.show()

def plot_cum_reward(POMDP_log_filename:str,
                    MDP_log_filename:str,
                    K_end:int):
    plt.close()
    # read POMDP file
    pomdp_log_file_directory=os.path.join('log',crts,POMDP_log_filename+'.txt')
    with open(pomdp_log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(pomdp_log_file_directory)[0:K_end+1,0]    
        log_episode_file.close()

    # read MDP file
    mdp_log_file_directory=os.path.join('log',crts,MDP_log_filename+'.txt')
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
    plt.xlabel(f'Number of Episodes $K$')               # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Cumulative Rewards')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,'single-cum-rewards.jpg'))
    plt.show()
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
        log_file_directory=os.path.join('./log',crts,POMDP_log_files[i]+'.txt')
        print(f"will read file from {log_file_directory}")
        with open(log_file_directory, 'r') as file:
            a=np.loadtxt(log_file_directory)[:,0]
            print(f"{a.shape}")
            episodic_rewards[:,i]=np.loadtxt(log_file_directory)[:,0]
            print(f"{episodic_rewards.shape}")
            file.close()
    
    # save various output of risk level into a single file
    np.savetxt(os.path.join('./log',crts,'various_risk','all.txt'), episodic_rewards)

def plot_various_risk(gamma_range:list,
                      num_episodes:int,
                      curve_type:str):
    print(f"Will print the experimental results of gamma in range {gamma_range}")
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))
    plt.close()
    # determine the size of the plot.
    num_plots=len(gamma_range)
    rows = int(np.sqrt(num_plots))  # Calculate the number of rows
    cols = (num_plots + rows - 1) // rows # Calculate the number of columns

    # read in episodic rewards of all the risk levels.
    episodic_rewards=np.zeros([num_episodes,num_plots])
    episodic_rewards=np.loadtxt(os.path.join('./log',crts,'various_risk','all.txt'))

    # calculate the metrics.
    num_episodes=len(episodic_rewards[:,0])
    indices=np.arange(episodic_rewards.shape[0])
    type_of_the_curve=curve_type  # 'episodic_rewards'   'pac'   'regret'
    yax_label=''
    suptitle_label=''
    if type_of_the_curve == 'cum_rewards':
        yax_label='Cumulative Rewards'
        suptitle_label=r"Cumulative Reward of BVVI under Various Risk Levels"
        curves=np.cumsum(episodic_rewards,axis=0)  # of shape [K,num_plots]
    elif type_of_the_curve=='episodic_rewards':
        yax_label='Episodic Rewards'
        suptitle_label=r"Episodic Returns of BVVI under Various Risk Levels"
        curves=episodic_rewards
    elif type_of_the_curve=='regret':
        yax_label='Cumulative Regret'
        suptitle_label=r"Regret of BVVI under Various Risk Levels"+"\n"+\
        r"$\operatorname{Regret}=\tilde{O}\left(\frac{e^{|\gamma| H}-1}{|\gamma| H} H^2 \sqrt{K H S^2 O A}\right)$"
        curves=np.zeros_like(episodic_rewards)
        curves_fit=curves
        regret_optim=[3.22,
                      3.97,
                      3.45,
                      3.895,
                      3.95,
                      4.02]
        # 3.2195387315999215
        # 3.9999945401999017
        # 3.9946185155693534
        # 4.000009607893934
        # 4.000000001831564
        # 4.000000000000204
        #max((episodic_rewards))  #3.1 #H*0.8 #
        indices=np.arange(num_episodes)
        for i in range(num_plots):
            # calculate regret from episodic returns
            print(max(episodic_rewards[:,i]))
            optimal_value_POMDP=regret_optim[i]#max(episodic_rewards[:,i]) #regret_optim[i] #regret_optim[i]   #max(episodic_rewards[:,i]) #max(episodic_rewards[int(num_episodes//2):-1,i]) #
            POMDP_regret=np.cumsum(optimal_value_POMDP-episodic_rewards[:,i])
            curves[:,i]=POMDP_regret
            # fit with \sqrt{K}
            fit_param, fit_curve = curve_fit(square_rt_pos, indices, POMDP_regret)
            POMDP_regret_fit=square_rt(indices, *fit_param)
            curves_fit[:,i]=POMDP_regret_fit
    
    elif type_of_the_curve=='pac':
        yax_label='Suboptimality'
        suptitle_label=r"PAC Guarantee of BVVI under Various Risk Levels"+"\n"\
            +r"$J(\pi^\star;\mathcal{P},\gamma)-\frac{1}{K}\sum_{k=1}^K J(\pi^k;\mathcal{P},\gamma)=\tilde{O}\left(\frac{1}{\sqrt{K}}\frac{e^{|\gamma| H}-1}{|\gamma| H} H^2 \sqrt{H S^2 O A}\right)$"
        curves=np.zeros_like(episodic_rewards)
        for i in range(num_plots):
            optimal_value=max(np.cumsum(episodic_rewards[:,i])/(1+np.arange(num_episodes)))
            print(f"gamma={gamma_range[i]}, optimal_value={optimal_value}")
            curves[:,i]=optimal_value-np.cumsum(episodic_rewards[:,i])/(1+np.arange(num_episodes))
    else:
        print(f'curve_type must be in cum_rewards, episodic_rewards, regret or pac, but got %s' % curve_type)
        raise NotImplementedError
    ################################################################################################################################
    # Create a grid of subplots based on the number of elements in the array
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    # Plot data on each subplot
    for i, ax in enumerate(axs.flat):
        print(f"i={i}")
        gamma=gamma_range[i]
        if i < num_plots:
            if curve_type=='regret':
                # plot original data
                scatter_size=np.ones_like(indices)*3
                ax.scatter(indices, curves[:,i],linestyle='dashdot', s=scatter_size,alpha=1)
                # plot fitted curve.
                ax.plot(indices, curves_fit[:,i] ,c='blue',linestyle='dashed',
                    label=r"$\gamma=$"+f'{gamma}') #+r'Fitted with $O\left(\sqrt{K}\right)$'
            else:
                ax.plot(indices,curves[:,i],label=r'$\gamma=$'+f"{gamma}")
            ax.set_xlabel(f'Number of Episodes $K$')           
            ax.set_ylabel(yax_label)    
            ax.set_title(f'$\gamma$={gamma}')
            ax.set_xlim(-10,2000)
            # ax.legend(loc='upper right')
    # add a super title to the entire plot.
    plt.suptitle(suptitle_label)
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,yax_label+'various-cum-sep.jpg'))
    plt.show()
    plt.close()
    plt.figure()
    ################################################################################################################################
    color=['black','purple','darkblue','green','orange','darkorange','red']
    # plot all the curves in a single graph:
    for i in range(num_plots):
        plt.plot(indices,curves[:,i],c=color[i])
        if curve_type=='regret':
            plt.plot(indices,curves_fit[:,i],label=r"$\gamma=$"+f"{gamma_range[i]}",c=color[i],alpha=0.7)
    plt.xlabel(f'Number of Episodes $K$')
    plt.ylabel(yax_label)
    plt.xlim(-20,2000)
    plt.suptitle(suptitle_label)
    if curve_type=='regret':
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,yax_label+'various-cum-all.jpg'))
    plt.show()


def BVVI_plot(num_episodes:int,
            config_filename:str,
            POMDP_log_filename:str,
            MDP_log_filename:str,
            ):
    plt.close()
    # load hyper parameters
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param(os.path.join('config',config_filename+'.yaml'))
    # read POMDP file
    log_file_directory=os.path.join('./log',crts,POMDP_log_filename+'.txt').replace('\\', '/')
    with open(log_file_directory,mode='r') as log_episode_file:
        POMDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:num_episodes+1,0]    
        log_episode_file.close()
    # read MDP file
    log_file_directory=os.path.join('./log',crts,MDP_log_filename+'.txt').replace('\\', '/')
    with open(log_file_directory,mode='r') as log_episode_file:
        MDP_single_episode_rewards=np.loadtxt(log_file_directory)[0:num_episodes+1,0]    
        log_episode_file.close()    

    # optimal values
    # Todo:
    optimal_value_POMDP=H #*1.002  1/gamma*(np.exp(gamma*H)) 
    # max(np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards))))
    optimal_value_MDP=H   # 1/gamma*np.exp(gamma*H)*1.002

    ###############################################################################################################################
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
    plt.plot(indices,MDP_episodic_smooth, c='black', label='Fully Observable',linestyle = 'dashdot')

    # Optimal Policy
    plt.axhline(y=optimal_value_MDP*1.001, color = 'red', label='Optimal Policy') 
    # MDP and POMDP
    plt.ylim((min(min(POMDP_episodic_smooth),min(MDP_episodic_smooth))*0.95,
              (max(max(POMDP_episodic_smooth),max(MDP_episodic_smooth)))*1.10))
    plt.title(f'Episodic Return of BVVI')
    plt.xlabel(f'Number of Episodes $K$') # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Episodic Return')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    print(f"POMDP_max={max(POMDP_single_episode_rewards)}, smoothed max={max(POMDP_episodic_smooth)}")
    print(f"MDP_max={max(MDP_single_episode_rewards)}, smoothed max={max(MDP_episodic_smooth)}")
    print(f"Optimal Max={H}")
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,'single-Return.jpg'))
    plt.show()
    
    ###############################################################################################################################
    # Subplot 2: Regret
    # plt.subplot(3,2,3)
    #plt.subplot(1,2,1)
    # MDP: raw data
    # plt.subplot(3,2,4)
    optimal_value_MDP=H*1.005
    MDP_regret=np.cumsum(optimal_value_MDP-MDP_single_episode_rewards)
    indices=np.arange(MDP_regret.shape[0])
    scatter_size=np.ones_like(indices)*3
    plt.scatter(indices, MDP_regret,linestyle='dashdot', c='orange', s=scatter_size,alpha=1)
    # plt.plot(indices, MDP_regret, label='Fully observable(Raw Data)')
    # ,label=f'Fully Observable'
    
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
    plt.plot(indices, MDP_regret_fit,c='darkorange',linestyle='dashed',
             label=r'Fully Observable, Fitted with $O\left(\sqrt{K}\right)$') #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)
    # plot MDP regret
    plt.ylim((min(min(MDP_regret),min(MDP_regret))*0.3,(max(max(MDP_regret),max(MDP_regret)))*1.2))
    plt.xlabel(f'Number of Episodes $K$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Cumulative Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    
    #plt.subplot(1,2,2)
    # POMDP: raw data
    optimal_value_POMDP=H*0.985 #max((POMDP_single_episode_rewards))*0.91
    # 1/gamma*np.exp(gamma*H)*0.96
    # max(np.cumsum(POMDP_single_episode_rewards)/(1+np.arange(len(POMDP_single_episode_rewards))))
    POMDP_regret=np.cumsum(optimal_value_POMDP-POMDP_single_episode_rewards)
    indices=np.arange(POMDP_regret.shape[0])
    scatter_size=np.ones_like(indices)*3
    plt.scatter(indices, POMDP_regret,linestyle='dashdot', s=scatter_size,alpha=1)  
    # plt.plot(indices, POMDP_regret, label='Partially Observable(Raw Data)')
    # ,label='Partially Observable'
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
    plt.plot(indices, POMDP_regret_fit,c='royalblue',linestyle='dashed',
             label=r'Partially Observable, Fitted with $O\left(\sqrt{K}\right)$') #  #: a=%5.3f, b=%5.3f, d=%5.3f' % tuple(fit_param)
    #(min(min(POMDP_regret),min(POMDP_regret))*0.3,(max(max(POMDP_regret),max(POMDP_regret)))*1.7)
    plt.ylim(0,(max(max(POMDP_regret),max(POMDP_regret)))*1.7)
    #plt.title(f'Partially Observable Environment')
   
    plt.xlabel(f'Number of Episodes $K$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Cumulative Regret')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.suptitle('Regret of BVVI')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,'single-Regret.jpg'))
    plt.show()

    ###############################################################################################################################
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
    plt.ylim(0,(max(max(MDP_PAC_raw),max(MDP_PAC_raw)))*1.05)
    plt.title(f'Fully Observable Environment')
    plt.xlabel(f'Number of Episodes $K$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Suboptimality')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
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
    plt.ylim(0,(max(max(POMDP_PAC_raw),max(POMDP_PAC_raw)))*1.05)
    plt.title(f'Partially Observable Environment')
    plt.xlabel(f'Number of Episodes $K$')           # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Suboptimality')        # plt.ylabel( r'$\frac{1}{k}\sum_{t=1}^{k} \frac{1}{\gamma} \mathbb{E}^{\pi^k} \sum_{h=1}^H e^{\gamma r_h(S_h,A_h)}$')
    plt.legend(loc='upper right')
    plt.suptitle('PAC Guarantee of BVVI')
    plt.tight_layout()
    plt.savefig(os.path.join('plots',crts,'single-PAC.jpg'))
    plt.show()
    # raise ValueError(f"hellow")
    plt.close()


def square_rt(x,a,b,d):
    return a*np.sqrt(b*x)+d

def square_rt_pos(x,a,b,d):
    return a*np.sqrt(b*x)+d**2

if __name__ == '__main__':
    gamma_range=[ -5.0,-3.0, -1.0, 1.0, 3.0, 5.0]   #0.01, 
    config_filename='naive' #'naive-medium'
    log_filename_pomdp='pomdp'
    log_filename_mdp='mdp'
    config_files=[ _ for _ in range(len(gamma_range))]
    log_files=config_files 
    for i,gamma in enumerate(gamma_range):
        config_files[i]=os.path.join("various_risk",f"gamma={gamma}")
        log_files[i]=os.path.join("various_risk",f"gamma={gamma}")
    num_episodes=2000

    # multi_risk_level_plot(config_files=config_files,
    #                       POMDP_log_files=log_files,
    #                       num_episodes=num_episodes)

    # plot_pac(config_filename=config_filename,
    #               POMDP_log_filename=log_filename_pomdp,
    #               MDP_log_filename=log_filename_mdp,
    #               K_end=num_episodes)
 
    
    BVVI_plot(num_episodes=num_episodes,
            config_filename=config_filename,
            POMDP_log_filename=log_filename_pomdp,
            MDP_log_filename=log_filename_mdp)
    
    # plot_various_risk(gamma_range=gamma_range,
    #                   num_episodes=num_episodes,
    #                   curve_type='pac')   #
 
    