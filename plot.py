'''
    PAC_BVVI_plot(config_filename='hyper_param_naive_long',
                  POMDP_log_filename='log_episode_naive_long',
                  MDP_log_filename='log_episode_naive_long_id',
                  K_end=1000)
    
    Regret_BVVI_plot(window_width_MDP=3,
                     window_width_POMDP=30,
                     config_filename='hyper_param_naive_long',
                    POMDP_log_filename='log_episode_naive_long',
                    MDP_log_filename='log_episode_naive_long_id',
                    K_end=1000)
    '''







def PAC_BVVI_plot(config_filename='hyper_param_naive',
                  POMDP_log_filename='log_episode_naive',
                  MDP_log_filename='log_episode_naive_2',
                  K_end=1000):
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
    plt.savefig('plots/POMDP_MDP_PAC'+current_time_str()+'.jpg')
    plt.show()

def Regret_BVVI_plot(window_width_MDP:int,
                                window_width_POMDP:int,
                                config_filename='hyper_param_naive',
                                POMDP_log_filename='log_episode_naive',
                                MDP_log_filename='log_episode_naive_2',
                                K_end=1000):
    nS,nO,nA,H,K,nF,delta,gamma,iota =load_hyper_param('config\\'+config_filename+'.yaml')

    from func import moving_average
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
    optimal_value_POMDP=max(POMDP_single_episode_rewards[500:-1])
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
    from func import smooth
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
    plt.savefig('plots/POMDP_MDP_PAC'+current_time_str()+'.jpg')
    plt.show()
