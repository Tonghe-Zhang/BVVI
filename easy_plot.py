from func import log_output_tested_rewards, current_time_str
import matplotlib.pyplot as plt
import numpy as np
H=3

with open('log\log_episode_naive.txt',mode='r+') as log_episode_file:
    averge_risk_measure_of_each_episode=np.loadtxt('log\log_episode_naive.txt')[0:30,0]

    loss_curve=averge_risk_measure_of_each_episode
    indices=np.arange(loss_curve.shape[0]) 

    plt.plot(indices, loss_curve,label='BVVI(ours)') 
    plt.axhline(y=np.exp(3), color='orange', linestyle='--',label='Optimal Policy')
    plt.title(f'Accumulated Risk-Sensitive Reward of Policies')   # . Horizon H={H}
    
    plt.xlabel(f'Episode $k$')    # H transitions per iteration.   Samples N (=iteration $K$ * {H})
    plt.ylabel(f'Average Risk Measure')         # $\sum_{h=1}^{H}r_h(\mathbf{S}_h,\mathbf{A}_h)$
    plt.ylim((0.0,31.5))
    plt.legend(loc='upper right')
    plt.savefig('plots/Reward'+current_time_str()+'.jpg')
    plt.show()



