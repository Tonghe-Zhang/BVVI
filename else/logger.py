import numpy as np
import ast

with open('log_episode.txt',mode='w') as log_episode_file:
    for _ in range(100):
        # [, , ,  ]
        # array_str = np.array2string(np.array([1,2,3,4]).reshape(1,4), separator=',')[1:-1]
        write_str=str(tested_returns[k])+'\t'+str(mu_err[k])+'\t'+str(T_err[k])+'\t'+str(O_err[k])+'\t'
        log_episode_file.write(write_str+ "\n")
    log_episode_file.close()
with open('log_episode.txt', mode='r') as log_episode_file:
    episode_data=np.loadtxt('log_episode.txt', dtype=np.float64)
    print(episode_data)
    print(episode_data.shape)
    log_episode_file.close()
