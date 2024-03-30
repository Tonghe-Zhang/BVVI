import numpy as np
'''
We will only compare RSVI with BVVI in the fully-observable setting.
Degenerate the BVVI to fully-observable setting.
It should be the same with BVVI when emission()==id.
But much worse when it is not.

Reference: https://arxiv.org/pdf/2111.03947.pdf
page 7

'''

import torch
from tqdm import tqdm

def RSVI2(hyper_param,
          model_true:tuple,
          reward_true:torch.Tensor
          )->tuple:
   '''
   RSVI2(nS,nA,H,K,gamma,delta,model_true,reward_true)
   Return: (policy_learnt, None, evaluation_results)
   '''
   import numpy as np
   from POMDP_model import sample_trajectory
   nS,_,nA,H,K,_,delta,gamma,_ =hyper_param
   tested_risk_measure=np.zeros([K])

   w=[np.zeros([nS,nA]) for _ in range(H)]
   b=[np.zeros([nS,nA]) for _ in range(H)]
   G=[np.zeros([nS,nA]) for _ in range(H)]
   V=[np.ones([nS])*(H-h) for h in range(H)]
   Q=[np.ones([nS,nA])*(H-h) for h in range(H)]
   N=[np.zeros([nS,nA]) for h in range(H)]
   policy_learnt=[np.zeros([nS,nA]) for h in range(H)]
   
   traj_s=np.zeros([H+1,K])  # \{ (s_h^k, a_h^k) \}_{h,k}
   traj_a=np.zeros([H+1,K])
   for k in tqdm(range(K)):
      for h in range(H):
        for s in range(nS):
            for a in range(nA):
               w[h][s][a]=1/np.max(N[h][s][a], 1)*(
                  sum([ (traj_s[h][tau]==s and  traj_a[h][tau]==a )*\
                       np.exp(gamma*(reward_true[h][s][a]+V[h+1][traj_s[h+1][tau]]))\
                        for tau in range(k-1)])
               )
               b[h][s][a]=3*np.abs(np.exp(gamma*(H-h))-1)*\
                  np.sqrt(nS*np.log(H*nS*nA*K/delta)/max(N[h][s][a],1))
               if gamma>0:
                  G[h][s][a]=min(w[h][s][a]+b[h][s][a], np.exp(gamma*(H-h)))
               else:
                  G[h][s][a]=max(w[h][s][a]-b[h][s][a], np.exp(gamma*(H-h)))
               V[h][s]=np.max(1/gamma*np.log(G[h][s,:]))
               policy_learnt[h][s]=np.argmax(1/gamma*np.log(G[h][s,:]))
      
      traj_full=sample_trajectory(horizon=H,policy=policy_learnt,model=model_true,reward=reward_true)
      traj_s[:,k]=traj_full[1,:]    #when deploying RSIV2 to POMDP, we should mistake observations as states. This will only work when emission()==id(). 
      traj_a[:,k]=traj_full[2,:]
      for h in range(H):
         s_h_hat=traj_s[h,k]    # s_h
         a_h_hat=traj_s[h,k]    
         ss_h_hat=traj_s[h+1,k]
         N[h][s_h_hat][a_h_hat]+=1

      num_samples=7
      tested_risk_measure[k]=(1/gamma)*np.array([np.exp(gamma*sum(sample_trajectory(H,policy_learnt,model_true,reward_true,output_reward=True)))\
                                                 for _ in range(num_samples)]).mean()
      evaluation_results=(None,None,None,tested_risk_measure)
   # since this is a model-free algorithm, returned model is None, model errors are also none.
   return (policy_learnt, None, evaluation_results)

