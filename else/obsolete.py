# def main():
#     #initialize
#     '''
#     reward:[H,nS,nA]       r_h(s,a)
#     muhat: [1,nS]         \widehat\mu(a)
#     That:  [H,nS,nS,nA]     \widehat\mathbb{T}_{h}(s'|s,a)
#     Ohat:  [H,nO,nS]       \widehat\mathbb{nO}_{h}(o|s)   
#     sigmahat: [H,F,nS]    \widehat{\sigma}_{h,f_h}^{k}(s)
#     Nhat1:  [H,nS]        \widehat{N}_{h}^{k}(s)
#     Nhat2:  [H,nS,nA]      \widehat{N}_{h}^{k}(s,a)
#     tt:     [H,nS,nA]      \mathsf{t}_h^{k}(s,a)
#     oo:     [H,nS]        \mathsf{o}_{h+1}^{k}(s,a)
#     bouns   [H,nS,nA]      \mathsf{b}_{h}^k(s,a)
#     '''
#     mu_hat=1/nS*torch.ones([1,nS]).reshape(nS)
#     T_hat=mu_hat.reshape([nS,1,1]).repeat(1,nS,nA)
#     O_hat=(1/nO*torch.ones([1,nO])).transpose(0,1).tile(1,nS)

#     sigma_hat=torch.zeros([H,F,nS])
#     N_hatS=torch.ones([H,nS,nA])
#     N_hatO=torch.ones([H,nS])
#     tt=torch.ones([H,nS,nA])
#     oo=torch.zeros([H,nS])
#     bonus=torch.zeros([H,nS,nA])

#     # these counter are used to record
#     '''
#     \sum_{t=1}^{k} \mathds{1}\{s',s,a\} and \sum_{t=1}^{k} \mathds{1}\{o,s\}
#     in line 35.
#     '''
#     OS_cnt=torch.zeros([H,nO,nS])
#     SSA_cnt=torch.zeros([H,nS,nS,nA])
#     for k in range(K):
#         #  //Planning
#         #  //Forward belief propagation
#         sigma_hat[0,:]=mu_hat                 # line 6
#         for h in range(0,H,1):
#             # //Update risk belief by Eq.(35)   line 9
#             sigma_hat[h+1][f[h+1]]=\
#                 nO * torch.diag(O_hat[h,o,:]) * T_hat[h,a,:,:]\
#                     *torch.diag(torch.exp(gamma*reward[h,:,a]))\
#                         * sigma_hat[h][f[h]]
#             # //Prepare bonus by Eq.(62)        line 11 - 12
#             tt[h]=torch.min(torch.ones_like(tt[h]),3*np.sqrt(nS*H*iota)*(torch.pow(N_hatS[h],-0.5)))
#             oo[h]=torch.min(torch.ones_like(oo[h]),3*np.sqrt(nO*H*iota)*(torch.pow(N_hatO[h],-0.5)))
#             bonus[h]=np.abs(np.exp(gamma*(H-h+1))-1)*\
#                 torch.min(torch.ones_like(bonus[h]), tt[h]+\
#                           torch.einsum('ijk,i->jk',T_hat[h],oo[h+1]))
#     pi_hat=1
#     # line 29-30
#     '''output trajectory 3 x H:
#     s1 s2 ... sH
#     o1 o2 ... oH
#     a1 a2 ... aH
#     '''
#     #traj=interact(pi_hat)  
#     for h in range(0,H,1):
#         # line 33
#         [s,o,a,ss]=torch.cat((traj[0:3,h],traj[0:1,h+1])).to(torch.int64)    # overflow alert!H+1
#         # line 34
#         N_hatS[h][s][a]=N_hatS[h][s][a]+1
#         N_hatO[h][s]   =N_hatO[h][s]+1
#         # line 35: update T, nO, \mu estimates.
#         OS_cnt[h][o][s]=OS_cnt[h][o][s]+1
#         SSA_cnt[h][ss][s][a]=SSA_cnt[h][ss][s][a]+1
#         T_hat[h,:,:,:]=SSA_cnt[h]/(torch.max(torch.ones([nS,nS,nA]),N_hatS[h,:,:].repeat(nS,1,1)))
#         O_hat[h,:,:]=OS_cnt[h]/(torch.max(torch.ones([nO,nS]),N_hatO[h,:].repeat(nO,1)))
#     s_1_hat=traj[0][1].to(torch.int64)
#     mu_hat=1/k*(F.one_hot(s_1_hat,nS))+(1-1/k)*mu_hat

# #main()  





