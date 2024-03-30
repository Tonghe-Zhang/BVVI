


def RSVI(P, nS, nA, gamma=0.9, eps=1e-3):
    import numpy as np
    '''
    Risk sensitive value iteration in the tabular setting, adapted from Fei et al. ref: 
    '''
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    next_value_function=np.zeros(nS)
    Q_function=np.zeros((nS,nA))
    
    while True:
      value_function=next_value_function
      for s in range(nS):
          for a in range(nA):
             # compute Q[s][a]
             Q_function[s][a]=0
             for i in range(len(P[s][a])):
                p,ss,r,terminal=P[s][a][i]
                Q_function[s][a]=Q_function[s][a]+p*(r+gamma*value_function[ss]*(1-terminal))    #*(1-terminal)
      next_value_function=np.max(Q_function, axis=1)
      policy=np.argmax(Q_function,axis=1)
      if np.max(np.abs(next_value_function-value_function)) < eps:
         break
    return value_function, policy