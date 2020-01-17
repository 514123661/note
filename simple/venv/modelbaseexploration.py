import numpy as np
from agent import *
class modelbaseagent(agent):
    def __init__(self,nState,nAciton,epLen,alpha0=1.,mu0=0.,tau0=1.,tau=1.,**kwargs):
        self.nState = nState
        self.naction = nAciton
        self.epLen = epLen
        self.alpha0 = alpha0
        self.mu0 =  mu0
        self.tau0 = tau0
        self.tau = tau
        self.R_prior = {}
        self.P_prior = {}
        self.qVal = {}
        self.qMax = {}

        for s in range(nState):
            for a in range(nAciton):
                self.R_prior[s,a] = self.mu0
                self.P_prior[s,a] = (self.alpha0*np.ones(nState,dtype=np.float32))

    def update_obs(self,obs,reward,action,newobs):
        mu0,tau0 = self.R_prior[obs,action]
        tau = self.tau
        tau1 = (tau*tau0)/(tau+tau0)
        mu1  = reward*(tau0/(tau0+tau)) + mu0*(tau/(tau+tau0))

        self.R_prior[obs,action] = (mu1,tau1)
        self.P_prior[obs,action][newobs] += 1


