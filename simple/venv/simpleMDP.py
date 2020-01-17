import numpy as np

from simpleMDP import *


class simpleMDP(object):
    def __init__(self,nstate,naction,nep,noisy=1):
        self.nstate = nstate
        self.naction = naction
        self.nep = nep
        self.noisy = 1
        self.R = {}
        self.P = {}
        for state in range(nstate):
            for action in range(naction):
                self.R[state,action] = (1,noisy)
                self.P[state,action] = np.ones(nstate)/nstate
        self.reset()

    def reset(self):
        self.timestep = 0
        self.state = 0
        self.pContinue = 1

    def advance(self,action):
        if self.R[self.state,action][1]<1e-9:
            reward = self.R[self.state,action][0]
        else:
            reward = np.random.normal(loc = self.R[self.state,action][0],scale = self.R[self.state,action][1])
        newstate = np.random.choice(self.nstate,p=self.P[self.state,action])
        self.state =  newstate
        self.timestep+=1
        if self.timestep == self.nep:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        return reward,newstate,pContinue

    def compute_qVal(self):
        qVal=  {}
        qMax = {}
        qMax[self.nep] = np.zeros(self.nstate)
        for i in range(self.nep):
            j = self.nep-i-1
            qMax[j] = np.zeros(self.nstate)
            for s in range(self.nstate):
                qVal[s,j] = np.zeros(self.naction)
                for a in range(self.naction):
                    qVal[s,j][a] = self.R[s,a][0]+np.dot(self.P[s,a],qMax[j+1])

                qMax[j][s]  = np.max(qVal[s,j])

        return qVal,qMax


class bandit(simpleMDP):
    def __init__(self,actions=3,nep=10,noisy=1):
        super().__init__(nstate=1,naction = actions,nep=nep,noisy=noisy)

class RiverSwim(simpleMDP):
    def __init__(self,states = 6,actions =2,nep=50,noisy =1,gamma = 0.99):
        super().__init__(naction=actions,nep = nep,noisy=noisy,nstate=states)
        self.gamma = gamma
        self.P[0,0] = [1,0,0,0,0,0]
        self.P[0,1] = [0.7,0.3,0,0,0,0]
        self.P[1,0] = [1,0,0,0,0,0]
        self.P[1,1] = [0.1,0.6,0.3,0,0,0]
        self.P[2,0] = [0,1,0,0,0,0]
        self.P[2,1] = [0,0.1,0.6,0.3,0,0]
        self.P[3,0] = [0,0,1,0,0,0]
        self.P[3,1] = [0,0,0.1,0.6,0.3,0]
        self.P[4,0] = [0,0,0,1,0,0]
        self.P[4,1] = [0,0,0,0.1,0.6,0.3]
        self.P[5,0] = [0,0,0,0,1,0]
        self.P[5,1] = [0,0,0,0,0.7,0.3]
        for state in range(self.nstate):
            for action in range(self.naction):
                if state==0 and action==0:
                    self.R[state,action] = 5
                elif state==5 and action == 1:
                    self.R[state,action] = 10000
                else:
                    self.R[state,action] = 0

    def compute_qVal(self):
        qVal=  {}
        qMax = {}
        qMax[self.nep] = np.zeros(self.nstate)
        for i in range(self.nep):
            j = self.nep-i-1
            qMax[j] = np.zeros(self.nstate)
            for s in range(self.nstate):
                qVal[s,j] = np.zeros(self.naction)
                for a in range(self.naction):
                    qVal[s,j][a] = self.R[s,a]+self.gamma*np.dot(self.P[s,a],qMax[j+1])

                qMax[j][s]  = np.max(qVal[s,j])

        return qVal,qMax

    def advance(self,action):
        reward = self.R[self.state,action]
        newstate = np.random.choice(self.nstate, p=self.P[self.state,action])
        self.state = newstate
        self.timestep += 1
        if self.timestep == self.nep:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        return reward,newstate,pContinue

    def reset(self):
        self.timestep = 0
        self.state = 1
        self.pContinue = 1

class SixArms(simpleMDP):
    def __init__(self,actions = 6,states = 7,nep = 1000000,noisy =1):
        super().__init__(naction= actions,nstate = states,nep = nep,noisy = noisy)
        self.P[0,0] = [0,1,0,0,0,0,0]
        self.P[0,1] = [0.85,0.15,0,0,0,0,0]
        self.P[0,2] = [0.9,0,0.1,0,0,0,0]
        self.P[0,3] = [0.95,0,0,0.05,0,0,0]
        self.P[0,4] = [0.97,0,0,0,0.03,0,0]
        self.P[0,5] = [0.99,0,0,0,0,0,0.01]

class HardExplorationMDP(simpleMDP):
    def __init__(self,actions = 2,states=4,nep=10000,noisy=1,r_param = 0.7,p_reward = 1000,gamma = 0.99):
        super().__init__(naction=actions,nstate=states,nep = nep,noisy= noisy)
        self.gamma = gamma
        self.r_param = r_param
        self.p_reward = p_reward
        self.P[0,0] = [1,0,0,0]
        self.P[0,1] = [0,1,0,0]
        self.P[1,0] = [1,0,0,0]
        self.P[1,1] = [0,0,1,0]
        self.P[2,0] = [0,1,0,0]
        self.P[2,1] = [0,0,0,1]
        self.P[3,0] = [0,0,1,0]
        self.P[3,1] = [0,0,0,1]
        for s in range(states):
            for a in range(actions):
                self.R[s,a] = [0]
        self.R[0,0] = [2]


    def Terminal_Reward(self,param,p_reward):
        #终态奖励是服从伯努利分布的
        if np.random.uniform()>1-param:
            return p_reward
        else:
            return 0

    def compute_qVal(self):
        qVal = {}
        qMax = {}
        qMax[self.nep] = np.zeros(self.nstate)
        for i in range(self.nep):
            j = self.nep - i - 1
            qMax[j] = np.zeros(self.nstate)
            for s in range(self.nstate):
                qVal[s, j] = np.zeros(self.naction)
                for a in range(self.naction):
                    if (s,a) == (3,1):
                        qVal[s,j][a] = self.p_reward*self.r_param+self.gamma * np.dot(self.P[s, a], qMax[j + 1])
                    else:
                        qVal[s, j][a] = self.R[s, a] + self.gamma * np.dot(self.P[s, a], qMax[j + 1])

                qMax[j][s] = np.round(np.max(qVal[s, j]),2)

        return qVal, qMax

    def advance(self,action):
        reward = self.R[self.state,action]
        if (self.state,action)==(3,1):
            reward = self.Terminal_Reward(param=self.r_param,p_reward=self.p_reward)
        newstate = np.random.choice(self.nstate, p=self.P[self.state,action])
        self.state = newstate
        self.timestep += 1
        if self.timestep == self.nep:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        return reward,newstate,pContinue

if __name__ == '__main__':
    hardexploration = HardExplorationMDP(gamma=1)
    qval, qmax = hardexploration.compute_qVal()

    for k, v in qmax.items():
        print("timestep: {}".format(k))
        print("qmax: {}".format(v))
        print()