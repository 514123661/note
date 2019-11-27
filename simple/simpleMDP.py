import numpy as np


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

if __name__ == '__main__':
    simpleMDP = simpleMDP(nstate=1,naction=3,nep = 10)
    qval,qmax = simpleMDP.compute_qVal()
    for i in range(10):
        action = np.random.choice([0,1,2],p = np.ones(3)/3)
        reward,newstate,pContinue = simpleMDP.advance(action)
        print("action: {},  reward:  {}".format(action,reward))
