import numpy as np
import pandas as pd
from simpleMDP import *
import matplotlib.pyplot as plt
class bandit(simpleMDP):
    def __init__(self,actions=3,nep=10,noisy=1):
        self.noisy = noisy
        self.nep = nep
        self.actions = actions
        super().__init__(nstate=1,naction = self.actions,nep=self.nep,noisy=self.noisy)
    def changeReward(self):
        for k,v in self.R.items():
            self.R[k] = (0,0.1)
        self.R[0,self.actions-1] = (1,1)

def egreedy(epsilon,naction,N,experiment):
    bd = bandit(actions=naction,nep=N)
    bd.changeReward()
    print(bd.R)
    if epsilon>1 or epsilon<0:
        return "ERROR"
    scores=np.zeros(N+1)
    for _ in range(experiment):
        score = np.zeros(N+1)
        total_reward = np.zeros(naction)
        hat_q = np.zeros(naction)
        count = np.zeros(naction)
        for i in range(1,N+1):
            if np.random.rand()<epsilon:
                action = np.argmax(hat_q)
            else:
                action = np.random.choice(naction,p = np.ones(naction)/naction)
            count[action]+=1
            reward = bd.advance(action)[0]
            total_reward[action]+=reward
            hat_q[action] = hat_q[action]+(1/count[action])*(reward-hat_q[action])
            score[i]=reward
        scores+=score
    return scores/experiment

def rmax(naction,N):
    bd = bandit(actions=naction, nep=N)
    total_reward = np.zeros(naction)
    hat_q = np.ones(naction)+1
    count = np.zeros(naction)
    for i in range(1,N+1):
        action = np.argmax(hat_q)
        count[action]+=1
        reward = bd.advance(action)[0]
        total_reward[action]+=reward
        hat_q[action] = total_reward[action]/count[action]
    return hat_q,total_reward,count

def UCB(naction,N):
    bd = bandit(actions=naction,nep=N)
    total_reward = np.zeros(naction)
    hat_q = np.zeros(naction)
    count = np.zeros(naction)
    for i in range(1,N+1):
        bonus = np.sqrt(2*np.log(i)/(count+1e-9))
        temp = hat_q+bonus
        action = np.argmax(temp)
        count[action]+=1
        reward = bd.advance(action)[0]
        total_reward[action]+=reward
        hat_q[action] = total_reward[action]/count[action]
    return hat_q,total_reward,count

def UCB(naction,N):
    bd = bandit(actions=naction,nep=N)
    total_reward = np.zeros(naction)
    hat_q = np.zeros(naction)
    count = np.zeros(naction)
    for i in range(1,N+1):
        bonus = np.sqrt(2*np.log(i)/(count+1e-9))
        temp = hat_q+bonus
        action = np.argmax(temp)
        count[action]+=1
        reward = bd.advance(action)[0]
        total_reward[action]+=reward
        hat_q[action] = total_reward[action]/count[action]
    return hat_q,total_reward,count

def MBIE_EB(naction,N,beta):
    bd = bandit(actions=naction,nep=N)
    total_reward = np.zeros(naction)
    hat_q = np.zeros(naction)
    count = np.zeros(naction)
    for i in range(1,N+1):
        bonus = beta/np.sqrt(count+1e-9)
        temp = hat_q+bonus
        action = np.argmax(temp)
        count[action]+=1
        reward = bd.advance(action)[0]
        total_reward[action]+=reward
        hat_q[action] = total_reward[action]/count[action]
    return hat_q, total_reward, count

def BEB(naction,N,beta):
    bd = bandit(actions=naction,nep=N)
    total_reward = np.zeros(naction)
    hat_q = np.zeros(naction)
    count = np.zeros(naction)
    for i in range(1,N+1):
        bonus = beta/(count+1e-9)
        temp = hat_q+bonus
        action = np.argmax(temp)
        count[action]+=1
        reward = bd.advance(action)[0]
        total_reward[action]+=reward
        hat_q[action] = total_reward[action]/count[action]
    return hat_q, total_reward, count



if __name__ == '__main__':
    scores= egreedy(naction=10,epsilon=0.99,N=2000,experiment=100)
    plt.plot(scores)
    plt.show()