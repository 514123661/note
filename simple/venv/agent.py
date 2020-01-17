import numpy as np

class Agent(object):
    def __init__(self):
        pass


    def update_obs(self,obs,action,reward,newobs):
        pass

    def update_policy(self,h):
        pass
    def pick_action(self,obs):
        pass

class MBIE_agent(Agent):
    def __init__(self,A = 0.3,B = 0,nstate=3, naction=3,gamma=0.99,rmax=1):
        self.A = A
        self.B = B
        self.R = {}
        self.R_count = {}
        self.T = {}
        self.T_count = {}
        self.qval = {}
        self.qmax = {}
        self.nstate = nstate
        self.naction = naction
        self.n = {}
        self.rmax = rmax
        self.gamma = gamma
        self.epLen = 6
        for s in range(nstate):
            for a in range(naction):
                self.R[s,a] = rmax
                self.T[s,a] = np.ones(nstate)/nstate

                self.T_count[s,a] = np.zeros(nstate)
                self.qval[s,a] = self.rmax/(1+self.gamma)
                self.n[s,a] = 0
        #print("参数A： {}, 参数B： {}".format(self.A,self.B))


    def update_obs(self,obs,action,reward,newobs):
        self.R[s,a] = (reward+self.R_count[s,a]*self.R[s,a])/(self.R_count[s,a]+1)
        self.R_count[obs, action] += 1
        self.T_count[obs,action][newobs]+=1
        self.T[obs,action] = self.T_count[obs,action]/np.sum(self.T_count)



    def update_policy(self):
        R_confident ,P_confident = self.gen_confident()
        qVal ,qMax = self.compute_qVals_MBIEVI(self.R,self.T,R_confident,P_confident)
        self.qval = qVal
        self.qmax = qMax

    def pick_action(self,obs,timestep):
        action = self.egreedy(state=obs,timestep=timestep)
        return action

    def egreedy(self,state,timestep,epsilon = 0):
        Q = self.qval[state,timestep]
        nactions = Q.size()
        if np.random.uniform()<=epsilon:
            action = np.where(Q==Q.max())
        else:
            action = np.random.choice(naction)
        return action

    def gen_confident(self):
        R_confident = {}
        P_confident = {}
        for s in range(self.nstate):
            for a in range(self.naction):
                R_confident[s,a] = self.A/np.sqrt(self.n[s,a])
                P_confident[s,a] = self.B/np.sqrt(self.n[s,a])
        return R_confident,P_confident



    def compute_qVals_MBIEVI(self, R, P, R_confident, P_confident):
        '''
        通过MBIE计算Q值表
        Args:
            R - R[s,a] ： 奖励平均值 数据类型是浮点型
            P - P[s,a] ： 状态转移频率 数据类型是 |S|维向量
            R_confident - R_confident[s,a] = R的置信度
            P_confident - P_confident[s,a] = P的置信度

        Returns:
            qVals - qVals[state, timestep] 是timestep的Q值
            qMax - qMax[timestep] 是当前timestep的最大Q值
        '''

        qVals = {}
        qMax = {}
        qMax[self.epLen] = np.zeros(self.nstate)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nstate)
            for s in range(self.nstate):
                qVals[s, j] = np.zeros(self.naction)
                for a in range(self.naction):
                    rOpt = R[s, a] + R_confident[s, a]
                    pInd = np.argsort(qMax[j + 1])  # 排序对应步骤2
                    pOpt = P[s, a]
                    # 求最大值
                    if pOpt[np.where[pInd == 0]] + P_confident[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.nstate)
                        pOpt[np.where[pInd == (self.nstate - 1)]] = 1
                    else:
                        pOpt[np.where[pInd == (self.nstate - 1)]] += P_confident[s, a] * 0.5
                        # 步骤6
                        while np.sum(pOpt) > 1:
                            worst = pInd[np.where(pInd == sLoop)]
                            pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                            sLoop += 1
                        # 步骤7&8
                        qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])
                    # 步骤8
                    qMax[j][s] = np.max(qVals[s, j])
            return qVals, qMax

class TD2_varince_model_free_agent(MBIE_agent):
    def __init__(self,nstate=4,naction=2,gamma=0.99,init_var=0,nep = 1000,learn_rate_value = 0.0001,learn_rate_var=0.01):
        super().__init__(A=0.3,B=0.3,nstate=nstate,naction=naction,gamma=gamma,rmax=1)
        self.qval = {}
        self.qmax = {}
        self.nstate = nstate
        self.naction = naction
        self.Q_table = {}
        self.init_var = init_var
        self.learn_rate_value = learn_rate_value
        self.learn_rate_var = learn_rate_var
        self.var={}
        self.nep = nep
        for s in range(nstate):
            self.Q_table[s] = np.zeros(self.naction)
        self.init_var_table(self.init_var)

    def update_obs(self, obs, action, reward, newobs):
        Q_ = 0.3*self.Q_table[obs][action] + 0.7*(reward+self.gamma*max(self.Q_table[newobs])-self.Q_table[obs][action])
        # simplified DVTD
        action_ = self.pick_action(newobs,timestep=None)
        TD_error = reward+self.gamma*np.mean(self.Q_table[newobs])-self.Q_table[obs][action]
        gamma_ = self.gamma**2
        R_ = TD_error**2
        TD_error_ = R_+gamma_*self.var[newobs][action_]-self.var[obs][action]
        self.var[obs][action] = 0.8*self.var[obs][action]+0.2*TD_error
        self.Q_table[obs][action] = Q_
        #print(self.var)


    def pick_action(self,obs,timestep):
        action = self.ucb_var(obs)
        return action



    def init_var_table(self,init_var):
        for s in range(self.nstate):
            self.var[s] = init_var*np.ones(self.naction)

    def ucb_var(self,obs):
        q = self.Q_table[obs]
        var =self.var[obs]
        explorationB = np.sqrt(var)
        action = q+explorationB
        #print(action)
        if action[0]==action[1]:
            return np.random.choice([0,1])
        else:
            return np.argmax(action)




if __name__ == '__main__':
    agent = TD2_varince_model_free_agent()
