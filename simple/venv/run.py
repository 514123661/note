from agent import TD2_varince_model_free_agent as agent
from simpleMDP import HardExplorationMDP
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = HardExplorationMDP(nep=4,r_param=1,gamma=1,p_reward=100)
    nep =env.nep
    agent = agent(learn_rate_value=0.001,nep = nep,gamma = 1,init_var = 100)
    #np.random.seed(1)
    qval,qmax = env.compute_qVal()

    eps =1000
    for ep in range(1,eps+1):
        env.reset()
        pContinue=1
        while pContinue>0:
            h = env.timestep
            oldstate = env.state
            action = agent.pick_action(obs=oldstate,timestep=h)

            reward,newstate,pContinue = env.advance(action=action)

            agent.update_obs(obs=oldstate,action = action,reward=reward,newobs = newstate)
        print("q表",agent.Q_table)
        print("方差",agent.var)
        print( )


    print(agent.var)
    print(agent.Q_table)
    print(qmax)
