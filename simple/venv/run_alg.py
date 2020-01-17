from agent import MBIE_agent as MBIE
from simpleMDP import RiverSwim
import numpy as np
if __name__ == '__main__':
    riverswim = RiverSwim(nep=6)
    nstate = riverswim.nstate
    naction = riverswim.naction
    gamma  = riverswim.gamma
    R = riverswim.R
    rmax = 0
    for k,v in R.items():
        rmax = max(v,rmax)

    agent = MBIE(nstate = nstate,naction =  naction,gamma=gamma,rmax = rmax)

    np.random.seed(1)
    qVal, qMax = riverswim.compute_qVal()

    eps = 1000
    for ep in range(1,eps+1):
        riverswim.reset()
        pContinue =  1

        agent.update_policy()
        while pContinue>0:
            h = riverswim.timestep
            oldstate = riverswim.state
            action = agent.pick_action(obs=oldstate,timestep=h)

            reward,newstate,pContinue = riverswim.advance(action=action)

            agent.update_obs(obs=oldstate,action = action,reward=reward,newobs = newstate)






