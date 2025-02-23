import numpy as np
import sys
import os

# I had problems with importing from the sibling directory, so I did this fix
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from HA1 import RiverSwim   # the import in question


######################################### 1. Monte Carlo Simulation #########################################

def monte_carlo():

    # defined hyperparameters
    gamma = 0.96
    T = 300
    n = 50
    nS = 5

    env = RiverSwim.riverswim(nS)
    V_hat = np.zeros(nS)

    for s in range(nS):

        total = 0.0
        
        for _ in range(n):
        
            env.reset()
            env.s = s  # set the starting state
            current_s = s
        
            discounted_sum = 0.0
        
            for t in range(T):
                
                # defining our policy as per HA2.pdf
                if current_s in [0, 1, 2]:
                    action = np.random.choice([0, 1], p=[0.35, 0.65])
                else:
                    action = 1 # when s = 4 or 5
        
        
                # deciding the reward and the next state s_t+1
                next_s, reward = env.step(action)


                discounted_sum += (gamma ** t) * reward # V^pi estimate as defined in the HA2.pdf
                current_s = next_s
        
            total += discounted_sum # we, of course, accumulate the sum
        
        V_hat[s] = total / n
    
    
    return V_hat

######################################### 2 - Exact Value Calculation #########################################

def exact_v_pi():
    
    # again the hyperparameters
    gamma = 0.96
    nS = 5


    # reward function in the 1st and the 5th state
    R_pi = np.zeros(nS)
    R_pi[0] = 0.05  # defined in the picture?
    R_pi[4] = 1.0

    # transition probability matrix
    P_pi = np.zeros((nS, nS))
    P_pi[0, 0] = 0.74 
    P_pi[0, 1] = 0.26
    # explanation:
    # P(0, 0, L) = 1, P(1, 0, L) = 0, P(0, 0, R) = 0.6, P(1, 0, R) = 0.4 
    # pi(L|0) = 0.35, pi(R|0) = 0.65
    # we multiply them accordingly: P[0, 0] = 0.35*1 + 0.65 * 6 = 0.74

    P_pi[1, 0] = 0.3825
    P_pi[1, 1] = 0.3575
    P_pi[1, 2] = 0.26

    # state 2 transitions
    P_pi[2, 1] = 0.3825
    P_pi[2, 2] = 0.3575
    P_pi[2, 3] = 0.26

    # State 3 transitions
    P_pi[3, 2] = 0.05
    P_pi[3, 3] = 0.55
    P_pi[3, 4] = 0.4

    # State 4 transitions
    P_pi[4, 3] = 0.4
    P_pi[4, 4] = 0.6


    # solve linear system V^pi = (I - gamma*P_pi)^-1 r_pi
    I = np.eye(nS)
    A = I - gamma * P_pi
    V_exact = np.linalg.solve(A, R_pi)
    return V_exact

if __name__ == "__main__":

    V_hat = monte_carlo()
    V_exact = exact_v_pi()

    print("Monte Carlo Approximation of V^π:")
    print(np.round(V_hat, 4))
    print("\nExact Value Function V^π:")
    print(np.round(V_exact, 4))
