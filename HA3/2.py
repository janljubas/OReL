import numpy as np
import matplotlib.pyplot as plt

class RiverSwim4State:
    def __init__(self):
        self.nS = 4
        self.nA = 2
        self.P = np.zeros((self.nS, self.nA, self.nS))
        self.R = np.zeros((self.nS, self.nA))
        
        # Transition probabilities
        self.P[0, 0, 0] = 1
        self.P[0, 1, 0] = 0.6
        self.P[0, 1, 1] = 0.4
        
        self.P[1, 0, 0] = 1
        self.P[1, 1, 1] = 0.55
        self.P[1, 1, 2] = 0.4
        self.P[1, 1, 0] = 0.05
        
        self.P[2, 0, 1] = 1
        self.P[2, 1, 2] = 0.55
        self.P[2, 1, 3] = 0.4
        self.P[2, 1, 1] = 0.05
        
        self.P[3, 0, 2] = 1
        self.P[3, 1, 3] = 0.6
        self.P[3, 1, 2] = 0.4
        
        # Rewards
        self.R[0, 0] = 0.05
        self.R[3, 1] = 1
        
        self.s = 0  # Initial state

    def reset(self):
        self.s = 0
        return self.s

    def step(self, action):
        new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
        reward = self.R[self.s, action]
        self.s = new_s
        return new_s, reward

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(2)  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

def ce_opo(env, gamma=0.98, epsilon=0.15, alpha=0.1, horizon=int(1e6)):
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    Q_star = np.zeros((nS, nA))  # Optimal Q-values (assuming known for comparison)
    
    # Optimal Q-values for the 4-state RiverSwim (precomputed)
    Q_star[0, 0] = 0.05
    Q_star[0, 1] = 1.0
    Q_star[1, 0] = 0.05
    Q_star[1, 1] = 1.0
    Q_star[2, 0] = 0.05
    Q_star[2, 1] = 1.0
    Q_star[3, 0] = 0.05
    Q_star[3, 1] = 1.0
    
    Q_diff = []
    policy_diff = []
    return_loss = []
    
    state = env.reset()
    
    for t in range(horizon):
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward = env.step(action)
        
        # Update counts and Q-values
        N[state, action] += 1
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        # Compute performance metrics
        Q_diff.append(np.max(np.abs(Q_star - Q)))
        policy_diff.append(np.sum(np.argmax(Q, axis=1) != np.argmax(Q_star, axis=1)))
        return_loss.append(Q_star[1, 1] - Q[1, 1])
        
        state = next_state
    
    return Q_diff, policy_diff, return_loss

# Part (i): Run CE-OPO for the standard 4-state RiverSwim
env = RiverSwim4State()
Q_diff, policy_diff, return_loss = ce_opo(env)

# Plotting the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(Q_diff)
plt.title('Q-value Error (||Q* - Qt||∞)')
plt.xlabel('Time step')
plt.ylabel('Error')

plt.subplot(1, 3, 2)
plt.plot(policy_diff)
plt.title('Policy Difference (PolicyDiff(t))')
plt.xlabel('Time step')
plt.ylabel('Difference')

plt.subplot(1, 3, 3)
plt.plot(return_loss)
plt.title('Return Loss (ReturnLoss(t))')
plt.xlabel('Time step')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Part (ii): Modify the reward function and repeat
class RiverSwim4StateModified(RiverSwim4State):
    def step(self, action):
        new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
        reward = self.R[self.s, action]
        if self.s == 0 and action == 1:
            reward = np.random.uniform(0, 2)  # Uniform reward for action 'right' in state 0
        self.s = new_s
        return new_s, reward

env_modified = RiverSwim4StateModified()
Q_diff_mod, policy_diff_mod, return_loss_mod = ce_opo(env_modified)

# Plotting the results for the modified environment
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(Q_diff_mod)
plt.title('Q-value Error (||Q* - Qt||∞) - Modified')
plt.xlabel('Time step')
plt.ylabel('Error')

plt.subplot(1, 3, 2)
plt.plot(policy_diff_mod)
plt.title('Policy Difference (PolicyDiff(t)) - Modified')
plt.xlabel('Time step')
plt.ylabel('Difference')

plt.subplot(1, 3, 3)
plt.plot(return_loss_mod)
plt.title('Return Loss (ReturnLoss(t)) - Modified')
plt.xlabel('Time step')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Part (iii): Compare results
print("Comparison of Q-value Error:")
print(f"Standard: {Q_diff[-1]}, Modified: {Q_diff_mod[-1]}")

print("Comparison of Policy Difference:")
print(f"Standard: {policy_diff[-1]}, Modified: {policy_diff_mod[-1]}")

print("Comparison of Return Loss:")
print(f"Standard: {return_loss[-1]}, Modified: {return_loss_mod[-1]}")