import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Bandit Environments
# =============================================
class StochasticBandit:
    def __init__(self, mus):
        self.mus = mus
        self.K = len(mus)
    
    def pull(self, arm):
        return np.random.rand() < self.mus[arm]

class AdversarialBandit:
    def __init__(self, K):
        self.K = K
    
    def pull(self, arm):
        # Adversarial reward: 0 for chosen arm, 1 for others (unobserved)
        return 0  # Only the chosen arm's reward is observed (0)

# =============================================
# Algorithms
# =============================================
class UCB1:
    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.avg_rewards = np.zeros(K)
    
    def select_arm(self, t):
        ucb = self.avg_rewards + np.sqrt(np.log(t + 1) / (2 * (self.counts + 1e-8)))
        return np.argmax(ucb)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.avg_rewards[arm] += (reward - self.avg_rewards[arm]) / self.counts[arm]

class EXP3:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.weights = np.ones(K)
        self.t = 0  # Track current time step
    
    def select_arm(self):
        self.t += 1
        sum_weights = np.sum(self.weights)
        if sum_weights <= 0:
            self.weights = np.ones(self.K)
            sum_weights = self.K
        probs = self.weights / sum_weights
        # Numerical stability: clip probabilities
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()
        arm = np.random.choice(self.K, p=probs)
        return arm, probs
    
    def update(self, arm, reward, probs):
        eta_t = np.sqrt(np.log(self.K) / (16 * self.t))
        r_hat = np.zeros(self.K)
        # Avoid division by zero in r_hat
        r_hat[arm] = reward / (probs[arm] + 1e-8)
        # Update weights with clipping
        self.weights *= np.exp(eta_t * r_hat)
        self.weights = np.clip(self.weights, 1e-8, 1e8)

# =============================================
# Experiment Runner
# =============================================
def run_experiment(T, bandit, algorithm_class, is_adversarial=False):
    if algorithm_class == EXP3:
        algo = algorithm_class(bandit.K, T)
    else:
        algo = algorithm_class(bandit.K)
    
    regret = np.zeros(T)
    mu_star = np.max(bandit.mus) if not is_adversarial else 1.0  # Best possible reward
    
    for t in range(T):
        if algorithm_class == EXP3:
            arm, probs = algo.select_arm()
        else:
            arm = algo.select_arm(t + 1)  # t starts at 1
        
        reward = bandit.pull(arm)
        regret[t] = mu_star - (reward if not is_adversarial else 0)
        
        if algorithm_class == EXP3:
            algo.update(arm, reward, probs)
        else:
            algo.update(arm, reward)
    
    return np.cumsum(regret)

# =============================================
# Plotting Function
# =============================================
def plot_results(ucb_data, exp3_data, title):
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(ucb_data.shape[1])
    
    # UCB1 Mean ± Std
    ucb_mean = np.mean(ucb_data, axis=0)
    ucb_std = np.std(ucb_data, axis=0)
    plt.plot(ucb_mean, label='UCB1', color='blue')
    plt.fill_between(time_steps, ucb_mean - ucb_std, ucb_mean + ucb_std, alpha=0.2, color='blue')
    
    # EXP3 Mean ± Std
    exp3_mean = np.mean(exp3_data, axis=0)
    exp3_std = np.std(exp3_data, axis=0)
    plt.plot(exp3_mean, label='EXP3', color='orange')
    plt.fill_between(time_steps, exp3_mean - exp3_std, exp3_mean + exp3_std, alpha=0.2, color='orange')
    
    plt.xlabel('Time Step')
    plt.ylabel('Average Pseudo-Regret')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================
# Main Execution
# =============================================
def main():
    T = 10000
    reps = 20
    
    # -----------------------------------------
    # IID Setting (K=2,4,8,16; Δ=0.25,0.125,0.0625)
    # -----------------------------------------
    iid_ucb = []
    iid_exp3 = []
    
    for K in [2, 4, 8, 16]:
        for delta in [0.25, 0.125, 0.0625]:
            mus = [0.5] + [0.5 - delta] * (K - 1)
            bandit = StochasticBandit(mus)
            
            # Run UCB1
            ucb_regrets = np.array([run_experiment(T, bandit, UCB1) for _ in range(reps)])
            iid_ucb.append(ucb_regrets)
            
            # Run EXP3
            exp3_regrets = np.array([run_experiment(T, bandit, EXP3) for _ in range(reps)])
            iid_exp3.append(exp3_regrets)
    
    # Aggregate results
    iid_ucb = np.vstack(iid_ucb)
    iid_exp3 = np.vstack(iid_exp3)
    plot_results(iid_ucb, iid_exp3, 'IID Setting: UCB1 vs EXP3 (All K and Δ)')
    
    # -----------------------------------------
    # Adversarial Setting (K=2)
    # -----------------------------------------
    adv_ucb = []
    adv_exp3 = []
    K_adv = 2
    bandit_adv = AdversarialBandit(K_adv)
    
    # Run UCB1
    adv_ucb = np.array([run_experiment(T, bandit_adv, UCB1, is_adversarial=True) for _ in range(reps)])
    
    # Run EXP3
    adv_exp3 = np.array([run_experiment(T, bandit_adv, EXP3, is_adversarial=True) for _ in range(reps)])
    
    plot_results(adv_ucb, adv_exp3, 'Adversarial Setting: UCB1 vs EXP3 (K=2)')

if __name__ == "__main__":
    main()