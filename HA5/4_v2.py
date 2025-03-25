import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import math

# Set random seed for reproducibility
np.random.seed(42)

class UCB1:
    def __init__(self, K, alpha=2):
        self.K = K  # Number of arms
        self.alpha = alpha  # Exploration parameter
        self.reset()
    
    def reset(self):
        self.counts = np.zeros(self.K)  # Number of times each arm was pulled
        self.values = np.zeros(self.K)  # Estimated reward for each arm
        self.t = 0  # Total number of pulls
    
    def select_arm(self):
        self.t += 1
        # Initially try each arm once
        if np.any(self.counts == 0):
            return np.where(self.counts == 0)[0][0]
        
        # UCB formula
        ucb_values = self.values + np.sqrt(self.alpha * np.log(self.t) / self.counts)
        
        # Handle ties
        max_value = np.max(ucb_values)
        max_indices = np.where(ucb_values == max_value)[0]
        
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            # Random tie-breaking
            return np.random.choice(max_indices)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

class EXP3:
    def __init__(self, K, eta=None):
        self.K = K
        self.eta = eta  # Learning rate
        self.reset()
    
    def reset(self):
        self.weights = np.ones(self.K)
        self.t = 0
    
    def select_arm(self):
        self.t += 1
        # Update eta if it's time-varying
        if self.eta is None:
            self.current_eta = np.sqrt(np.log(self.K) / (self.t * self.K))
        else:
            self.current_eta = self.eta
            
        # Compute probabilities
        probs = self.weights / np.sum(self.weights)
        return np.random.choice(self.K, p=probs)
    
    def update(self, arm, reward):
        # Compute probability distribution
        probs = self.weights / np.sum(self.weights)
        
        # Update only the weight of the selected arm
        estimated_reward = reward / probs[arm]
        
        # Update weights using exponential update
        self.weights[arm] *= np.exp(self.current_eta * estimated_reward)

def run_iid_experiment(K, T, best_arm_reward, suboptimal_rewards, num_repetitions=20):
    """
    Run the i.i.d. experiment with UCB1 and EXP3
    """
    ucb1_regrets = np.zeros((num_repetitions, T))
    exp3_regrets = np.zeros((num_repetitions, T))
    
    for rep in tqdm(range(num_repetitions), desc="Running i.i.d. experiments"):
        # Set up true rewards
        true_means = np.zeros(K)
        true_means[0] = best_arm_reward
        true_means[1:] = suboptimal_rewards
        
        # Initialize algorithms
        ucb1 = UCB1(K)
        exp3 = EXP3(K)
        
        ucb1_cumulative_regret = 0
        exp3_cumulative_regret = 0
        
        for t in range(T):
            # UCB1
            ucb1_arm = ucb1.select_arm()
            ucb1_reward = np.random.binomial(1, true_means[ucb1_arm])
            ucb1.update(ucb1_arm, ucb1_reward)
            ucb1_regret = true_means[0] - true_means[ucb1_arm]
            ucb1_cumulative_regret += ucb1_regret
            ucb1_regrets[rep, t] = ucb1_cumulative_regret
            
            # EXP3
            exp3_arm = exp3.select_arm()
            exp3_reward = np.random.binomial(1, true_means[exp3_arm])
            exp3.update(exp3_arm, exp3_reward)
            exp3_regret = true_means[0] - true_means[exp3_arm]
            exp3_cumulative_regret += exp3_regret
            exp3_regrets[rep, t] = exp3_cumulative_regret
    
    return ucb1_regrets, exp3_regrets

def design_adversarial_sequence(K, T):
    """
    Design an adversarial sequence that forces UCB1 to incur linear regret
    """
    # Create rewards where a different arm is optimal at each time step in a cyclical pattern
    rewards = np.zeros((T, K))
    
    # Create a pattern that cycles through arms, making a different arm optimal at each step
    for t in range(T):
        optimal_arm = t % K
        # All arms give 0 reward except the optimal one
        rewards[t, optimal_arm] = 1.0
    
    return rewards

def run_adversarial_experiment(K, T, rewards, num_repetitions=20):
    """
    Run the adversarial experiment with UCB1 and EXP3 on the designed sequence
    """
    ucb1_regrets = np.zeros((num_repetitions, T))
    exp3_regrets = np.zeros((num_repetitions, T))
    
    for rep in tqdm(range(num_repetitions), desc="Running adversarial experiments"):
        # Initialize algorithms
        ucb1 = UCB1(K)
        exp3 = EXP3(K)
        
        ucb1_cumulative_regret = 0
        exp3_cumulative_regret = 0
        
        for t in range(T):
            # Find the optimal arm for this time step
            optimal_arm = np.argmax(rewards[t])
            
            # UCB1
            ucb1_arm = ucb1.select_arm()
            ucb1_reward = rewards[t, ucb1_arm]
            ucb1.update(ucb1_arm, ucb1_reward)
            ucb1_regret = rewards[t, optimal_arm] - rewards[t, ucb1_arm]
            ucb1_cumulative_regret += ucb1_regret
            ucb1_regrets[rep, t] = ucb1_cumulative_regret
            
            # EXP3
            exp3_arm = exp3.select_arm()
            exp3_reward = rewards[t, exp3_arm]
            exp3.update(exp3_arm, exp3_reward)
            exp3_regret = rewards[t, optimal_arm] - rewards[t, exp3_arm]
            exp3_cumulative_regret += exp3_regret
            exp3_regrets[rep, t] = exp3_cumulative_regret
    
    return ucb1_regrets, exp3_regrets

def plot_results(ucb1_regrets, exp3_regrets, title, filename):
    """
    Plot the average regret and standard deviation for both algorithms
    """
    T = ucb1_regrets.shape[1]
    time_steps = np.arange(1, T+1)
    
    # Calculate means and standard deviations
    ucb1_mean = np.mean(ucb1_regrets, axis=0)
    ucb1_std = np.std(ucb1_regrets, axis=0)
    exp3_mean = np.mean(exp3_regrets, axis=0)
    exp3_std = np.std(exp3_regrets, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    # Plot UCB1
    plt.plot(time_steps, ucb1_mean, label='UCB1', color='blue')
    plt.fill_between(time_steps, ucb1_mean - ucb1_std, ucb1_mean + ucb1_std, 
                    alpha=0.2, color='blue')
    
    # Plot EXP3
    plt.plot(time_steps, exp3_mean, label='EXP3', color='red')
    plt.fill_between(time_steps, exp3_mean - exp3_std, exp3_mean + exp3_std, 
                    alpha=0.2, color='red')
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Pseudo-Regret')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.show()

def main():
    # Parameters
    T = 10000  # Time horizon
    best_arm_reward = 0.5  # μ*
    
    # Loop through different K values
    for K in [2, 4, 8, 16]:
        print(f"\nRunning experiments with K = {K}")
        
        # Different suboptimal arm rewards
        for gap in [1/4, 1/8, 1/16]:
            suboptimal_reward = best_arm_reward - gap
            suboptimal_rewards = np.ones(K-1) * suboptimal_reward
            
            print(f"Gap: {gap}")
            
            # I.I.D. experiments
            ucb1_regrets, exp3_regrets = run_iid_experiment(K, T, best_arm_reward, suboptimal_rewards)
            
            plot_results(
                ucb1_regrets, exp3_regrets,
                f"I.I.D. Setting: K={K}, Gap={gap}", 
                f"iid_k{K}_gap{gap}.png"
            )
        
        # Adversarial setting
        print("Running adversarial experiment")
        adversarial_rewards = design_adversarial_sequence(K, T)
        ucb1_adv_regrets, exp3_adv_regrets = run_adversarial_experiment(K, T, adversarial_rewards)
        
        plot_results(
            ucb1_adv_regrets, exp3_adv_regrets,
            f"Adversarial Setting: K={K}", 
            f"adversarial_k{K}.png"
        )

if __name__ == "__main__":
    main()