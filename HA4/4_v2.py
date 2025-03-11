import numpy as np
import matplotlib.pyplot as plt

class BinaryPredictionGame:
    def __init__(self, horizon=2000, mu_values=None):
        """
        Initialize the binary prediction game simulation
        
        Args:
        - horizon (int): Total number of rounds
        - mu_values (list): List of bias values to test
        """
        if mu_values is None:
            mu_values = [1/2, 1/4, 1/8, 1/16, 1/32]
        self.horizon = horizon
        self.mu_values = mu_values

    def generate_sequence(self, mu, horizon):
        """
        Generate a random binary sequence with given bias
        
        Args:
        - mu (float): Bias parameter
        - horizon (int): Sequence length
        
        Returns:
        - numpy array of binary outcomes
        """
        return np.random.binomial(1, mu, horizon)

    def ftl_algorithm(self, sequence):
        """
        Follow the Leader (FTL) algorithm implementation
        
        Args:
        - sequence (numpy array): Binary sequence
        
        Returns:
        - numpy array of predictions
        """
        predictions = np.zeros_like(sequence)
        cumulative_rewards = np.zeros(2)
        
        for t in range(1, len(sequence)):
            # Choose action based on previous best performance
            action = 0 if cumulative_rewards[0] >= cumulative_rewards[1] else 1
            predictions[t] = action
            
            # Update cumulative rewards
            cumulative_rewards[predictions[t]] += 1 - sequence[t]
        
        return predictions

    def hedge_algorithm(self, sequence, eta):
        """
        Hedge algorithm implementation
        
        Args:
        - sequence (numpy array): Binary sequence
        - eta (float): Learning rate
        
        Returns:
        - numpy array of predictions
        """
        predictions = np.zeros_like(sequence)
        weights = np.ones(2)
        
        for t in range(1, len(sequence)):
            # compute probabilities based on weights
            probabilities = weights / np.sum(weights)
            
            # stochastic action selection
            action = np.random.choice(2, p=probabilities)
            predictions[t] = action
            
            # update weights
            loss = 1 - sequence[t]
            weights[action] *= np.exp(-eta * loss)
        
        return predictions

    def compute_pseudo_regret(self, sequence, predictions):
        best_action_rewards = np.zeros(2)
        for t in range(len(sequence)):
            best_action_rewards[sequence[t]] += 1
        
        best_action = np.argmax(best_action_rewards)
        
        regret = np.sum(sequence[1:] != best_action) - np.sum(sequence[1:] != predictions[1:])
        return regret

    def run_simulation(self):
        results = {
            'FTL': [],
            'Hedge_sqrt': [],
            'Hedge_anytime': []
        }
        
        for mu in self.mu_values:   # different mu values
            ftl_regrets = []
            hedge_sqrt_regrets = []
            hedge_anytime_regrets = []
            
            for _ in range(10):  # 10 repetitions

                sequence = self.generate_sequence(mu, self.horizon)
                
                # FTL
                ftl_predictions = self.ftl_algorithm(sequence)
                ftl_regret = self.compute_pseudo_regret(sequence, ftl_predictions)
                ftl_regrets.append(ftl_regret)
                
                # Hedge (sqrt version)
                eta_sqrt = np.sqrt(2 * np.log(2) / self.horizon)
                hedge_sqrt_predictions = self.hedge_algorithm(sequence, eta_sqrt)
                hedge_sqrt_regret = self.compute_pseudo_regret(sequence, hedge_sqrt_predictions)
                hedge_sqrt_regrets.append(hedge_sqrt_regret)
                
                # "Anytime" Hedge
                eta_anytime = np.sqrt(np.log(2) / self.horizon)
                hedge_anytime_predictions = self.hedge_algorithm(sequence, eta_anytime)
                hedge_anytime_regret = self.compute_pseudo_regret(sequence, hedge_anytime_predictions)
                hedge_anytime_regrets.append(hedge_anytime_regret)
            
            # average regrets
            results['FTL'].append((np.mean(ftl_regrets), np.std(ftl_regrets)))
            results['Hedge_sqrt'].append((np.mean(hedge_sqrt_regrets), np.std(hedge_sqrt_regrets)))
            results['Hedge_anytime'].append((np.mean(hedge_anytime_regrets), np.std(hedge_anytime_regrets)))
        
        return results

    def plot_results(self, results):

        plt.figure(figsize=(10, 6))
        x = self.mu_values
        
        # Plot mean regrets with error bars
        plt.errorbar(x, [r[0] for r in results['FTL']], 
                     yerr=[r[1] for r in results['FTL']], 
                     fmt='o-', label='FTL')
        plt.errorbar(x, [r[0] for r in results['Hedge_sqrt']], 
                     yerr=[r[1] for r in results['Hedge_sqrt']], 
                     fmt='s-', label='Hedge (sqrt)')
        plt.errorbar(x, [r[0] for r in results['Hedge_anytime']], 
                     yerr=[r[1] for r in results['Hedge_anytime']], 
                     fmt='^-', label='Hedge (Anytime)')
        
        ticks = np.arange(0.1, 0.6, 0.1)
        plt.xticks(ticks, [f'{t:.1f}' for t in ticks])

        plt.xlabel('μ (Bias)')
        plt.ylabel('Empirical Pseudo Regret')
        plt.title('Performance Comparison: FTL vs Hedge Algorithms')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if (__name__ == "__main__"):
    simulation = BinaryPredictionGame()
    results = simulation.run_simulation()
    simulation.plot_results(results)