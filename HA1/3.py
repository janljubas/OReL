import numpy as np
import matplotlib.pyplot as plt

def run_simulation(delta, T, use_modified):
    """
    We simulate the UCB1 algorithm (modified or the origial one) and output the regret over time.
    The function inputs:
         delta -> the subomptimality gap 
         `T` -> time horizon
         `use_modified` -> indicator for which algorithm version
    """

    mu = [0.5 + 0.5 * delta, 0.5 - 0.5 * delta]
    counts = np.array([0.0, 0.0])
    sums = np.array([0.0, 0.0])
    cumulative_regret = np.zeros(T) # we initialize it to zeros
    
    # Initial plays (each arm once)
    # Step 1 (t=0)
    a = 0
    reward = np.random.binomial(1, mu[a])
    counts[a] += 1
    sums[a] += reward
    cumulative_regret[0] = 0  # Optimal arm played (it is the only one)
    
    # Step 2 (t=1)
    a = 1
    reward = np.random.binomial(1, mu[a])
    counts[a] += 1
    sums[a] += reward
    cumulative_regret[1] = delta  # Suboptimal arm played
    
    # Steps 3 to T (t=2 to T-1)
    for t in range(2, T):
        current_time = t + 1  # Time starts at 1 for t=0
        ucb = []
        for arm in range(2):
            mu_hat = sums[arm] / counts[arm]
            if use_modified:
                confidence = np.sqrt(np.log(current_time) / counts[arm])
            else:
                confidence = np.sqrt((3 * np.log(current_time)) / (2 * counts[arm]))
            ucb.append(mu_hat + confidence)
        
        a = np.argmax(ucb)
        reward = np.random.binomial(1, mu[a])
        counts[a] += 1
        sums[a] += reward
        current_regret = delta if a == 1 else 0
        cumulative_regret[t] = cumulative_regret[t-1] + current_regret
    
    return cumulative_regret



########################################################################################################


if __name__ == "__main__":

    # Parameters as defined in the Exercises
    T = 100000
    deltas = [0.8, 0.25, 0.125, 0.0625, 0.0001]
    num_runs = 20


    print("Simulation started")

    for delta in deltas:
        
        # Lists for collecting data for original and modified UCB
        regrets_original = []
        regrets_modified = []
        
        # Running simulations
        for _ in range(num_runs):
            regrets_original.append(run_simulation(delta, T, use_modified=False))
            regrets_modified.append(run_simulation(delta, T, use_modified=True))
        
        # Computing statistics
        avg_original = np.mean(regrets_original, axis=0)
        std_original = np.std(regrets_original, axis=0)

        avg_modified = np.mean(regrets_modified, axis=0)
        std_modified = np.std(regrets_modified, axis=0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(avg_original, label='UCB1 Original')
        plt.plot(avg_modified, label='UCB1 Modified')
        plt.fill_between(range(T), avg_original - std_original, avg_original + std_original, alpha=0.2)
        plt.fill_between(range(T), avg_modified - std_modified, avg_modified + std_modified, alpha=0.2)
        plt.xlabel('Time Step')
        plt.ylabel('Empirical Pseudo Regret')
        plt.title(f'Δ = {delta}')
        plt.legend()
        plt.grid(True)
        plt.show()