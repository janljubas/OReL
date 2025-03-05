import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2000
mu_values = [0.25, 0.375, 0.4375]
num_runs = 10
algorithms = ['FTL', 'Hedge', 'ReparamHedge', 'AnytimeHedgeSimple', 'AnytimeHedgeTighter']

def simulate(mu, num_runs, T):
    results = {algo: [] for algo in algorithms}
    for run in range(num_runs):
        np.random.seed(run)
        X = np.random.binomial(1, mu, T)
        cum_sum = np.cumsum(X)
        best_action_loss = np.minimum(cum_sum, np.arange(1, T+1) - cum_sum)
        
        # FTL
        ftl_cum_loss = np.zeros(T)
        cum_loss_0, cum_loss_1 = 0, 0
        current_ftl_loss = 0
        for i in range(T):
            if i == 0:
                pred = 0
            else:
                pred = 0 if cum_loss_0 < cum_loss_1 else 1 if cum_loss_0 > cum_loss_1 else 0
            current_X = X[i]
            loss = 1 if pred != current_X else 0
            current_ftl_loss += loss
            ftl_cum_loss[i] = current_ftl_loss
            cum_loss_0 += current_X
            cum_loss_1 = (i + 1) - cum_loss_0
        results['FTL'].append(ftl_cum_loss - best_action_loss)
        
        # Hedge algorithms
        for algo in algorithms[1:]:
            if algo == 'Hedge':
                eta = np.sqrt(2 * np.log(2) / T)
            elif algo == 'ReparamHedge':
                eta = np.sqrt(8 * np.log(2) / T)
            else:
                eta = None
            
            weights = np.array([1.0, 1.0])
            algo_cum_loss = np.zeros(T)
            cum_loss = 0
            for i in range(T):
                current_X = X[i]
                if algo == 'AnytimeHedgeSimple':
                    eta_t = np.sqrt(np.log(2) / (i + 1))
                elif algo == 'AnytimeHedgeTighter':
                    eta_t = 2 * np.sqrt(np.log(2) / (i + 1))
                else:
                    eta_t = eta
                
                prob0 = weights[0] / np.sum(weights)
                action = np.random.choice([0, 1], p=[prob0, 1 - prob0])
                loss = 1 if action != current_X else 0
                cum_loss += loss
                algo_cum_loss[i] = cum_loss
                loss0, loss1 = current_X, 1 - current_X
                update = np.exp(-eta_t * np.array([loss0, loss1]))
                weights *= update
            results[algo].append(algo_cum_loss - best_action_loss)
    return results

# Plot for each mu
plt.figure(figsize=(15, 10))
for idx, mu in enumerate(mu_values):
    results = simulate(mu, num_runs, T)
    plt.subplot(3, 1, idx + 1)
    for algo in algorithms:
        avg_regret = np.mean(results[algo], axis=0)
        plt.plot(avg_regret, label=algo)
        plt.fill_between(range(T), avg_regret - np.std(results[algo], axis=0), avg_regret + np.std(results[algo], axis=0), alpha=0.2)
    plt.title(f'μ = {mu}')
    plt.xlabel('Time')
    plt.ylabel('Pseudo-Regret')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()

# Part 3: Adversarial sequence
def adversarial_simulation(num_runs, T):
    results = {algo: [] for algo in algorithms}
    X = np.array([t % 2 for t in range(T)])
    best_action_loss = T // 2  # Best fixed action
    
    for run in range(num_runs):
        # FTL
        ftl_cum_loss = np.zeros(T)
        cum_loss_0, cum_loss_1 = 0, 0
        current_ftl_loss = 0
        for i in range(T):
            if i == 0:
                pred = 0
            else:
                pred = 0 if cum_loss_0 < cum_loss_1 else 1 if cum_loss_0 > cum_loss_1 else 0
            loss = 1 if pred != X[i] else 0
            current_ftl_loss += loss
            ftl_cum_loss[i] = current_ftl_loss
            cum_loss_0 += X[i]
            cum_loss_1 = (i + 1) - cum_loss_0
        results['FTL'].append(ftl_cum_loss - best_action_loss)
        
        # Hedge algorithms
        for algo in algorithms[1:]:
            if algo == 'Hedge':
                eta = np.sqrt(2 * np.log(2) / T)
            elif algo == 'ReparamHedge':
                eta = np.sqrt(8 * np.log(2) / T)
            else:
                eta = None
            
            weights = np.array([1.0, 1.0])
            algo_cum_loss = np.zeros(T)
            cum_loss = 0
            for i in range(T):
                current_X = X[i]
                if algo == 'AnytimeHedgeSimple':
                    eta_t = np.sqrt(np.log(2) / (i + 1))
                elif algo == 'AnytimeHedgeTighter':
                    eta_t = 2 * np.sqrt(np.log(2) / (i + 1))
                else:
                    eta_t = eta
                
                prob0 = weights[0] / np.sum(weights)
                action = np.random.choice([0, 1], p=[prob0, 1 - prob0])
                loss = 1 if action != current_X else 0
                cum_loss += loss
                algo_cum_loss[i] = cum_loss
                loss0, loss1 = current_X, 1 - current_X
                update = np.exp(-eta_t * np.array([loss0, loss1]))
                weights *= update
            results[algo].append(algo_cum_loss - best_action_loss)
    return results

# Plot adversarial results
adv_results = adversarial_simulation(num_runs, T)
plt.figure(figsize=(12, 6))
for algo in algorithms:
    avg_regret = np.mean(adv_results[algo], axis=0)
    plt.plot(avg_regret, label=algo)
    plt.fill_between(range(T), avg_regret - np.std(adv_results[algo], axis=0), avg_regret + np.std(adv_results[algo], axis=0), alpha=0.2)
plt.title('Adversarial Sequence Regret')
plt.xlabel('Time')
plt.ylabel('Regret')
plt.legend()
plt.grid()
plt.show()