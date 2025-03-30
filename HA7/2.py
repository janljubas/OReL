import numpy as np
import matplotlib.pyplot as plt

# Read data
data = []
with open('HA7\data_preprocessed_features', 'r') as f:
    for line in f:
        parts = line.strip().split()
        action = int(parts[0])
        click = int(parts[1])
        data.append((action, click))

# Compute cumulative rewards per arm
cumulative_rewards = np.zeros(16)
for action, click in data:
    cumulative_rewards[action] += click

# Determine arm indices for subsets
sorted_arms = np.argsort(cumulative_rewards)
best_arm = sorted_arms[-1]
worst_arm = sorted_arms[0]
two_worst = sorted_arms[:2].tolist()
three_worst = sorted_arms[:3].tolist()
median_arm = sorted_arms[7]  # Lower median

subsets = {
    'all arms': list(range(16)),
    'best and worst arm': [best_arm, worst_arm],
    'best and 2 worst arms': [best_arm] + two_worst,
    'best and 3 worst arms': [best_arm] + three_worst,
    'best, median and the worst arm': [best_arm, median_arm, worst_arm]
}

# Precompute subset data and best cumulative for each subset
subset_info = {}
for key, arms in subsets.items():
    subset_rows = [row for row in data if row[0] in arms]
    K_prime = len(arms)
    arm_cumulative = {a: 0 for a in arms}
    best_cumulative = []
    for row in subset_rows:
        a_log, r = row
        arm_cumulative[a_log] += K_prime * r
        best_cumulative.append(max(arm_cumulative.values()))
    subset_info[key] = {
        'data': subset_rows,
        'arms': arms,
        'K_prime': K_prime,
        'best_cumulative': best_cumulative
    }

# UCB1 implementation
def run_ucb(subset_rows, subset_arms, K_prime, best_cumulative, num_repetitions=10):
    all_regrets = []
    for _ in range(num_repetitions):
        N = {a: 0 for a in subset_arms}
        cumulative_sum = {a: 0.0 for a in subset_arms}
        algo_cumulative = 0
        regrets = []
        for t, (a_log, r) in enumerate(subset_rows):
            if t < K_prime:
                A_t = subset_arms[t % K_prime]
            else:
                ucb_values = {}
                for a in subset_arms:
                    if N[a] == 0:
                        ucb = float('inf')
                    else:
                        mu = cumulative_sum[a] / N[a]
                        ucb = mu + np.sqrt((K_prime**2 * np.log(t+1)) / N[a])
                    ucb_values[a] = ucb
                A_t = max(ucb_values, key=lambda x: ucb_values[x])
            if A_t == a_log:
                R_t = K_prime * r
                N[A_t] += 1
                cumulative_sum[A_t] += R_t
                algo_cumulative += R_t
            current_best = best_cumulative[t]
            regrets.append(current_best - algo_cumulative)
        all_regrets.append(regrets)
    return np.mean(all_regrets, axis=0), np.std(all_regrets, axis=0)

# EXP3 implementation
def run_exp3(subset_rows, subset_arms, K_prime, best_cumulative, num_repetitions=10):
    all_regrets = []
    for _ in range(num_repetitions):
        weights = {a: 1.0 for a in subset_arms}
        algo_cumulative = 0
        regrets = []
        for t, (a_log, r) in enumerate(subset_rows):
            t1 = t + 1
            eta = np.sqrt(np.log(K_prime) / (K_prime * t1))
            total_weight = sum(weights.values())
            probs = [weights[a] / total_weight for a in subset_arms]
            A_t = np.random.choice(subset_arms, p=probs)
            
            # Update weights for ALL arms
            for a in subset_arms:
                if a == A_t and A_t == a_log:
                    loss = 1 - r
                    l_tilde = K_prime * loss
                else:
                    l_tilde = 0
                weights[a] *= np.exp(-eta * l_tilde)
            
            # Track cumulative reward
            if A_t == a_log:
                algo_cumulative += K_prime * r
            
            # Compute regret
            current_best = best_cumulative[t]
            regrets.append(current_best - algo_cumulative)
        all_regrets.append(regrets)
    return np.mean(all_regrets, axis=0), np.std(all_regrets, axis=0)

# Random strategy
# def random_regret(best_cumulative, subset_rows, K_prime):
#     avg_cumulative = []
#     total = 0
#     count = 0
#     for t, (a_log, r) in enumerate(subset_rows):
#         total += r
#         count += 1
#         avg = total / count
#         avg_scaled = avg * K_prime * (t + 1) / len(info['arms'])
#         current_best = best_cumulative[t]
#         avg_regret = current_best - avg_scaled
#         avg_cumulative.append(avg_regret)
#     return avg_cumulative


# Replace the existing random_regret function with:
def random_regret(best_cumulative, subset_rows, K_prime):
    sum_r = 0
    random_cumulative = []
    for t, (a_log, r) in enumerate(subset_rows):
        sum_r += r
        current_random = sum_r / K_prime  # Proper scaling
        current_best = best_cumulative[t]
        random_cumulative.append(current_best - current_random)
    return random_cumulative


# Theoretical EXP3 regret bound
def exp3_bound(K_prime, T):
    t_values = np.arange(0, T+1)
    bound = 2 * K_prime * np.sqrt(t_values * np.log(K_prime))
    return bound

# Generate plots for each subset
for key in subsets:
    info = subset_info[key]
    subset_rows = info['data']
    K_prime = info['K_prime']
    best_cumulative = info['best_cumulative']
    T = len(subset_rows)

    print(K_prime)
    
    # Run algorithms
    ucb_mean, ucb_std = run_ucb(subset_rows, info['arms'], K_prime, best_cumulative)
    exp3_mean, exp3_std = run_exp3(subset_rows, info['arms'], K_prime, best_cumulative)
    
    # Random strategy
    random_reg = random_regret(best_cumulative, subset_rows, K_prime)
    
    # EXP3 bound
    bound = exp3_bound(16, T)
    bound_2 = exp3_bound(K_prime, T)

    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ucb_mean, label='UCB1')
    plt.fill_between(range(T), ucb_mean - ucb_std, ucb_mean + ucb_std, alpha=0.2)
    plt.plot(exp3_mean, label='EXP3')
    plt.fill_between(range(T), exp3_mean - exp3_std, exp3_mean + exp3_std, alpha=0.2)
    plt.plot(random_reg, label='Random')
    plt.plot(bound, label=f"EXP3 Bound with K = {K_prime}", linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.title(f'Regret for {key}')
    plt.legend()
    plt.show()