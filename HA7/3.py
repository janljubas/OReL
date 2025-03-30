import numpy as np
import matplotlib.pyplot as plt
from HA7_gridworld import Four_Room_Teleportation, display_4room_policy

def value_iteration_average_reward(P, R, epsilon=1e-10, max_iter=10000):
    """
    Perform Value Iteration for Average-Reward MDP
    
    Parameters:
    - P: Transition probability matrix (states x actions x states)
    - R: Reward matrix (states x actions)
    - epsilon: Convergence threshold
    - max_iter: Maximum number of iterations
    
    Returns:
    - policy: Optimal policy
    - g_star: Optimal average reward (gain)
    - b_star: Bias function
    """
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    # Init bias and gain
    b = np.zeros(n_states)
    g = 0
    
    for iteration in range(max_iter):
        prev_b = b.copy()
        prev_g = g
        
        # compute Q-values
        Q = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                # compute expected value considering current bias
                Q[s, a] = R[s, a] - g + np.sum(P[s, a] * b)
        
        for s in range(n_states):
            b[s] = np.max(Q[s])
        
        # "centering" the bias function
        b -= np.mean(b)
        
        g_diff = np.abs(g - np.mean(np.max(Q, axis=1)))
        b_diff = np.max(np.abs(b - prev_b))
        
        # update gain
        g = np.mean(np.max(Q, axis=1))
        
        # check convergence criteria
        if g_diff < epsilon and b_diff < epsilon:
            break
    
    # Compute optimal policy
    policy = np.argmax(Q, axis=1)
    
    return policy, g, b

def main():
    # Create the grid-world environment
    env = Four_Room_Teleportation()
    
    # Perform Value Iteration
    policy, g_star, b_star = value_iteration_average_reward(
        env.P, env.R, epsilon=1e-6
    )
    
    # Calculate the span of the bias function
    bias_span = np.max(b_star) - np.min(b_star)
    
    # Print results
    print(f"Optimal Gain (g*): {g_star}")
    print(f"Bias Function Span: {bias_span}")
    
    # Visualize the policy
    policy_matrix = display_4room_policy(policy)
    
    # Print policy matrix
    print("\nOptimal Policy:")
    for row in policy_matrix:
        print(" ".join(row))
    
    # Return the results as a dictionary for additional processing if needed
    return {
        'policy': policy,
        'optimal_gain': g_star,
        'bias_span': bias_span,
        'policy_matrix': policy_matrix
    }

if __name__ == "__main__":
    results = main()

    print(results['optimal_gain'] + results['bias_span'])