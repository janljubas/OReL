import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy import stats

class RiverSwim:
    def __init__(self, gamma=0.92):
        self.n_states = 5
        self.n_actions = 2
        self.gamma = gamma
        
        # Define state indices for clarity
        self.states = np.arange(self.n_states)
        
        # Actions: 0 = left (downstream), 1 = right (upstream)
        self.actions = [0, 1]
        
        # Initialize transition probabilities and rewards
        self.initialize_mdp()
        
        # Calculate optimal values
        self.calculate_optimal_values()
    
    def initialize_mdp(self):
        # P[s, a, s'] gives transition probability from s to s' under action a
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Action 0 (left/downstream)
        for s in range(self.n_states):
            if s == 0:  # leftmost state
                self.P[s, 0, s] = 1.0  # stay in the same state
            else:
                self.P[s, 0, s-1] = 0.7  # move left with high probability
                self.P[s, 0, s] = 0.3    # small chance to stay
        
        # Action 1 (right/upstream)
        for s in range(self.n_states):
            if s == self.n_states - 1:  # rightmost state
                self.P[s, 1, s] = 0.7    # high probability to stay
                self.P[s, 1, s-1] = 0.3  # small chance to move left
                self.R[s, 1, s] = 1.0    # reward for staying in rightmost state
            elif s == 0:  # leftmost state
                self.P[s, 1, s+1] = 0.6  # move right with some probability
                self.P[s, 1, s] = 0.4    # decent chance to stay
                self.R[s, 1, s] = 0.05   # small reward for left state
            else:  # middle states
                self.P[s, 1, s+1] = 0.6  # move right with some probability
                self.P[s, 1, s] = 0.1    # small chance to stay
                self.P[s, 1, s-1] = 0.3  # decent chance to move left
    
    def calculate_optimal_values(self):
        """Calculate optimal state values using value iteration."""
        threshold = 1e-10
        V = np.zeros(self.n_states)
        
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                Q_values = np.zeros(self.n_actions)
                
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        Q_values[a] += self.P[s, a, s_next] * (self.R[s, a, s_next] + self.gamma * V[s_next])
                
                V[s] = np.max(Q_values)
                delta = max(delta, abs(v - V[s]))
            
            if delta < threshold:
                break
        
        self.V_star = V
        
        # Calculate optimal policy
        self.pi_star = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            Q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    Q_values[a] += self.P[s, a, s_next] * (self.R[s, a, s_next] + self.gamma * V[s_next])
            self.pi_star[s] = np.argmax(Q_values)
    
    def step(self, state, action):
        """Take a step in the environment given state and action."""
        next_state_probs = self.P[state, action]
        next_state = np.random.choice(self.n_states, p=next_state_probs)
        reward = self.R[state, action, next_state]
        return next_state, reward

class UCBQLearning:
    def __init__(self, env, epsilon=0.13, delta=0.05, gamma=0.92, T=2000000):
        self.env = env
        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma
        self.T = T
        
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # Initialize Q-values, visit counts, and bonus parameters
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.N = np.zeros((self.n_states, self.n_actions), dtype=int)
        
        # Set horizon based on discount and epsilon as suggested
        self.H = int(1/(1-gamma) * np.log(1/epsilon))
    
    def get_ucb_bonus(self, s, a, t):
        """Calculate the UCB bonus term."""
        if self.N[s, a] == 0:
            return np.inf
        
        # Log term inside sqrt
        log_term = np.log(self.n_states * self.n_actions * np.log(t + 1) / self.delta)
        
        # Bonus term 
        b = np.sqrt((self.H / self.N[s, a]) * log_term)
        
        return b
    
    def get_action(self, state, t):
        """Select action based on UCB value."""
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            ucb_values[a] = self.Q[state, a] + self.get_ucb_bonus(state, a, t)
        
        return np.argmax(ucb_values)
    
    def get_greedy_policy(self):
        """Return the current greedy policy based on Q-values."""
        return np.argmax(self.Q, axis=1)
    
    def get_policy_value(self, state, policy):
        """Calculate value of a policy from a given state."""
        V = np.zeros(self.n_states)
        
        # Solve the linear system for value function
        threshold = 1e-10
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                a = policy[s]
                
                new_v = 0
                for s_next in range(self.n_states):
                    new_v += self.env.P[s, a, s_next] * (self.env.R[s, a, s_next] + self.gamma * V[s_next])
                
                V[s] = new_v
                delta = max(delta, abs(v - V[s]))
            
            if delta < threshold:
                break
        
        return V[state]
    
    def is_eps_bad(self, state, policy):
        """Check if policy is ε-bad in given state."""
        policy_value = self.get_policy_value(state, policy)
        return policy_value < self.env.V_star[state] - self.epsilon
    
    def train(self, num_runs=1):
        """Train UCB-QL algorithm for multiple runs and track ε-bad timesteps."""
        all_eps_bad_counts = np.zeros((num_runs, self.T))
        
        for run in tqdm(range(num_runs), desc="Training runs"):
            # Reset counters and Q-values for this run
            self.Q = np.zeros((self.n_states, self.n_actions))
            self.N = np.zeros((self.n_states, self.n_actions), dtype=int)
            
            # Start at a random state
            current_state = np.random.randint(0, self.n_states)
            
            # Cumulative count of ε-bad timesteps
            eps_bad_count = 0
            
            for t in range(self.T):
                # Get policy (greedy with respect to Q)
                policy = self.get_greedy_policy()
                
                # Check if policy is ε-bad
                if self.is_eps_bad(current_state, policy):
                    eps_bad_count += 1
                
                # Record cumulative number of ε-bad timesteps
                all_eps_bad_counts[run, t] = eps_bad_count
                
                # Choose action using UCB rule
                action = self.get_action(current_state, t)
                
                # Take action and observe next state and reward
                next_state, reward = self.env.step(current_state, action)
                
                # Update visit count
                self.N[current_state, action] += 1
                
                # Update Q-value (using standard Q-learning update)
                best_next_q = np.max(self.Q[next_state])
                lr = 1.0 / np.sqrt(self.N[current_state, action])  # Learning rate
                
                # Q-learning update
                self.Q[current_state, action] += lr * (reward + self.gamma * best_next_q - self.Q[current_state, action])
                
                # Move to next state
                current_state = next_state
        
        return all_eps_bad_counts

def plot_single_run(eps_bad_counts):
    """Plot sample path of n(t) for a single run."""
    plt.figure(figsize=(10, 6))
    plt.plot(eps_bad_counts)
    plt.xlabel('Timestep (t)')
    plt.ylabel('Cumulative ε-bad timesteps n(t)')
    plt.title('Sample Path of Cumulative ε-bad Timesteps')
    plt.grid(True)
    plt.savefig('single_run.png')
    plt.close()

def plot_average_with_ci(all_eps_bad_counts):
    """Plot average n(t) with 95% confidence intervals."""
    mean_counts = np.mean(all_eps_bad_counts, axis=0)
    std_counts = np.std(all_eps_bad_counts, axis=0)
    n_runs = all_eps_bad_counts.shape[0]
    
    # Calculate 95% confidence interval
    ci_95 = 1.96 * std_counts / np.sqrt(n_runs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_counts, label='Average n(t)')
    plt.fill_between(np.arange(len(mean_counts)), mean_counts - ci_95, mean_counts + ci_95, 
                     alpha=0.3, label='95% Confidence Interval')
    
    plt.xlabel('Timestep (t)')
    plt.ylabel('Cumulative ε-bad timesteps n(t)')
    plt.title(f'Average Cumulative ε-bad Timesteps (over {n_runs} runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_run.png')
    plt.close()

def main():
    # Set parameters
    gamma = 0.92
    epsilon = 0.13
    delta = 0.05
    T = 2000  # 2 million
    
    # Create environment
    env = RiverSwim(gamma=gamma)
    
    # Create agent
    agent = UCBQLearning(env, epsilon=epsilon, delta=delta, gamma=gamma, T=T)
    
    # Reduced T for testing - comment this out for full run
    # T = 20000
    # agent.T = T
    
    # Run single iteration for part (i)
    single_run_counts = agent.train(num_runs=1)[0]
    plot_single_run(single_run_counts)
    
    # Run multiple iterations for part (ii)
    all_eps_bad_counts = agent.train(num_runs=100)
    plot_average_with_ci(all_eps_bad_counts)

if __name__ == "__main__":
    main()