## Modifying the C' definition => we want the P' to be 0 when it's not in the suport set !!
    ## in the Value iteration (Algorithm 3 in the lecture notes)
            ## instead of 1/2 * beta we want to choose beta over the support states

# modifying max proba (find out what it0s doing (algorithm 3 etc) and then modify it)






#############################################################################################
################################ CODE FOR 4th EXERCISE ######################################
#############################################################################################

# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2025.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np
import copy as cp
import pylab as pl
from datetime import datetime

current_time = datetime.now().strftime("%H%M")



####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	ENVIRONMENTS

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################




# A simple riverswim implementation with chosen number of state 'nS' chosen in input.
# We arbitrarily chose the action '0' = 'go to the left' thus '1' = 'go to the right'.
# Finally the state '0' is the leftmost, 'nS - 1' is the rightmost.
class riverswim():

	def __init__(self, nS):
		self.nS = nS
		self.nA = 2

		# We build the transitions matrix P, and its associated support lists.
		self.P = np.zeros((nS, 2, nS))
		self.support = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for s in range(nS):
			if s == 0:
				self.P[s, 0, s] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s + 1] = 0.4
				self.support[s][0] += [0]
				self.support[s][1] += [0, 1]
			elif s == nS - 1:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s - 1] = 0.4
				self.support[s][0] += [s - 1]
				self.support[s][1] += [s - 1, s]
			else:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.55
				self.P[s, 1, s + 1] = 0.4
				self.P[s, 1, s - 1] = 0.05
				self.support[s][0] += [s - 1]
				self.support[s][1] += [s - 1, s, s + 1]
		
		# We build the reward matrix R.
		self.R = np.zeros((nS, 2))
		self.R[0, 0] = 0.05
		self.R[nS - 1, 1] = 1

		# We (arbitrarily) set the initial state in the leftmost position.
		self.s = 0

	# To reset the environment in initial settings.
	def reset(self):
		self.s = 0
		return self.s

	# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
	def step(self, action):
		new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
		reward = self.R[self.s, action]
		self.s = new_s
		return new_s, reward



class ErgodicRiverSwim(riverswim):
    def __init__(self, nS):
        super().__init__(nS)
        # Make leftmost state non-absorbing
        self.P[0,0,0] = 0.4
        self.P[0,0,1] = 0.6
        self.support[0][0] = [0, 1]    # override support for (s=0, a=0)

        # Make rightmost state non-absorbing
        self.P[nS-1,1,nS-1] = 0.4
        self.P[nS-1,1,nS-2] = 0.6
        self.support[nS-1][1] = [nS-2, nS-1]
        

####################################################################################################################################################
####################################################################################################################################################


# An implementation of the Value Iteration algorithm for a given environment 'env' in an average reward setting.
# An arbitrary 'max_iter' is a maximum number of iteration, usefull to catch any error in your code!
# Return the number of iterations, the final value, the optimal policy and the gain.
def VI(env, max_iter = 10**3, epsilon = 10**(-2)):

	# The variable containing the optimal policy estimate at the current iteration.
	policy = np.zeros(env.nS, dtype=int)
	niter = 0

	# Initialise the value and epsilon as proposed in the course.
	V0 = np.zeros(env.nS)
	V1 = np.zeros(env.nS)

	# The main loop of the Value Iteration algorithm.
	while True:
		niter += 1
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + sum([V * p for (V, p) in zip(V0, env.P[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy[s] = a
		
		# Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
		gain = 0.5*(max(V1 - V0) + min(V1 - V0))
		diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
		if (max(diff) - min(diff)) < epsilon:
			return niter, V0, policy, gain
		else:
			V0 = V1
			V1 = np.zeros(env.nS)
		if niter > max_iter:
			print("No convergence in VI after: ", max_iter, " steps!")
			return niter, V0, policy, gain





# A simple implementation of the UCRL2 algorithm from Jacksh et al. 2010 with improved L1-Laplace confidence intervals.
class UCRL2_supp:
	def __init__(self, nS, nA, gamma, epsilon = 0.01, delta = 0.05, support = None):
		self.nS = nS
		self.nA = nA
		self.gamma = gamma
		self.delta = delta
		self.epsilon = epsilon
		self.s = None
		self.t = 1      # Track total time steps
		self.t_k = 1    # Start time of the episode k
		self.episode_count = 0
		self.support = support  # Store known transition support

		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA))
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA))
		self.confP = np.zeros((self.nS, self.nA))

		# The current policy (updated at each episode).
		self.policy = np.zeros((self.nS,), dtype=int)


	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]

	# Update the confidence intervals. Set with Laplace-L1 confidence intervals!
	def confidence(self):
		d = self.delta / (self.nS * self.nA)    # union bound over all (s, a)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				K = len(self.support[s][a])
				
                # Reward confidence
				self.confR[s, a] = np.sqrt( ((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n) )
				
                # Transition confidence
				if K == 0:
					self.confP[s, a] = 0
				else:
					log_term = np.log(np.sqrt(n + 1) * (2**K - 2) / d)
					self.confP[s, a] = np.sqrt( (2 * (1 + 1 / n) * log_term) / n )

	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	# From UCRL2 jacksh et al. 2010.
	def max_proba(self, s, a, V0):
		support = self.support[s][a]    # the known support states for (s, a)
		K = len(support)
		if K == 0:
			return np.zeros(self.nS)
		
		n = max(1, self.Nk[s, a])
		
		# Adjusted confidence radius with K instead of S
		beta = np.sqrt((14 * K / n) * np.log((2 * self.nA * self.t_k) / self.delta))

		# Compute optimistic transition within support
		max_p = np.zeros(self.nS)
		remaining_mass = 1.0
		sorted_support = sorted(support, key=lambda x: V0[x], reverse=True)
		
		
		for s_prime in sorted_support:
			alloc = min(remaining_mass, self.hatP[s, a, s_prime] + beta/2)
			max_p[s_prime] = alloc
			remaining_mass -= alloc
			
			if remaining_mass <= 1e-6:
				break
			
		# normalize max_p
		total = max_p.sum()
		if total > 0:
			max_p /= total
		else:
			max_p[support] = 1.0 / K    # fallback to uniform distribution

		return max_p
	

	# The Extended Value Iteration, perform an optimisitc VI over a set of MDP.
	def EVI(self, max_iter=200, epsilon=1e-2):
		
		V0, V1 = np.zeros(self.nS), np.zeros(self.nS)
		
		for _ in range(max_iter):
			
			for s in range(self.nS):
				
				max_val = -np.inf
				for a in range(self.nA):
					
					p = self.max_proba(s, a, V0)
					r_opt = self.hatR[s, a] + self.confR[s, a]
					val = r_opt + np.dot(p, V0)
					if val > max_val:
						max_val = val
						self.policy[s] = a
						
				V1[s] = max_val
				
			# Check convergence
			if np.max(V1 - V0) - np.min(V1 - V0) < epsilon:
				return self.policy

			V0 = V1.copy()
	
		return self.policy

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.t_k = self.t
		self.updateN() # We update the counter Nk.
		self.vk = np.zeros((self.nS, self.nA))

		# Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				self.hatR[s, a] = self.Rsa[s, a] / div
				for next_s in range(self.nS):
					self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

		# Update the confidence intervals and policy.
		self.confidence()
		self.policy = self.EVI()

	# To reinitialize the model and a give the new initial state init.
	def reset(self, init):
		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA))
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA))
		self.confP = np.zeros((self.nS, self.nA))

		# The current policy (updated at each episode).
		self.policy = np.zeros((self.nS,), dtype=int)

		# Set the initial state and last action:
		self.s = init
		self.last_action = -1

		# TODO: reset the self.episode
		self.episode_count = 0

		# Start the first episode.
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state, reward):
		if self.last_action >= 0: # Update if not first action.
			self.Nsas[self.s, self.last_action, state] += 1
			self.Rsa[self.s, self.last_action] += reward
		
		action = self.policy[state]
		if self.vk[state, action] > max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
			self.episode_count += 1
		
		# Update the variables:
		self.vk[state, action] += 1
		self.s = state
		self.last_action = action
		self.t += 1

		return action, self.policy, self.episode_count



# Plotting function.
def plot(data, names, y_label = "Regret", exp_name = "cumulativeRegret", delta=0.05, gstar=None):
	timeHorizon = len(data[0][0])
	colors= ['black', 'blue', 'purple','cyan','yellow', 'orange', 'red']
	nbFigure = pl.gcf().number+1

	# Average the results and plot them.
	avg_data = []
	pl.figure(nbFigure)
	for i in range(len(data)):
		avg_data.append(np.mean(data[i], axis=0))
		pl.plot(avg_data[i], label=names[i], color=colors[i%len(colors)])

	# Compute standard deviantion and plot the associated error bars.
	step=(timeHorizon//10)
	for i in range(len(data)):
		std_data = 1.96 * np.std(data[i], axis=0) / np.sqrt(len(data[i]))
		pl.errorbar(np.arange(0,timeHorizon,step), avg_data[i][0:timeHorizon:step], std_data[0:timeHorizon:step], color=colors[i%len(colors)], linestyle='None', capsize=10)
	
	# Label and format the plot.
	pl.legend()
	pl.xlabel("Time steps", fontsize=13, fontname = "Arial")
	pl.ylabel(y_label, fontsize=13, fontname = "Arial")
	pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))

	# Uncomment below to get log scale y-axis.
	#pl.yscale('log')
	#pl.ylim(1)
	
	if gstar is not None:
		pl.axhline(y=gstar, color='green', linestyle='--', label='$g^*$')
		pl.legend()
	
	# Save the plot.
	name = ""
	for n  in names:
		name += n + "_"
	pl.savefig("Figure_" + name + exp_name + current_time + f"delta={delta}" + '.pdf')

# Test function, plotting the cumulative regret.
def run():
	# Set the environment:
	env = ErgodicRiverSwim(nS=6)

	# Set the time horizon:
	T = 35*10**4
	nb_Replicates = 100

	UCRL2_support = UCRL2_supp(nS=6, nA=2, gamma=0.01, epsilon=0.01, delta=0.05, support=env.support)

	# Set the variables used for plotting.
	cumregret_UCRL2 = [[0] for _ in range(nb_Replicates)]

	# Estimate the optimal gain.
	print("Estimating the optimal gain...",)
	_, _, _, gstar = VI(env, 10**6, 10**(-6))

	# Run the experiments:
	print("Running experiments...")
	episode_counts_list = []
	for i in range(nb_Replicates):
		# Running an instance of UCRL2-L:
		env.reset()
		UCRL2_support.reset(env.s)	# here the episode counter should reset as well!
		reward = 0
		new_s = env.s
		for t in range(T):
			action, _, _ = UCRL2_support.play(new_s, reward)
			new_s, reward = env.step(action)
			cumregret_UCRL2[i].append(cumregret_UCRL2[i][-1]+ gstar - reward)
		print("|" + "#"*int(i/((nb_Replicates - 1) / 33)) + " "*(33 - int(i/((nb_Replicates - 1) / 33))) + "|", end="\r") # Making a rudimentary progress bar!

		episode_counts_list.append(UCRL2_support.episode_count)
	
	avg_episode_counts = np.mean(episode_counts_list)
	print(f"\nAverage episode count: {avg_episode_counts}")
	empirical_gain_data = []
	for i in range(nb_Replicates):
		eg = []
		for t in range(1, T+1):
			eg.append(gstar - (cumregret_UCRL2[i][t-1] / t))
		empirical_gain_data.append(eg)

	# Plot and finish.
	print("\nPlotting...")

	plot([cumregret_UCRL2], ["UCRL2"], y_label = "Cumulative Regret", exp_name = "cumulative_regret", delta=0.05)

	plot([empirical_gain_data], ["UCRL2_L_empirical_gain"], y_label = "Empirical Gain", exp_name = "empirical_gain", delta=0.05, gstar=gstar)

	print('Done!')


run()
