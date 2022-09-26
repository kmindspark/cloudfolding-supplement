import numpy as np
import matplotlib.pyplot as plt
# import cdf of normal distribution from scipy
from scipy.stats import norm

num_coverage_bins = 200
max_coverage = 45000

def s_sprime_to_distribution(filename, delta=False):
    # return a function that gives a distribution given the current coverage bin
    array = np.load(filename)
    # for now, just simply give a shifted coverage distribution
    # calculate delta coverage distribution
    delta_coverages = []
    print(len(array.shape[0]), filename)
    for transition_idx in range(len(array.shape[0])):
        s_sprime = array[transition_idx]
        delta_coverage = s_sprime[1] - s_sprime[0] if delta else s_sprime[1]
        delta_coverages.append(delta_coverage)
    delta_coverages = np.array(delta_coverages)

    # calculate the distribution discretized into 200 bins
    def get_coverage_dist(cur_state):
        if delta:
            resultant_coverages = delta_coverage + cur_state
        # discretize these into num_coverage_bins bins
        discretized_coverages = np.zeros(num_coverage_bins)
        for coverage in resultant_coverages:
            discretized_coverages[coverage_to_bin(coverage)] += 1
        # normalize the distribution
        discretized_coverages /= np.sum(discretized_coverages)

    return get_coverage_dist

class Primitives:
    num_primitives = 3
    PICKPLACE = 0
    FLING = 1
    DROP = 2

def coverage_to_bin(coverage):
    return min(max(int(coverage * num_coverage_bins), 0), num_coverage_bins - 1)

def reward(state, action):
    if state == num_coverage_bins - 1:
        return 0
    return -1

class CoverageEnvironment:
    def __init__(self):
        self.pickplace_delta_distribution_mu_delta = 0.01
        self.pickplace_delta_distribution_stdev = 0.02
        self.fling_delta_distribution_mu = 0.45
        self.fling_delta_distribution_stdev = 0.10
        self.drop_delta_distribution_mu = 0.55
        self.drop_delta_distribution_stdev = 0.05
        self.cache = {}

    def reset_coverage(self):
        return coverage_to_bin(np.random.uniform(0.2, 0.6))

    def step_gaussian(self, current_coverage, primitive):
        if (current_coverage, primitive) in self.cache:
            return self.cache[(current_coverage, primitive)]

        s_prime_probabilities = np.zeros(num_coverage_bins)
        if current_coverage == num_coverage_bins - 1:
            s_prime_probabilities[-1] = 1
            return s_prime_probabilities

        # return an array of probabilities for each of the num_coverage_bins s'
        normal_mu, normal_std = 0.0, 0.0
        if primitive == Primitives.PICKPLACE:
            normal_mu, normal_std = self.pickplace_delta_distribution_mu_delta + current_coverage/num_coverage_bins, self.pickplace_delta_distribution_stdev
        elif primitive == Primitives.FLING:
            normal_mu, normal_std = self.fling_delta_distribution_mu, self.fling_delta_distribution_stdev        
        elif primitive == Primitives.DROP:
            normal_mu, normal_std = self.drop_delta_distribution_mu, self.drop_delta_distribution_stdev
    
        # discretize the distribution into num_coverage_bins bins and renormalize
        for i in range(0, num_coverage_bins):
            # get the total probability mass associated with bin i
            bin_probability_mass = norm.cdf((i + 1) / num_coverage_bins, normal_mu, normal_std) - norm.cdf((i) / num_coverage_bins, normal_mu, normal_std)
            # add the probability mass to the corresponding bin
            s_prime_probabilities[i] = bin_probability_mass
        self.cache[(current_coverage, primitive)] = s_prime_probabilities / np.sum(s_prime_probabilities)
        return self.cache[(current_coverage, primitive)]

class CoverageEnvironment2:
    def __init__(self, bc_dist, drop_dist):
        self.bc_dist = bc_dist
        self.drop_dist = drop_dist

    def step_gaussian(self, current_coverage, primitive):
        if primitive == Primitives.PICKPLACE:
            return self.bc_dist[current_coverage]
        elif primitive == Primitives.FLING:
            return self.drop_dist[current_coverage]
        elif primitive == Primitives.DROP:
            return self.drop_dist[current_coverage]
    
if __name__ == "__main__":
    # initialize the environment
    pick_place_distribution_fn = s_sprime_to_distribution("data/pickplace_distribution.npy", delta=True)
    drop_distribution_fn = s_sprime_to_distribution("data/drop_distribution.npy", delta=False)

    env = CoverageEnvironment2()
    q_sa = np.zeros((num_coverage_bins, Primitives.num_primitives)) 

    converged = False
    num_iterations = 0
    changes = []
    while num_iterations < 1500:
        print(num_iterations)
        num_iterations += 1
        prev_qsa = np.copy(q_sa)
        for state in range(0, num_coverage_bins):
            for primitive in range(0, Primitives.num_primitives):
                # get the probability of reaching this state given the current action
                s_prime_probabilities = env.step(state, primitive)
                
                # get average of the q values of the next state using argmax sampling of the q values of the next state
                q_sa[state, primitive] = reward(state, primitive) + np.sum(np.multiply(s_prime_probabilities, np.max(q_sa, axis=1)))
        changes.append(np.sum(np.abs(prev_qsa - q_sa)))


    for state in range(0, num_coverage_bins):
        # print optimal action for each state
        print("State:", state, "Action:", np.argmax(q_sa[state, :]))
