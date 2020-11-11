# -*- coding: utf-8 -*-
#%matplotlib inline

import gym
import matplotlib
import sys
import numpy as np

from collections import defaultdict

g_epsilon_counter = 1



def update_V(episode, V, returns_sum, returns_count, discount_factor=1.0):
    """
    For each time step in the episode we carry out the first visit monte carlo method, 
    checking if this is the first index of this state. 
    Get the discounted reward and add it to the total reward for that 
    state. Increment the times we have seen this state action pair 
    and finally update the V values
    Args:
        episode        : array of (state, action, reward) tuples
        V              : State value function 
                        (average return for each visit in the episode
        returns_sum    : array of the sum of all the returns in the state
        returns_count  : array of the count of all the visits to the state
        discount_factor: Gamma, the discount factor
    """
    # obtain the states, actions, and rewards
    #states, actions, rewards = zip(*episode)
    state_values = []
    G = 0
    for state, action, reward in reversed(episode):
        # Sum up all rewards since the first occurance
        G = reward + discount_factor*G
        state_values.append((state, action, reward, G))
    
    state_values.reverse()
    states_visited = []
    
    for state, action, reward, G in state_values:
        # Check if first visit
        if (state) not in states_visited:
            # first visit
            states_visited.append((state))
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
        #else:
            # Not a first visit - skip this state
            

def first_visit_mc_prediction(policy, env, num_episodes, discount_factor=1.0 ):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    Ref: Sutton & Barto chapter 5.1
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: The discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
            
    """

    # Keeps track of sum and count of returns for each state to calculate an average. 
    # We could use an array to save all returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    #first_occurence = dict()    
    
    for i_episode in range(1,num_episodes+1):        
        # Generate an episode, array of (state, action, reward) tuples
        episode = generate_episode(env, policy)
        # Calculate and store the return for each visit in the episode
        update_V(episode, V, returns_sum, returns_count, discount_factor=1.0)
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
    return V  
    

def basic_policy(observation):
    """
    A policy that sticks if the player score is > 18 and hits otherwise.
    """
    (score, dealer_score, usable_ace) = observation
    if usable_ace:
        return 0 if score >= 16 else 1
    else:
        return 0 if score >= 14 else 1



def generate_episode_det(env, policy):
    '''
    Generates an episode using deterministic policy 
    Args:
        env     : openAI gym environment
        policy  : policy function
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
    '''   
    episode = []
    state = env.reset()  # Get Initial State (S0)
    for t in range(100):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break    
    return episode


def generate_episode(env, policy):
    '''
    Generates an episode using policy 
    Args:
        env     : openAI gym environment
        policy  : policy function
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
    '''   
    episode = []
    state = env.reset()
    done = False
    while not done:
        probs = policy(state)
        # np.arange(len(probs)) -  [0,1] Possible actions stay or hit
        action = np.random.choice(np.arange(len(probs)), p=probs)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q:      A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
        epsilon:The probability to select a random action . float between 0 and 1.
        nA:     Number of actions in the environment.
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        #A[best_action] = 1.0 - epsilon + epsilon / nA
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, epsilon, discount_factor=1.0):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_count = defaultdict(int)
    # Replaced with incremental calutation
    
    
    # Q = state-action-value function
    # A nested dictionary that maps state -> (action -> action-value).
    # Default value = array with zeros, one for each possible action
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        #if (i_episode % 100000 == 0) and (0.1 < epsilon):
        #    epsilon /= 2     
        #    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = generate_episode(env, policy)

        G = 0
        state_action_returns = []
        for state, action, reward in reversed(episode):
            # Sum up all rewards since the first occurance
            G = reward + discount_factor*G
            state_action_returns.append((state, action, G))
            
        state_action_returns.reverse()
        visited = set() 
        for state, action, G in state_action_returns:
            # Find all (state, action) pairs we've visited in this episode
            # We convert each (state, action) pair to a tuple so that 
            # we can use it as a dict key
            sa_pair = (state, action)
            if sa_pair not in visited:
                visited.add(sa_pair)
                N = returns_count[sa_pair] + 1
                returns_count[sa_pair] = N
                # Calculate average return for this state over all sampled episodes
                # new estimate = 1 / N * [sample - old estimate]
                Q[state][action] += (1 / N)*(G-Q[state][action])
                # The policy is improved implicitly by changing the Q dictionary 
    return Q, policy


def QtoVP(Q):
    """
    Create value function from action-value function
    by picking the best action at each state
    Args:
        Q: Array with State-Action-Values tuple(state, actions).
    Returns:
        V : Array with the optimal value function.
        P : Array with the optimal action in each state.
    """
    V = defaultdict(float)
    P = defaultdict(float)
    for state, actions in Q.items():
        optimal_action = np.argmax(actions)
        action_value = actions[optimal_action]
        V[state] = action_value
        P[state] = optimal_action
    return V, P
    

if __name__ == '__main__':
    if "../" not in sys.path:
        sys.path.append("../") 
    from lib import plotting
    
    env_name = 'Blackjack-v0'
    env = gym.make(env_name)
    
    matplotlib.style.use('ggplot')
    
    #V_10k = first_visit_mc_prediction(basic_policy, env, num_episodes=10000)
    #plotting.plot_value_function(V_10k, title="10,000 Steps")
    
    #V_500k = first_visit_mc_prediction(basic_policy, env, num_episodes=500000)
    #plotting.plot_value_function(V_500k, title="500,000 Steps")


    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=0.3)
    V,P = QtoVP(Q)
    plotting.plot_value_function(V, title="Optimal Value Function - 100k, 0.3")
    plotting.plot_value_function(P, title="Optimal Policy - 100k, 0.3")
    
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=1000000, epsilon=0.2)
    V,P = QtoVP(Q)
    plotting.plot_value_function(V, title="Optimal Value Function - 1m, 0.2")
    plotting.plot_value_function(P, title="Optimal Policy - 1m, 0.2")
    
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=10000000, epsilon=0.1)
    V,P = QtoVP(Q)
    plotting.plot_value_function(V, title="Optimal Value Function - 10m, 0.1")
    plotting.plot_value_function(P, title="Optimal Policy - 10m, 0.1")
    
