# -*- coding: utf-8 -*-
#%matplotlib inline

import gym
import matplotlib
import sys

from collections import defaultdict



def generate_episode(env, policy):
    '''
    Generates an episode using policy 
    Args:
        env     : openAI gym environment
        policy  : policy function
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
    '''   
    episode = []
    state = env.reset()  # Get Initial State (S0)
    #done = False
    #while not done:
    for t in range(100):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break    
    return episode

def update_V_AA(episode, V, returns_sum, returns_count, discount_factor=1.0):
    """
    For each time step in the episode we carry out the first visit monte carlo method, 
    checking if this is the first index of this state. 
    Get the discounted reward and add it to the total reward for that 
    state. Increment the times we have seen this state action pair 
    and finally update the V values
    """
    # obtain the states, actions, and rewards
    #states, actions, rewards = zip(*episode)
    state_values = []
    G = 0
    for state, action, reward in reversed(episode):
        # Sum up all rewards since the first occurance
        if 0 < reward:
            # got some reward
            G = reward + discount_factor*G
        else:    
            # no reward
            G = reward + discount_factor*G
        
        #G = reward + discount_factor*G
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
    
def update_V_DB(episode, V, returns_sum, returns_count, discount_factor=1.0):
    """
    For each time step in the episode we carry out the first visit monte carlo method, 
    checking if this is the first index of this state. 
    Get the discounted reward and add it to the total reward for that 
    state. Increment the times we have seen this state action pair 
    and finally update the V values
    """

    # Find all states the we've visited in this episode
    # We convert each state to a tuple so that we can use it as a dict key
    states_in_episode = set([tuple(x[0]) for x in episode])
    for state in states_in_episode:
        # Find the first occurance of the state in the episode
        first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
        # Sum up all rewards since the first occurance
        G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
        # Calculate average return for this state over all sampled episodes
        returns_sum[state] += G
        returns_count[state] += 1.0
        V[state] = returns_sum[state] / returns_count[state]


            

def first_visit_mc_prediction(policy, env, num_episodes, discount_factor=1.0, update_V=update_V_AA ):
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
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
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
    return 0 if score >= 18 else 1


if __name__ == '__main__':
    if "../" not in sys.path:
        sys.path.append("../") 
    from lib import plotting
    
    env_name = 'Blackjack-v0'
    env = gym.make(env_name)
    
    matplotlib.style.use('ggplot')
    
    
    V_10k = first_visit_mc_prediction(basic_policy, env, num_episodes=10000)
    plotting.plot_value_function(V_10k, title="10,000 Steps")
    
    V_500k = first_visit_mc_prediction(basic_policy, env, num_episodes=500000)
    plotting.plot_value_function(V_500k, title="500,000 Steps")

