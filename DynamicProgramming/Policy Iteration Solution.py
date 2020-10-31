#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv


# In[6]:


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


# In[7]:


# Taken from Policy Evaluation Exercise!
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
   
    # V = Estimate of the value function
    # Initialize V arbitrarily, except that V (terminal) = 0
    # In this implementation we initialize V to zeros 
    V = np.zeros(env.nS)
    iter = 0
    printouts = 0
    print_factor = 1
    while True:
        iter += 1
        delta = 0
        # Loop over all states and perform an update
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action available in each state
                # look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton Barto eq. 4.5.
                    # Sum over all actions i each state    
                    """
                    To think about...
                    How to handle the cases where you are done. For example, in the cliff walking environment, 
                    if done = TRUE, do you add the values for the next_state?
                    if done:
                        print("-- -- -- -- -- -- -- -- -- -- --")
                        print("Done:", done, "Current state:", s, "Next state:", next_state)
                        print("action_prob:", action_prob, "prob:", prob, "Reward:", reward, "V[next_state]", V[next_state])
                        print("-- -- -- -- -- -- -- -- -- -- --")
                        v += action_prob * prob * reward
                    else:
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                    """
                    v += action_prob * prob * (reward + discount_factor * V[next_state])

            # Calculate How much our value function changed (across any states)
            delta = max(delta, abs(v - V[s]))
            # Update the value function
            V[s] = v
            
        # Some printing for debugging
        if 0 == ((iter-1) % print_factor):
            printouts += 1
            print("Value Function:", iter, printouts, print_factor)
            print(V)
            if 0 == (printouts % 10):
                print_factor *= 10
        
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


# In[ ]:


def lookahead(env, state, V):
    """
    The greedy policy takes the action that looks best in the short term,
    after one step of lookahead—according to V.
    Args:
        env: The OpenAI envrionment.
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS
    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A


# In[13]:


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        # Init policy_stable = TRUE
        policy_stable = TRUE
        
        # This is the policy improvment part
        # Loop over all states and update/improve the policy
        for s in range(env.nS):
            # The best action we could take in the current state 
            # under the current policy. Ref: Sutton Barto eq. 4.9.
            best_action = np.argmax(policy[s])
            
            #replace one action
            one step-look-ahead
            
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):            
            
            # Calculate How much our value function changed (across any states)
            delta = max(delta, abs(v - V[s]))
            
            # Stop evaluating once our value function change is below a threshold
        if 0 < delta:
            policy_stable = FALSE
            break   
    return policy, np.zeros(env.nS)


# In[ ]:





# In[14]:


policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# In[15]:


# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


# In[ ]:




