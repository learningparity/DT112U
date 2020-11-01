#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def lookahead(env, state, V, discount_factor):
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
    action_values = np.zeros(env.nA)
    for a in range(env.nA):
        # [(prob, next_state, reward, done)] = env.P[state][a]
        for prob, next_state, reward, done in env.P[state][a]:
            action_values[a] += prob * (reward + discount_factor * V[next_state])
    return action_values



def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    
    # Start with a "random" (all 0) value function, 
    # except for V(terminal) which must be zero.
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    
    iter = 0
    while True:
        iter += 1
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            
            best_action_value = 0
            # Look at the possible next actions
            for a in range(env.nA):
                # For each action, look at the possible next states and
                # calculate the expected value
                [(prob, next_state, reward, done)] = env.P[s][a]
                current_value = prob * (reward + discount_factor * V[next_state])
                best_action_value = max(current_value, best_action_value)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            print("Number of iterations= ",iter)
            break
    
    # Output a deterministic policy, such that 
    # Policy(s) = argmax(action, Sum(prob * (reward + discount_factor * V[next_state])))
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        best_action = lookahead(env,s, V, discount_factor)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V


def value_iteration_denny(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = lookahead(env, s, V, discount_factor)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = lookahead(env, s, V, discount_factor)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V


policy, v = value_iteration(env)

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


policy, v = value_iteration_denny(env)
#policy, v = value_iteration(env)

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


# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

