#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv







# Taken from Policy Evaluation Exercise!
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    This version has better convergence compared to the policy_eval_copy() function
    
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
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    iter = 0
    while True:
        iter += 1
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    if done:
                        v += action_prob * prob * reward
                    else:
                        v += action_prob * prob * (reward + discount_factor * V[next_state])

            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # if 1 == iter: print("V= ",V)
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            print("Number of iterations= ",iter)
            break
    return np.array(V)


def policy_eval_copy(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the 
    environment's dynamics. This version makes a copy of the V(Estimate of 
    the value function) and calculates all v before updating V, as in the book.
    The other version with stepwise updates of V outperforms in both computations
    and in convergence.
    
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
    """
    print("Policy (0=up, 1=right, 2=down, 3=left):")
    print(policy)
    print("")
    """
    # V = Estimate of the value function
    # Initialize V arbitrarily, except that V (terminal) = 0
    # In this implementation we initialize V to zeros 
    V = np.zeros(env.nS)
    iter = 0
    #printouts = 0
    #print_factor = 1
    while True:
        iter += 1
        # Loop over all states and perform an update
        v = np.zeros(env.nS)
        for s in range(env.nS):
            # Loop over all actions in each state
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton Barto eq. 4.5.
                    # Sum over all actions i each state    
                    if done:
                        v[s] += action_prob * prob * reward
                    else:
                        v[s] += action_prob * prob * (reward + discount_factor * V[next_state])
        """
        if 1 == iter :
            print("V= ",V)
            print("v= ",v)
        """
        delta = 0
        for s in range(env.nS):
            # Calculate How much our value function changed (across any states)
            delta = max(delta, abs(v[s] - V[s]))
        V = v
        
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            print("Number of iterations= ",iter)
            break
        #else:
        #    print("Delta= ",delta)
    return np.array(V)




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
            if done:
                action_values[a] += prob * reward
            else:
                action_values[a] += prob * (reward + discount_factor * V[next_state])

    return np.argmax(action_values)


def policy_improvement(env, policy_eval_fn=policy_eval, theta=0.00001, discount_factor=1.0):
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
        V = policy_eval_fn(policy, env, discount_factor, theta)
        """
        print("Reshaped Grid Value Function:")
        print(V.reshape(env.shape))
        print("")    
        """
        # Init policy_stable = True
        policy_stable = True
        
        # This is the policy improvment part
        # Loop over all states and update/improve the policy
        for s in range(env.nS):
            # The best action we could take in the current state 
            # under the current policy. Ref: Sutton Barto eq. 4.9.
            
            # Retrive the index of the action with the highest probability (greedy)
            current_action = np.argmax(policy[s]) 
            
            # Retrive(greedy) the index of the action that provides the higest value (one-step-lookahed)
            best_action = lookahead(env, s, V, discount_factor)   
            """
            if s in (3,7):
                print("Paus: Breakpoint") 
            if s in (4,5,8,9):
                print("Current action:", current_action,", ", policy[s][current_action], "Best action:", best_action, ", ", policy[s][best_action]) 
            """
            if current_action != best_action:
                policy_stable = False
            """
            for a, action_prob in enumerate(policy[s]):
                if a == best_action:
                    policy[s][a] = 1
                else:
                    policy[s][a] = 0
            """
            policy[s] = np.eye(env.nA)[best_action]
        
                
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V
"""
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()
policy, v = policy_improvement(env,policy_eval_copy, 1)
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
"""




