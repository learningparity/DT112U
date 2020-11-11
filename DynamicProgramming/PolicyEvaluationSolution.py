import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.gridworld import GridworldEnv
env = GridworldEnv()

#from lib.envs.cliff_walking import CliffWalkingEnv
#env = CliffWalkingEnv()

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
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if 1 == iter :
            print("V= ",V)
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
                [(prob, next_state, reward, done)] = env.P[s][a]
                # Calculate the expected value. Ref: Sutton Barto eq. 4.5.
                # Sum over all actions i each state    
                if done:
                    v[s] += action_prob * prob * reward
                    #print("-- -- -- Done:", done, "Current state:", s, "Next state:", next_state)
                    #print("action_prob:", action_prob, "prob:", prob, "Reward:", reward, "V[next_state]", V[next_state], "v[s]=", v[s] )
                else:
                    v[s] += action_prob * prob * (reward + discount_factor * V[next_state])
                    #print("Normal case:","Current state:", s, "Next state:", next_state, "action_prob=", action_prob, "prob=", prob, "Reward=", reward, "V[next_state]=", V[next_state], "v[s]=", v[s] )
        if 1 == iter :
            print("V= ",V)
            print("v= ",v)
        
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



equal_dist_policy = np.ones([env.nS, env.nA]) / env.nA
print(equal_dist_policy)
#zeros_policy = np.zeros([env.nS, env.nA]) 
#print(zeros_policy)
# Zeros will not work as all the action_probabilities be zero 


#Discount factor
df = 1.0
theta = 0.001
v = policy_eval(equal_dist_policy, env, df, theta)

print("Value Function:")
print(v)
print("")
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test: Make sure the evaluated policy is what we expected
# Gridworld
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


v = policy_eval_copy(equal_dist_policy, env, df, theta)

print("Value Function:")
print(v)
print("")
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
# Test: Make sure the evaluated policy is what we expected
# Gridworld
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


