#!/usr/bin/env python
# coding: utf-8


"""
## Implement Neural Fitted Q Iteration
Reinforcement learning with function approximators poses two critical challenges. 
Not only do we need to fit the function approximator to iteratively observed new data, 
we also need to make sure that the function approximator does not un-learn or forget 
in other parts of the state and actions space. 
This is particularly difficult for (large) neural network function approximators and 
for a long time these issues have limited reinforcement learning to small function 
approximator models with few parameters. 
For temporal-difference learning, a common and important approach to address these problems 
is storing and re-using transitions and this technique will be in the focus here.
In this task, you are asked to re-implement one of the early and seminal papers about 
using neural networks for temporal-difference learning. If you have not done so already, 
read the paper

Riedmiller, Martin. 
"Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method." 
In European Conference on Machine Learning, pp. 317-328. Springer, Berlin, Heidelberg, 2005.

Solution of Open AI gym environment "Cartpole-v0" 
(https://gym.openai.com/envs/CartPole-v0) using NFQ and Pytorch.:

1. Set up a gym environment with (discrete and) finite action space and 
continuous state space such as CartPole-v0 or MountainCar-v0 and 
make sure that you can control the apparatus in the gym correctly.

2. Implement the NFQ algorithm and make the necessary adaptions. 
(You might want to take inspiration from the paper and make some 
 adaptions to the reward function.) As the function approximator use a 
neural network model that you construct in PyTorch and be prepared 
to try different variants.

3. Train NFQ with the gym using the (low-dimensional) state representation 
provided by the gym. Conduct a thorough investigation with trying out 
different parameter settings and collecting and recording the necessary 
information from the learning process needed to determine what works, 
what does not work, and why. Share and discuss your findings in 
the discussion board if you have problems or if you find something interesting.

4. Optional - This step is optional but very helpful for the next homework assignment. 
Repeat step 3 but this time use the (high-dimensional) rendered visual 
representation as the state space. You will have to find out how to access 
the images for the next task anyway.

5. Submit your implementation for this assignment. Take notes and records of your results 
and bring them to the second session. Be prepared to present and explain your implementation, 
your findings, and your conclusions. Keep your notes and records for 
preparing the seminar presentation at the final session where you will be 
asked to compare to another algorithm.
"""

import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# hyper parameters
EPISODES = 300  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
#GAMMA = 0.95  # discount factor
GAMMA = 0.80  
LR = 0.0005  # NN optimizer learning rate
HIDDEN_LAYER = 5  # NN hidden layer size
BATCH_SIZE = 10  # Q-learning batch size
MEMORY_CAPACITY = 100000

# if gpu is to be used
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
#ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
#Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.i = 0

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.memory)

    def __next__(self):
        if self.i < len(self.memory):
            self.i += 1
            return self.memory[self.i-1]
        else:
            raise StopIteration()


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        #x = F.relu(self.l1(x))
        #x = F.relu(self.l2(x))
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = self.l3(x)
        return x
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                torch.nn.init.xavier_uniform(layer.weight)
                layer.bias.data.fill_(0.01)

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './tmp/cartpole-v0-1', force=True)
env.seed(500)
torch.manual_seed(500)

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(MEMORY_CAPACITY)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []

writer = SummaryWriter()

    
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(state).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        steps += 1
        environment.render()
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action.item())
        # zero reward when attempt ends
        if done:
            if (steps < 200):
                reward = -1
            else:
                reward = 0
            
        # Store the transition in memory
        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        # Move to the next state
        state = next_state
        #learn()
        #loss = optimize_model()
            
            
        if done:
            loss = optimize_model_2()
        
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            writer.add_scalar("Steps until finish:", steps, e)
            writer.add_scalar("Loss:", loss, e)
            episode_durations.append(steps)
            #plot_durations()
            break



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    # current Q values are estimated by NN for all actions
    q_current = model(batch_state)
    current_q_values = q_current.gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    q_next = model(batch_next_state)
    max_next_q_values = q_next.max(1)[0].detach()#max_next_q_values = q_next.detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def optimize_model_2():
    model.reset_parameters()
    for transition in memory:
        state, action, next_state, reward = transition
        # current Q values are estimated by NN for all actions
        q_current = model(state)
        print('q_current: ', q_current)
        print('q_current.gather(1, action): ', q_current.gather(1, action))
        current_q_values = q_current.gather(1, action)
        # expected Q values are estimated from actions which gives maximum Q value
        q_next = model(next_state)
        print('q_next:  ', q_next)
        print('q_next.max(1)[0].detach():  ', q_next.max(1)[0].detach())
        max_next_q_values = q_next.max(1)[0].detach()#max_next_q_values = q_next.detach().max(1)[0]
        expected_q_values = reward + (GAMMA * max_next_q_values)
    
        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        # backpropagation of loss to NN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

for e in range(EPISODES):
    run_episode(e, env)


print('Complete')
writer.flush()
print('Run(Anaconda prompt): tensorboard --logdir=runs')
print('Go to the URL it provides OR to http://localhost:6006/')
env.render(close=True)
env.close()
plt.ioff()
plt.show()
