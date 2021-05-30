from datetime import datetime
import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# hyper parameters
EPISODES = 200  # number of episodes
BATCH_SIZE = 32  # Q-learning batch size
REPLAY_START_SIZE = 100
GAMMA = 0.9  # Q-learning discount factor
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
TARGET_UPDATE = 10
LR = 0.0005  # NN optimizer learning rate
HIDDEN_LAYER = 16  # NN hidden layer size

C_TRANS = 0.01
THRESHOLD_CART_POS_TARGET = 0.05
THRESHOLD_POLE_ANGLE_TARGET = 0.05
BOUNDARY_CART_POS = 2.4
BOUNDARY_CART_VELOCITY = 1
BOUNDARY_POLE_ANGLE = 0.2
BOUNDARY_ANGLE_RATE = 3.5
        

# if gpu is to be used
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './tmp/cartpole-v0-1', force=True)

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []

writer = SummaryWriter()
run_time = datetime.now().strftime("%y%m%d_%H%M_")
run_name = 'dqn.py_AdamLR=0.001_' + run_time
#writer = SummaryWriter('DQN/runs', comment=run_name)
#writer.add_custom_scalars_multilinechart(['loss/train', 'loss/test'], title='losses')

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(state).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

def greedy_action(state):
    return model(state).data.max(1)[1].view(1, 1)


######################################################################
# Training loop
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
#def learn():
    #if len(memory) < BATCH_SIZE:
    if len(memory) < REPLAY_START_SIZE:
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
    #loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def run_episode(e, environment):
    global successful_runs
    if e == 0: successful_runs = 0
    state = environment.reset()
    steps = 0
    while True:
        steps += 1
        environment.render()
        if 9 < successful_runs:
            action = greedy_action(FloatTensor([state]))
        else: 
            action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action.item())
        
        cart_pos = next_state[0]         # [-2.4, 2.4]
        cart_velocity = next_state[1]    # [-1, 1]
        pole_angle = next_state[2]       # [-0.2, 0.2]
        angle_rate = next_state[3]       # [-3.5, 3.5]
        if (abs(cart_pos) > BOUNDARY_CART_POS) or \
            (abs(pole_angle) > BOUNDARY_POLE_ANGLE) or \
            (abs(cart_velocity) > BOUNDARY_CART_VELOCITY) or \
            (abs(angle_rate) > BOUNDARY_ANGLE_RATE):
            reward = -1  # S-
        elif abs(cart_pos) < THRESHOLD_CART_POS_TARGET and abs(pole_angle) < THRESHOLD_POLE_ANGLE_TARGET:
            if done:
                reward = 1  # S+
            else:
                reward = 0  # S+
        else:
            reward = -C_TRANS
        if done and (200 <= steps):
            successful_runs += 1
            if successful_runs == 10:
                print("---- 10 succesful runs ----")


        # zero reward when attempt ends
        #if done and (steps < 200):
        #    reward = 0

        # Store the transition in memory
        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        
        #learn()
        loss = optimize_model()
        
        # Move to the next state
        state = next_state

        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            writer.add_scalar("Steps until finish:", steps, e)
            if loss is not None:
                writer.add_scalar("Loss:", loss, e)
            episode_durations.append(steps)
            #plot_durations()
            break




for e in range(EPISODES):
    run_episode(e, env)


print('Complete')
writer.flush()
print('Run(Anaconda prompt): tensorboard --logdir=runs')
print('Go to the URL it provides OR to http://localhost:6006/')
env.render()
env.close()
