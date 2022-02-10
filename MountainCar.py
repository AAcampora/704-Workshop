import gym
from gym import spaces
import random
import time
import numpy as np
from gym.envs.registration import register
from IPython.display import clear_output
#create the enviroment

try:
    register(
        id='FrozenLakeNoSlip-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery':False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    pass

env_name = 'FrozenLake-v1'
env = gym.make(env_name)
type(env.action_space)


#The acting Agent
class Agent():
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete

        if self.is_discrete:
            self.action_size = env.action_space.n
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape



    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)

        return action


#QLearning
class QAgent(Agent):
    def __init__(self, env, discount_rate = 0.97, learning_rate = 0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("state Size", self.state_size)

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])
        print (self.q_table)

    def get_action(self, state):
        print('current state', state)
        q_state = self.q_table[state]
        action = np.argmax(q_state)
        return action

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

#generate the running enviroment
agent = QAgent(env)

total_reward = 0
for ep in range(100):
    state = env.reset()
    done = False
    #run the enviroment
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train((state, action, next_state, reward, done))
        state = next_state
        total_reward +=reward
        print("s:", state, "a:", action)
        print("Episode: {}", "Total reward: {}".format(ep, total_reward))
        env.render()
        time.sleep(0.05)
        clear_output(wait=True)