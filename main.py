import gym
from stable_baselines3 import DQN

env = gym.make('CartPole-v0')

def run_environment(env, agent):
    observation = env.reset()
    for step in range(5000):
        action = agent( env, observation )
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print("ended after {} steps".format(step))
            obs = env.reset()
            return done


def random(env, obs):
    return env.action_space.sample()


def random_reward(obs,action, reward, new_obs):
    pass

class BaselineWrap(object):

    def __init__(self, ppo):
        self.ppo = ppo

    def predict(self, env, obs):
        action, _states = model.predict(obs, deterministic=True)
        return action

    def reward(self, state, action, reward, new_state):
        pass

# Train the model
print("Training...")
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.01)
model.learn(total_timesteps=10000,)

# wrap our model for use with our environment
print("evaluating...")
agent = BaselineWrap( model )
for episode in range(10):
    run_environment( env, agent.predict ) # <- we can pass bound methods in like functions (python = awesome)

# Save a model once trained
model = model.save("dqn_cartpole")
# Load a model from disk
model = DQN.load("dqn_cartpole")