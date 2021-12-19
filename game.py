import gym
import numpy as np
import time

from session import ACTIONS, ENV_NAME, NUM_INPUTS
from model import Model


class RandomPlayer:
    def __init__(self) -> None:
        pass

    def predict(self, *args):
        return np.random.choice(ACTIONS), None


class Game:
    def __init__(self, seed: int = 42, env_name=ENV_NAME) -> None:
        # Configuration parameters for the whole setup
        self.num_actions = len(ACTIONS)
        self.env = gym.make(env_name)  # Create the environment
        self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def run_until_done(self, agent: Model, sleep=0.01):
        state = self.reset()
        count = 0

        while True:
            self.env.render()
            action = self.player(state, agent)
            state, reward, done, _ = self.step(action)
            count += 1

            time.sleep(sleep)

            if done:
                self.env.close()
                return count

    @staticmethod
    def player(x, model):
        y, _ = model.predict(x)
        y = np.squeeze(y)
        return np.argmax(y)
